"""Offline end-to-end API/CLI tests; no external service or ML training."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import threading
import unittest
import zipfile

from pack_solution import build, RUNTIME_FILES


ROOT = Path(__file__).resolve().parents[1]


def tool_call(name, arguments):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_" + name, "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments)}}]}


@contextmanager
def fake_api(responses):
    requests = []
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                index = len(requests) - 1
                item = responses[index] if index < len(responses) else 500
                if isinstance(item, int):
                    status = item
                    payload = {"error": {"message": "offline simulated outage", "type": "server_error"}}
                else:
                    status = 200
                    payload = {"id": f"chatcmpl-{index}", "object": "chat.completion",
                               "created": 1, "model": "offline-model", "choices": [
                                   {"index": 0, "message": item, "finish_reason": "tool_calls"}],
                               "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                                         "total_tokens": 150}}
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except Exception as exc:
                errors.append(exc)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if errors:
            raise AssertionError(f"Fake API handler errors: {errors}")


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.task_dir = Path(self.temporary.name) / "task with spaces"
        self.task_dir.mkdir()
        self.task = self.task_dir / "task.md"
        self.task.write_text("Write CSV columns id,prediction, two rows, preserving IDs a,b.", encoding="utf-8")

    def invoke(self, url, solution=None):
        environment = {k: v for k, v in os.environ.items()
                       if not k.startswith("AGENT_") and not k.upper().endswith("_PROXY")}
        environment.update(OPENAI_API_KEY="offline-only-not-a-real-secret", OPENAI_BASE_URL=url,
                           OPENAI_MODEL="offline-model", AGENT_TOTAL_SECONDS="30",
                           AGENT_RESERVE_SECONDS="5", AGENT_CALL_SECONDS="5",
                           AGENT_API_SECONDS="5", AGENT_MAX_STEPS="8", AGENT_RETRIES="0",
                           NO_PROXY="127.0.0.1,localhost")
        return subprocess.run([sys.executable, str(solution or ROOT / "solution.py"), str(self.task)],
                              cwd=self.task_dir.parent, env=environment, capture_output=True,
                              text=True, timeout=45)

    def successful_steps(self):
        return [tool_call("configure_submission", {
                    "columns": ["id", "prediction"], "expected_rows": 2,
                    "id_columns": ["id"], "numeric_columns": ["prediction"]}),
                tool_call("run_python", {"code":
                    "from pathlib import Path\nPath('candidate.csv').write_text('id,prediction\\na,0.2\\nb,0.8\\n', encoding='utf-8')"}),
                tool_call("register_candidate", {"candidate_path": "candidate.csv"})]

    def assert_submission(self, completed):
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        result = json.loads(completed.stdout)
        self.assertTrue(result["success"])
        self.assertEqual(Path(result["submission"]), self.task_dir / "submission.csv")
        self.assertEqual((self.task_dir / "submission.csv").read_text(), "id,prediction\na,0.2\nb,0.8\n")
        return result

    def test_complete_tool_cycle_and_tool_error_recovery(self):
        steps = self.successful_steps()
        steps.insert(1, tool_call("run_python", {"code": "raise ValueError('expected experiment failure')"}))
        steps.append(tool_call("finish", {}))
        with fake_api(steps) as (url, requests):
            completed = self.invoke(url)
        result = self.assert_submission(completed)
        self.assertEqual(result["reason"], "completed")
        self.assertEqual(len(requests), 5)
        observations = [json.loads(m["content"]) for m in requests[2]["messages"] if m["role"] == "tool"]
        self.assertTrue(any(m.get("exit_code") == 1 for m in observations))
        for request in requests:
            self.assertEqual(request["model"], "offline-model")
            self.assertFalse(request["parallel_tool_calls"])
            self.assertEqual(request["tools"][0]["type"], "function")
        journal = (Path(result["state_dir"]) / "events.jsonl").read_text()
        self.assertNotIn("offline-only-not-a-real-secret", journal)

    def test_api_failure_publishes_registered_fallback(self):
        with fake_api(self.successful_steps() + [500]) as (url, requests):
            completed = self.invoke(url)
        result = self.assert_submission(completed)
        self.assertEqual(result["reason"], "InternalServerError")
        self.assertEqual(len(requests), 4)

    def test_failure_without_candidate_has_nonzero_exit(self):
        with fake_api([401]) as (url, requests):
            completed = self.invoke(url)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertFalse(json.loads(completed.stdout)["success"])
        self.assertFalse((self.task_dir / "submission.csv").exists())
        self.assertEqual(len(requests), 1)

    def test_extracted_zip_runs_with_mock_ml_plugin(self):
        source = Path(self.temporary.name) / 'source'
        source.mkdir()
        for name in RUNTIME_FILES:
            shutil.copy2(ROOT / name, source / name)
        shutil.copy2(ROOT / 'examples/mock_ml_backend.py', source / 'ml_backend.py')
        archive = Path(self.temporary.name) / 'solution.zip'
        result = build(archive, source)
        self.assertIn('ml_backend.py', result['files'])
        extracted = Path(self.temporary.name) / 'extracted'
        with zipfile.ZipFile(archive) as packaged:
            self.assertTrue(all('/' not in name for name in packaged.namelist()))
            packaged.extractall(extracted)
        steps = [tool_call('run_backend', {'request': {'action': 'smoke'}}),
                 self.successful_steps()[0],
                 tool_call('register_candidate', {'candidate_path': 'mock_candidate.csv'}),
                 tool_call('finish', {})]
        with fake_api(steps) as (url, requests):
            completed = self.invoke(url, extracted / 'solution.py')
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual((self.task_dir / 'submission.csv').read_text(), 'id,prediction\na,1.0\nb,2.0\n')
        backend = json.loads(next(m['content'] for m in requests[1]['messages'] if m['role'] == 'tool'))
        self.assertEqual(backend['response']['version'], 1)

    def test_packaging_rejects_source_overwrite(self):
        before = (ROOT / 'agent.py').read_bytes()
        with self.assertRaises(ValueError):
            build(ROOT / 'agent.py')
        self.assertEqual((ROOT / 'agent.py').read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
