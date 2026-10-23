"""Backend behavior at Agent.run and the real solution.py/provider HTTP boundary."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent import Agent
from agent_config import Config


class Call:
    def __init__(self, name, arguments, number=1):
        self.id = f"call_{number}"
        self.function = SimpleNamespace(name=name, arguments=(
            arguments if isinstance(arguments, str) else json.dumps(arguments)))

    def model_dump(self, **kwargs):
        return {"id": self.id, "type": "function", "function": vars(self.function)}


def reply(*calls, content=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=content, tool_calls=list(calls)))], usage=SimpleNamespace(total_tokens=50))


class ScriptedProvider:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return item


SCHEMA = {"columns": ["id", "prediction"], "expected_rows": 2,
          "id_columns": ["id"], "numeric_columns": ["prediction"]}
GOOD = "id,prediction\na,1\nb,2\n"


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = self.root / "task.md"
        self.task.write_text("Return two predictions, columns id,prediction.", encoding="utf-8")
        (self.root / "candidate.csv").write_text(GOOD, encoding="utf-8")

    def run_script(self, *responses, **config):
        provider = ScriptedProvider(*responses)
        runner = Agent(self.task, provider, Config(**{
            "max_steps": 8, "total_seconds": 30, "reserve_seconds": 5,
            "call_seconds": 5, **config}), sleep=lambda _: None)
        return runner.run(), provider

    def start(self):
        return reply(Call("configure_submission", SCHEMA),
                     Call("register_candidate", {"candidate_path": "candidate.csv"}, 2))

    def test_successful_tools_create_and_publish_submission(self):
        result, _ = self.run_script(
            reply(Call("configure_submission", SCHEMA)),
            reply(Call("run_python", {"code": "from pathlib import Path\nPath('created.csv').write_text('id,prediction\\na,7\\nb,8\\n')"})),
            reply(Call("register_candidate", {"candidate_path": "created.csv"})),
            reply(Call("finish", {})))
        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "completed")
        self.assertEqual(Path(result["submission"]).read_text(), "id,prediction\na,7\nb,8\n")

    def test_provider_failure_recovers_best_and_omits_secret_error_body(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sensitive-test-token"}):
            result, _ = self.run_script(self.start(), RuntimeError("sensitive-test-token"))
        self.assertTrue(result["success"])
        self.assertEqual(Path(result["submission"]).read_text(), GOOD)
        journal = (Path(result["state_dir"]) / "events.jsonl").read_text()
        self.assertNotIn("sensitive-test-token", journal)

    def test_invalid_tool_calls_are_reported_then_recovery_works(self):
        result, provider = self.run_script(
            reply(Call("configure_submission", "{")),
            reply(Call("run_python", {"code": "raise AssertionError", "unknown": 1})),
            reply(Call("register_candidate", {"candidate_path": "candidate.csv"})),
            self.start(), reply(Call("finish", {})))
        self.assertTrue(result["success"])
        errors = [json.loads(message["content"]) for message in provider.requests[-1]["messages"]
                  if message["role"] == "tool" and "error" in json.loads(message["content"])]
        self.assertEqual(len(errors), 3)
        self.assertTrue(any("configure_submission first" in item["error"] for item in errors))

    def test_step_limit_preserves_best(self):
        result, _ = self.run_script(self.start(), max_steps=1)
        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "step_limit")

    def test_plain_text_without_candidate_is_not_success(self):
        result, _ = self.run_script(reply(content="Done"), max_steps=1)
        self.assertFalse(result["success"])
        self.assertFalse((self.root / "submission.csv").exists())

    def test_invalid_later_file_and_schema_change_preserve_snapshot(self):
        result, _ = self.run_script(
            self.start(),
            reply(Call("configure_submission", {"columns": ["wrong"], "expected_rows": 99})),
            reply(Call("run_python", {"code": "from pathlib import Path\nPath('candidate.csv').write_text('id,prediction\\na,nan\\nb,2\\n')"})),
            reply(Call("register_candidate", {"candidate_path": "candidate.csv"})),
            reply(Call("finish", {})))
        self.assertTrue(result["success"])
        self.assertEqual(Path(result["submission"]).read_text(), GOOD)

    def test_child_does_not_receive_provider_credential(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "private-credential"}):
            result, provider = self.run_script(
                reply(Call("run_python", {"code": "import os\nprint('key_present', 'OPENAI_API_KEY' in os.environ)"})),
                self.start(), reply(Call("finish", {})))
        outputs = [json.loads(m["content"]) for m in provider.requests[1]["messages"] if m["role"] == "tool"]
        self.assertIn("key_present False", outputs[0]["stdout"])
        self.assertTrue(result["success"])

    def test_api_retry_count_is_bounded(self):
        error = RuntimeError("rate limited")
        error.status_code = 429
        result, provider = self.run_script(error, error, error, retries=2)
        self.assertFalse(result["success"])
        self.assertEqual(len(provider.requests), 3)

    def test_auth_failure_is_not_retried(self):
        error = RuntimeError("authentication failed")
        error.status_code = 401
        result, provider = self.run_script(error)
        self.assertFalse(result["success"])
        self.assertEqual(len(provider.requests), 1)

    def test_token_budget_prevents_network_call(self):
        result, provider = self.run_script(max_tokens=1)
        self.assertFalse(result["success"])
        self.assertEqual(provider.requests, [])

    def test_time_exhaustion_publishes_registered_result(self):
        now = [0.0]
        provider = ScriptedProvider(self.start(), reply(Call("run_python", {"code": "raise AssertionError('must not start')"})))
        original_create = provider.create

        def expire_after_response(**kwargs):
            response = original_create(**kwargs)
            if len(provider.requests) == 2:
                now[0] = 40.0
            return response

        provider.chat = SimpleNamespace(completions=SimpleNamespace(create=expire_after_response))
        result = Agent(self.task, provider, Config(total_seconds=30, reserve_seconds=5),
                       clock=lambda: now[0]).run()
        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "BudgetExceeded")
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(Path(result["submission"]).read_text(), GOOD)

    def test_weaker_score_does_not_replace_best(self):
        (self.root / "weaker.csv").write_text("id,prediction\na,9\nb,9\n", encoding="utf-8")
        result, _ = self.run_script(
            reply(Call("configure_submission", SCHEMA)),
            reply(Call("register_candidate", {"candidate_path": "candidate.csv", "score": 0.8,
                  "metric": "macro_f1", "direction": "maximize", "validation_id": "holdout-42"})),
            reply(Call("register_candidate", {"candidate_path": "weaker.csv", "score": 0.7,
                  "metric": "macro_f1", "direction": "maximize", "validation_id": "holdout-42"})),
            reply(Call("finish", {})))
        self.assertTrue(result["success"])
        self.assertEqual(Path(result["submission"]).read_text(), GOOD)


class CliHTTPTests(unittest.TestCase):
    def test_real_sdk_tool_roundtrip_from_unrelated_directory(self):
        requests = []
        responses = [
            [Call("configure_submission", SCHEMA).model_dump()],
            [Call("register_candidate", {"candidate_path": "candidate.csv"}).model_dump()],
            [Call("finish", {}).model_dump()],
        ]

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                calls = responses.pop(0) if responses else []
                body = json.dumps({"id": "chatcmpl-test", "object": "chat.completion",
                    "created": 1, "model": "local-test",
                    "choices": [{"index": 0, "finish_reason": "tool_calls",
                                 "message": {"role": "assistant", "content": None, "tool_calls": calls}}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                task_dir = root / "data with spaces"
                task_dir.mkdir()
                task = task_dir / "task.md"
                task.write_text("Write the provided candidate as submission.", encoding="utf-8")
                (task_dir / "candidate.csv").write_text(GOOD, encoding="utf-8")
                environment = dict(os.environ)
                environment.update(OPENAI_API_KEY="local-fake-key",
                    OPENAI_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
                    OPENAI_MODEL="local-test", AGENT_MAX_STEPS="4")
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "solution.py"), str(task)],
                    cwd=root, env=environment, capture_output=True, text=True, timeout=25)
                self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
                result = json.loads(completed.stdout)
                self.assertTrue(result["success"])
                self.assertEqual((task_dir / "submission.csv").read_text(), GOOD)
                self.assertFalse((root / "submission.csv").exists())
                self.assertEqual(len(requests), 3)
                self.assertEqual(requests[0]["model"], "local-test")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
