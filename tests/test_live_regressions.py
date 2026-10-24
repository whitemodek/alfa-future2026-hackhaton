"""Offline checks for native candidates surviving provider budget exhaustion."""
import csv
from pathlib import Path
import tempfile
import unittest

from agent import Agent
from agent_config import Config
from test_agent import Call, ScriptedProvider, reply


class NativeCandidateRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / "task.md"
        self.task.write_text(
            "Regression. Target: target. Metric RMSE. Read train.csv and test.csv. "
            "Write submission.csv matching sample_submission.csv.", encoding="utf-8")
        (self.root / "train.csv").write_text(
            "id,x,target\n" + "".join(f"{i},{i-80},{(i-80)*2}\n" for i in range(160)),
            encoding="utf-8")
        (self.root / "test.csv").write_text("id,x\na,-10\nb,0\nc,10\n", encoding="utf-8")
        (self.root / "sample_submission.csv").write_text(
            "id,target\na,0\nb,0\nc,0\n", encoding="utf-8")

    def runner(self, action, **request):
        provider = ScriptedProvider(reply(Call("run_backend", {
            "request": {"action": action, **request}, "timeout_seconds": 15})))
        runner = Agent(self.task, provider, Config(
            total_seconds=90, reserve_seconds=5, call_seconds=20,
            max_api_attempts=1, max_steps=3, warmup_seconds=0, threads=1))
        return runner, provider

    def test_native_backend_candidate_survives_next_api_budget_stop(self):
        runner, provider = self.runner("baseline")
        result = runner.run()
        self.assertEqual(result["reason"], "BudgetExceeded")
        self.assertEqual(len(provider.requests), 1)
        self.assertTrue(result["success"], result)
        self.assertEqual(runner.store.best["rows"], 3)
        self.assertTrue((self.root / "submission.csv").is_file())

    def test_improved_native_candidate_replaces_bootstrap_before_api_stop(self):
        runner, provider = self.runner("experiment", iterations=100, depth=4)
        result = runner.run(bootstrap=True)
        self.assertEqual(result["reason"], "BudgetExceeded")
        self.assertEqual(len(provider.requests), 1)
        self.assertTrue(result["success"], result)
        # A startup dummy has RMSE around 100 and constant test predictions;
        # real native boosting easily improves this separable synthetic task.
        self.assertLess(runner.store.best["score"], 25)
        with (self.root / "submission.csv").open(newline="") as stream:
            values = [float(row["target"]) for row in csv.DictReader(stream)]
        self.assertGreater(values[-1] - values[0], 20)

    def test_token_budget_drops_old_large_exchange_before_stopping(self):
        noisy = reply(Call("run_python", {"code":
            "from pathlib import Path\n"
            "Path('candidate.csv').write_text('id,target\\na,1\\nb,2\\nc,3\\n')\n"
            "print('x' * 12000)"}))
        noisy.usage.total_tokens = 1000
        register = reply(
            Call("configure_submission", {"columns": ["id", "target"],
                 "expected_rows": 3, "id_columns": ["id"], "numeric_columns": ["target"]}),
            Call("register_candidate", {"candidate_path": "candidate.csv"}, 2))
        register.usage.total_tokens = 10000
        provider = ScriptedProvider(noisy, register, reply(Call("finish", {})))
        runner = Agent(self.task, provider, Config(
            total_seconds=60, reserve_seconds=5, max_steps=4,
            max_tokens=30000, response_tokens=2048, tool_output_bytes=12000))
        result = runner.run()
        self.assertTrue(result["success"], result)
        self.assertEqual(result["reason"], "completed")
        self.assertEqual(len(provider.requests), 3)
        latest = provider.requests[-1]["messages"]
        self.assertEqual(latest[0]["role"], "system")
        self.assertIn("Regression", latest[1]["content"])
        # Preserve the newest schema/candidate result, trim the older noisy call.
        self.assertTrue(any(message["role"] == "tool" and '"accepted": true' in
                            message["content"] for message in latest))
        self.assertFalse(any("x" * 1000 in str(message) for message in latest))


if __name__ == "__main__":
    unittest.main()
