"""Final startup and artifact publication regressions, without network calls."""
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from artifacts import ArtifactError, ArtifactStore
import solution


class StartupTests(unittest.TestCase):
    def invoke(self, environment, args=None):
        factory = MagicMock()
        agent = MagicMock()
        agent.return_value.run.return_value = {"success": True}
        with patch.dict(os.environ, environment, clear=True), patch.dict(
            "sys.modules", {"openai": types.SimpleNamespace(OpenAI=factory)}
        ), patch.object(solution, "Agent", agent), patch("sys.stdout", new_callable=io.StringIO), patch(
            "sys.stderr", new_callable=io.StringIO
        ):
            code = solution.main(args or [])
        return code, factory, agent

    def test_credentials_and_proxy_are_both_required_before_client_creation(self):
        for environment in ({}, {"OPENAI_API_KEY": "secret"},
                            {"OPENAI_BASE_URL": "https://proxy.invalid/v1"},
                            {"OPENAI_API_KEY": "  ", "OPENAI_BASE_URL": "https://proxy.invalid/v1"},
                            {"OPENAI_API_KEY": "secret", "OPENAI_BASE_URL": "  "}):
            with self.subTest(keys=list(environment)):
                code, factory, agent = self.invoke(environment)
                self.assertEqual(code, 2)
                factory.assert_not_called()
                agent.assert_not_called()

    def test_explicit_proxy_and_optional_default_task(self):
        code, factory, agent = self.invoke({
            "OPENAI_API_KEY": "secret", "OPENAI_BASE_URL": "https://proxy.invalid/v1"
        })
        self.assertEqual(code, 0)
        self.assertEqual(factory.call_args.kwargs["base_url"], "https://proxy.invalid/v1")
        self.assertEqual(agent.call_args.args[0], Path("task.md"))
        agent.return_value.run.assert_called_once_with(bootstrap=True)

    def test_explicit_task_path_is_forwarded(self):
        code, _, agent = self.invoke({
            "OPENAI_API_KEY": "secret", "OPENAI_BASE_URL": "https://proxy.invalid/v1"
        }, ["data/task.md"])
        self.assertEqual(code, 0)
        self.assertEqual(agent.call_args.args[0], Path("data/task.md"))


class PublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / "task"
        self.task.mkdir()
        candidate = self.task / "candidate.csv"
        candidate.write_text("prediction\n1\n", encoding="utf-8")
        self.store = ArtifactStore(self.task, self.root / "state")
        self.store.consider(candidate)

    def test_output_and_parent_symlinks_rejected_before_resolution(self):
        # Simulate link metadata as Windows often lacks symlink creation privilege.
        for name, linked in (("submission.csv", "submission.csv"),
                             ("linked/submission.csv", "linked")):
            with self.subTest(name=name), patch.object(
                Path, "is_symlink", lambda path: path == self.task / linked
            ), self.assertRaises(ArtifactError):
                self.store.publish(name)
        self.assertFalse((self.task / "submission.csv").exists())

    def test_real_symlink_does_not_replace_its_input_target(self):
        source = self.task / "train.csv"
        source.write_text("input\nkeep\n", encoding="utf-8")
        try:
            (self.task / "submission.csv").symlink_to(source)
        except OSError:
            self.skipTest("Creating symlinks requires an OS privilege")
        with self.assertRaises(ArtifactError):
            self.store.publish()
        self.assertEqual(source.read_text(), "input\nkeep\n")

    def test_traversal_rejected_and_regular_nested_output_published(self):
        with self.assertRaises(ArtifactError):
            self.store.publish("../outside.csv")
        destination = self.store.publish("outputs/final.csv")
        self.assertEqual(destination.read_text(), "prediction\n1\n")
        destination.write_text("old\n2\n", encoding="utf-8")
        self.assertEqual(self.store.publish("outputs/final.csv").read_text(), "prediction\n1\n")


if __name__ == "__main__":
    unittest.main()
