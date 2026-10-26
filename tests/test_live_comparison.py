"""No-network tests of the shared real-API meter and original-source adapter."""
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from qa.live_baseline_comparison import BudgetLimit, MeteredClient, baseline_parts


class LiveComparisonTests(unittest.TestCase):
    def remote(self, result=None, error=None):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            if error:
                raise error
            return result or SimpleNamespace(usage=SimpleNamespace(total_tokens=20), choices=[
                SimpleNamespace(message=SimpleNamespace(tool_calls=[], content='ok'))])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls

    def test_reserves_before_send_and_refunds_reported_usage(self):
        remote, calls = self.remote()
        observed = []
        client = MeteredClient(remote, 5000, lambda c: observed.append(c.spent), time.monotonic() + 10)
        client.create(messages=[], tools=[], model='fake', max_tokens=100)
        self.assertEqual(len(calls), 1)
        self.assertGreater(observed[0], 20)
        self.assertEqual(observed[-1], 20)
        self.assertEqual(client.spent, 20)

    def test_budget_denies_before_any_network_and_attempts_are_bounded(self):
        remote, calls = self.remote()
        client = MeteredClient(remote, 1, lambda _: None, time.monotonic() + 10)
        with self.assertRaises(BudgetLimit):
            client.create(messages=[], tools=[], model='fake')
        self.assertFalse(calls)
        client = MeteredClient(remote, 15000, lambda _: None, time.monotonic() + 10)
        for _ in range(6):
            client.create(messages=[], tools=[], model='fake', max_tokens=100)
        with self.assertRaises(BudgetLimit):
            client.create(messages=[], tools=[], model='fake', max_tokens=100)
        self.assertEqual(len(calls), 6)

    def test_ambiguous_error_keeps_reservation_and_redacts_exception_body(self):
        remote, _ = self.remote(error=RuntimeError('private-secret-value'))
        client = MeteredClient(remote, 5000, lambda _: None, time.monotonic() + 10)
        with self.assertRaises(RuntimeError) as caught:
            client.create(messages=[], tools=[], model='fake', max_tokens=100)
        self.assertNotIn('private-secret-value', str(caught.exception))
        self.assertEqual(client.events[0]['status'], 'error')
        self.assertEqual(client.spent, client.events[0]['reserved'])

    def test_original_source_extraction_does_not_execute_top_level_imports(self):
        source = '''import missing_package
def run_python(code: str):
    """Run code."""
    return code
async def run_agent(task_path):
    messages = [{"role":"system", "content":"original prompt"}, {"role":"user", "content":task}]
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'baseline.py'
            path.write_text(source)
            prompt, tools, execute = baseline_parts(path, Path(directory), time.monotonic() + 10)
        self.assertEqual(prompt, 'original prompt')
        self.assertEqual(tools[0]['function']['name'], 'run_python')
        self.assertEqual(execute('print(1)'), 'print(1)')


if __name__ == '__main__':
    unittest.main()
