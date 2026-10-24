"""Offline resource regressions at the provider/tool orchestration boundary."""
import json
from pathlib import Path
import tempfile
import unittest
from agent import Agent, BudgetExceeded, shrink
from agent_config import Config
from test_agent import Call, ScriptedProvider, reply


class AgentFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / 'task.md'
        self.task.write_text('Regression; minimize RMSE; write submission.csv.', encoding='utf-8')

    def runner(self, provider=None, **config):
        return Agent(self.task, provider or ScriptedProvider(), Config(
            total_seconds=60, reserve_seconds=5, **config), clock=lambda: 0.0)


class AgentImprovementTests(AgentFixture):
    def test_output_bound_is_utf8_bytes_and_preserves_both_ends(self):
        text = 'Начало ' + 'данные ' * 3000 + ' Итоговая метрика 0.71'
        compact = shrink(text, 4000)
        self.assertLessEqual(len(compact.encode('utf-8')), 4000)
        self.assertTrue(compact.startswith('Начало '))
        self.assertTrue(compact.endswith('Итоговая метрика 0.71'))
        self.assertIn('truncated', compact)
        self.assertLessEqual(len(shrink('й' * 100, 3).encode('utf-8')), 3)

    def test_executed_long_code_is_reusable_without_retransmitting_source(self):
        code = "print('score 0.75')\n" + '# reproducible feature recipe\n' * 3000
        provider = ScriptedProvider(reply(Call('run_python', {'code': code})), reply(content='done'))
        runner = self.runner(provider, max_steps=2)
        runner.run()
        self.assertEqual(len(provider.requests), 2)
        history = provider.requests[1]['messages']
        call = next(message for message in history if message['role'] == 'assistant')['tool_calls'][0]
        self.assertNotIn('reproducible feature recipe', call['function']['arguments'])
        tool_result = json.loads(next(message for message in history if message['role'] == 'tool')['content'])
        source = self.root / tool_result['experiment_path']
        self.assertEqual(source.read_text(encoding='utf-8'), code)
        self.assertIn('score 0.75', tool_result['stdout'])
        self.assertLess(len(json.dumps(history)), len(code) // 2)

    def test_all_executed_sources_are_compacted_in_multi_tool_exchange(self):
        code = "print('ok')\n" + '# feature\n' * 200
        provider = ScriptedProvider(reply(Call('run_python', {'code': code}, 1),
                                          Call('run_python', {'code': code}, 2)), reply(content='done'))
        self.runner(provider, max_steps=2).run()
        assistant = next(message for message in provider.requests[1]['messages']
                         if message['role'] == 'assistant')
        for call in assistant['tool_calls']:
            self.assertIn('Executed source saved at', call['function']['arguments'])
            self.assertNotIn('# feature', call['function']['arguments'])

    def test_pruned_history_retains_native_settings_and_honest_score(self):
        runner = self.runner()
        native = {'exit_code': 0, 'response': {'model': 'catboost', 'score': .71,
                  'metric': 'macro_f1', 'validation_id': 'holdout42',
                  'candidate_path': 'trained.csv', 'fit_scope': 'holdout_train',
                  'validation': {'strategy': 'group', 'rationale': 'Duplicated entities',
                                 'diagnostics': {'group_columns': ['x'] * 5000}}},
                  'registration': {'accepted': True}}
        runner._remember_experiment('run_backend', {'request': {'action': 'experiment', 'depth': 7}}, native)
        runner.messages += [{'role': 'assistant', 'content': 'old experiment'},
                            {'role': 'tool', 'tool_call_id': 'old', 'content': 'old result'},
                            {'role': 'assistant', 'content': 'newest exchange'}]
        self.assertTrue(runner._discard_oldest_exchange())
        payload, _ = runner._request_payload()
        status = json.loads(payload[-1]['content'])
        experiment = status['recent_experiments'][0]
        self.assertEqual(json.loads(experiment['request'])['depth'], 7)
        self.assertEqual(experiment['score'], .71)
        self.assertEqual(experiment['validation_id'], 'holdout42')
        self.assertEqual(experiment['fit_scope'], 'holdout_train')
        self.assertEqual(experiment['validation'], {'strategy': 'group',
                                                   'rationale': 'Duplicated entities'})
        self.assertNotIn('old result', json.dumps(payload))
        for index in range(50):
            runner._remember_experiment('run_backend', {'request': {'depth': index}}, native)
        self.assertLessEqual(len(runner.experiments), 8)
        self.assertLessEqual(len(json.dumps(runner.experiments).encode()), 6000)

    def test_remaining_budget_allows_short_completion_with_full_reservation(self):
        provider = ScriptedProvider(reply(content='finish'))
        runner = self.runner(provider)
        _, input_bound = runner._request_payload()
        # Exact 700-token headroom is enough for a final tool call, but not 4096.
        runner.config = Config(total_seconds=60, reserve_seconds=5,
                               max_tokens=input_bound + 700)
        runner._request()
        request = provider.requests[0]
        self.assertEqual(request['max_tokens'], 700)
        self.assertEqual(runner.tokens, 50)  # Exact provider usage refunds the reservation.

    def test_ambiguous_failure_keeps_full_reduced_reservation_and_no_overspend(self):
        error = RuntimeError('server error')
        error.status_code = 500
        provider = ScriptedProvider(error, reply(content='must not be called'))
        runner = self.runner(provider)
        _, input_bound = runner._request_payload()
        runner.config = Config(total_seconds=60, reserve_seconds=5,
                               max_tokens=input_bound + 700, retries=1)
        runner.sleep = lambda _: None
        with self.assertRaises(BudgetExceeded):
            runner._request()
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(runner.tokens, runner.config.max_tokens)


if __name__ == '__main__':
    unittest.main()
