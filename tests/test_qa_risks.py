"""QA: регрессионные тесты на найденные риски.

Каждый тест описывает ЖЕЛАЕМОЕ поведение. Падающий тест = открытый баг
(номер BUG-N совпадает с отчётом QA). После исправления тест должен позеленеть.
Запуск: python -m unittest tests.test_qa_risks -v
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from types import SimpleNamespace
import unittest

from agent import Agent
from agent_config import Config

ROOT = Path(__file__).resolve().parents[1]


class Call:
    def __init__(self, name, arguments, number=1):
        self.id = f"call_{number}"
        self.function = SimpleNamespace(name=name, arguments=(
            arguments if isinstance(arguments, str) else json.dumps(arguments)))

    def model_dump(self, **kwargs):
        return {"id": self.id, "type": "function", "function": vars(self.function)}


def reply(*calls, content=None, tokens=50):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=content, tool_calls=list(calls)))], usage=SimpleNamespace(total_tokens=tokens))


class Scripted:
    def __init__(self, *responses, on_call=None):
        self.responses = iter(responses)
        self.on_call = on_call
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        if self.on_call:
            self.on_call(self.calls, kwargs)
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return item


class RealisticTokens:
    """Каждый шаг — run_python с ~6 KB вывода; токены ~ байты/3.5 (как у реального BPE)."""

    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        prompt = len(json.dumps(kwargs["messages"], ensure_ascii=False).encode()) / 3.5
        code = "print('metric report ' * 300)\n" + "# feature engineering line\n" * 60
        return reply(Call("run_python", {"code": code}, self.calls),
                     content="reasoning " * 40, tokens=int(prompt) + 800)


SCHEMA = {"columns": ["id", "y"], "expected_rows": 2, "id_columns": ["id"], "numeric_columns": ["y"]}


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = self.root / "task.md"
        self.task.write_text("Regression, RMSE. Columns id,y.", encoding="utf-8")
        (self.root / "a.csv").write_text("id,y\n1,10\n2,10\n", encoding="utf-8")
        (self.root / "b.csv").write_text("id,y\n1,5\n2,5\n", encoding="utf-8")


class TokenBudgetTests(Base):
    def test_bug1_default_budget_allows_at_least_20_llm_steps(self):
        """BUG-1: при дефолтном max_tokens=200000 агент успевает лишь ~7-15 вызовов LLM."""
        provider = RealisticTokens()
        Agent(self.task, provider, Config(), sleep=lambda _: None).run()
        self.assertGreaterEqual(provider.calls, 20, f"only {provider.calls} LLM calls fit the budget")

    def test_check_cyrillic_task_is_not_overcounted(self):
        """CHECK (BUG-10, minor): длинное русское условие не должно заметно сокращать число шагов."""
        self.task.write_text("Определите тип кибератаки по признакам сетевого трафика. " * 250, encoding="utf-8")
        provider = RealisticTokens()
        Agent(self.task, provider, Config(), sleep=lambda _: None).run()
        english = Path(self.temp.name) / "en"
        english.mkdir()
        (english / "task.md").write_text("Determine the cyber attack type from traffic features.... " * 250)
        provider_en = RealisticTokens()
        Agent(english / "task.md", provider_en, Config(), sleep=lambda _: None).run()
        self.assertGreaterEqual(provider.calls, provider_en.calls * 0.8,
                                f"RU {provider.calls} vs EN {provider_en.calls} LLM calls")


class SelectionTests(Base):
    def test_bug3_rmse_without_direction_keeps_lower_error(self):
        """BUG-3: direction необязателен и по умолчанию maximize → для RMSE выбирается ХУДШАЯ модель."""
        agent = Agent(self.task, Scripted(), Config(total_seconds=60, reserve_seconds=5))
        agent.execute("configure_submission", SCHEMA)
        agent.execute("register_candidate", {"candidate_path": "a.csv", "score": 10.0,
                                             "metric": "rmse", "validation_id": "kfold5"})
        result = agent.execute("register_candidate", {"candidate_path": "b.csv", "score": 5.0,
                                                      "metric": "rmse", "validation_id": "kfold5"})
        self.assertTrue(result["accepted"] or "direction" in json.dumps(result),
                        "RMSE 5 rejected in favour of RMSE 10 without any error")


class CrashSafetyTests(Base):
    def test_bug4_submission_written_as_soon_as_candidate_registered(self):
        """BUG-4: submission.csv пишется только в конце run(); kill -9 на 3-м часу = нет файла."""
        seen = {}

        def check(number, _):
            if number == 2:
                seen["exists"] = (self.root / "submission.csv").exists()

        provider = Scripted(reply(Call("configure_submission", SCHEMA),
                                  Call("register_candidate", {"candidate_path": "a.csv"}, 2)),
                            reply(Call("finish", {})), on_call=check)
        Agent(self.task, provider, Config(total_seconds=60, reserve_seconds=5), sleep=lambda _: None).run()
        self.assertTrue(seen.get("exists"), "submission.csv absent after first accepted candidate")

    def test_bug5_sigterm_publishes_best(self):
        """BUG-5: SIGTERM от платформы не обрабатывается — лучший кандидат не публикуется."""
        script = textwrap.dedent(f"""
            import json, sys, time
            sys.path.insert(0, {str(ROOT)!r}); sys.path.insert(0, {str(Path(__file__).parent)!r})
            from test_qa_risks import Scripted, reply, Call, SCHEMA
            from agent import Agent
            from agent_config import Config
            def hang(n, _):
                if n == 2:
                    print('READY', flush=True); time.sleep(60)
            p = Scripted(reply(Call('configure_submission', SCHEMA),
                               Call('register_candidate', {{'candidate_path': 'a.csv'}}, 2)),
                         reply(Call('finish', {{}})), on_call=hang)
            Agent({str(self.task)!r}, p, Config(total_seconds=120, reserve_seconds=5)).run()
        """)
        process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        self.assertEqual(process.stdout.readline().strip(), "READY")
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)
        self.assertTrue((self.root / "submission.csv").exists(), "SIGTERM left no submission.csv")


class FallbackTests(Base):
    def test_bug6_llm_unavailable_still_produces_submission(self):
        """BUG-6: если LLM недоступна с первого запроса, нет никакого сабмита (нужен baseline без LLM)."""
        (self.root / "train.csv").write_text("id,x,y\n1,1,2\n2,2,4\n3,3,6\n", encoding="utf-8")
        (self.root / "test.csv").write_text("id,x\n10,4\n11,5\n", encoding="utf-8")
        (self.root / "sample_submission.csv").write_text("id,y\n10,0\n11,0\n", encoding="utf-8")
        error = RuntimeError("401 auth")
        result = Agent(self.task, Scripted(error), Config(total_seconds=60, reserve_seconds=5,
                       warmup_seconds=0), sleep=lambda _: None).run(bootstrap=True)
        self.assertTrue(result["success"], result)
        self.assertTrue((self.root / "submission.csv").exists())

    def test_bug7_qwen_text_tool_call_is_understood(self):
        """BUG-7: Qwen иногда кладёт вызов в content как <tool_call>{...}</tool_call> — он теряется."""
        text = '<tool_call>\n{"name": "configure_submission", "arguments": ' + json.dumps(SCHEMA) + '}\n</tool_call>'
        provider = Scripted(reply(content=text), reply(Call("register_candidate", {"candidate_path": "a.csv"})),
                            reply(Call("finish", {})))
        result = Agent(self.task, provider, Config(total_seconds=60, reserve_seconds=5),
                       sleep=lambda _: None).run()
        self.assertTrue(result["success"], result)


class TimingTests(Base):
    def test_bug8_final_refit_leaves_time_to_register(self):
        """BUG-8: в фазе finalize run_python получает всё оставшееся время минус 2 с — refit не успеть зарегистрировать."""
        now = [0.0]
        agent = Agent(self.task, Scripted(), Config(total_seconds=10500, reserve_seconds=900),
                      clock=lambda: now[0])
        now[0] = 10500 - 600  # осталось 10 минут, фаза finalize
        timeout = agent._timeout(1800)
        self.assertLessEqual(timeout, 600 - 120, f"refit may use {timeout:.0f}s of last 600s")


class BaselineInterfaceTests(Base):
    """Сверка с baseline.py: каталог задачи доступен на запись (BUG-9 снят), имя ответа задаёт условие."""

    def test_output_name_from_task_is_also_published(self):
        self.task.write_text("Сохраните предсказания в predictions.csv (формат как sample_submission.csv).",
                             encoding="utf-8")
        provider = Scripted(reply(Call("configure_submission", SCHEMA),
                                  Call("register_candidate", {"candidate_path": "a.csv"}, 2)),
                            reply(Call("finish", {})))
        Agent(self.task, provider, Config(total_seconds=60, reserve_seconds=5), sleep=lambda _: None).run()
        self.assertTrue((self.root / "predictions.csv").exists())
        self.assertTrue((self.root / "submission.csv").exists())

    def test_output_path_cannot_overwrite_inputs(self):
        (self.root / "train.csv").write_text("id,x,y\n1,1,1\n", encoding="utf-8")
        agent = Agent(self.task, Scripted(), Config(total_seconds=60, reserve_seconds=5))
        with self.assertRaises(ValueError):
            agent.execute("configure_submission", {**SCHEMA, "output_path": "train.csv"})

    def test_solution_without_arguments_uses_task_md(self):
        completed = subprocess.run([sys.executable, str(ROOT / "solution.py")], cwd=self.root,
                                   capture_output=True, text=True, env={"PATH": os.defpath})
        self.assertNotIn("Usage", completed.stderr)

    def test_llm_schema_change_and_direction_inference(self):
        (self.root / "train.csv").write_text("id,x,y\n1,1,2\n2,2,4\n3,3,6\n", encoding="utf-8")
        (self.root / "test.csv").write_text("id,x\n10,4\n11,5\n", encoding="utf-8")
        (self.root / "sample_submission.csv").write_text("id,y\n10,0\n11,0\n", encoding="utf-8")
        (self.root / "good.csv").write_text("id,y\n10,8\n11,10\n", encoding="utf-8")
        provider = Scripted(
            reply(Call("configure_submission", {"columns": ["id", "WRONG"], "expected_rows": 2})),
            reply(Call("configure_submission", {"sample_path": "sample_submission.csv", "id_columns": ["id"],
                                                "numeric_columns": ["y"]})),
            reply(Call("register_candidate", {"candidate_path": "good.csv", "score": 0.1, "metric": "rmse",
                                              "validation_id": "kfold5"})),
            reply(Call("finish", {})))
        agent = Agent(self.task, provider, Config(total_seconds=120, reserve_seconds=5), sleep=lambda _: None)
        result = agent.run()
        self.assertTrue(result["success"], result)
        self.assertEqual(agent.store.best["direction"], "minimize")
        self.assertEqual((self.root / "submission.csv").read_text(), "id,y\n10,8\n11,10\n")


if __name__ == "__main__":
    unittest.main()
