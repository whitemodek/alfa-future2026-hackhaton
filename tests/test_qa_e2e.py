"""QA: сквозной прогон Agent на синтетических задачах платформенного формата.

LLM заменена сценарием, но код экспериментов реальный (pandas + sklearn) и
исполняется через run_python в отдельном процессе. Проверяется вся цепочка:
configure_submission(sample) → кандидаты → выбор лучшего → submission.csv,
плюс формат и метрика по скрытым ответам.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from agent import Agent
from agent_config import Config
from test_qa_risks import Call, Scripted, reply

ROOT = Path(__file__).resolve().parents[1]

MODEL_CODE = '''
import json, numpy as np, pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
train, test = pd.read_csv("train.csv"), pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")
target = sample.columns[-1]
X = train.drop(columns=[target, "id"]); T = test.drop(columns=["id"])[X.columns]
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        cats = pd.Categorical(pd.concat([X[c], T[c]]).astype(str)).categories
        X[c] = pd.Categorical(X[c].astype(str), categories=cats).codes
        T[c] = pd.Categorical(T[c].astype(str), categories=cats).codes
y = train[target]
TASK = "{task}"; STRONG = {strong}
if TASK == "reg":
    model = HistGradientBoostingRegressor(max_iter=200) if STRONG else DummyRegressor()
    cv = KFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X, y, cv=cv)
    score = float(np.sqrt(mean_squared_error(y, oof)))
else:
    model = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced") if STRONG else DummyClassifier()
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X, y, cv=cv)
    score = float(f1_score(y, oof, average="macro"))
model.fit(X, y)
pred = model.predict(T)
out = sample[["id"]].copy()
out[target] = pd.Series(pred, index=test["id"]).loc[sample["id"]].values
out.to_csv("cand_{name}.csv", index=False)
print(json.dumps({{"score": score}}))
'''


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        subprocess.run([sys.executable, str(ROOT / "qa" / "gen_tasks.py"), str(cls.root), "--small"],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def solve(self, name, kind, metric, direction, schema_extra):
        folder = self.root / name
        cv_scores = {}

        def capture(number, kwargs):
            for message in kwargs["messages"]:
                if message["role"] == "tool":
                    stdout = json.loads(message["content"]).get("stdout", "")
                    lines = [x for x in stdout.splitlines() if x.startswith('{"score"')]
                    if lines:
                        cv_scores[message["tool_call_id"]] = json.loads(lines[-1])["score"]

        def score_of(call_id):
            return cv_scores[call_id]

        steps = [
            reply(Call("configure_submission", {"sample_path": "sample_submission.csv", "id_columns": ["id"],
                                                **schema_extra})),
            reply(Call("run_python", {"code": MODEL_CODE.format(task=kind, strong=False, name="weak")}, 2)),
        ]
        provider = Scripted(*steps, on_call=capture)
        agent = Agent(folder / "task.md", provider, Config(total_seconds=600, reserve_seconds=60,
                                                           call_seconds=240), sleep=lambda _: None)
        # Динамический сценарий: регистрируем с реально посчитанным CV.
        dynamic = iter([
            lambda: reply(Call("register_candidate", {"candidate_path": "cand_weak.csv", "score": score_of("call_2"),
                                                      "metric": metric, "direction": direction,
                                                      "validation_id": "kfold5-seed42"}, 3)),
            lambda: reply(Call("run_python", {"code": MODEL_CODE.format(task=kind, strong=True, name="strong")}, 4)),
            lambda: reply(Call("register_candidate", {"candidate_path": "cand_strong.csv", "score": score_of("call_4"),
                                                      "metric": metric, "direction": direction,
                                                      "validation_id": "kfold5-seed42"}, 5)),
            lambda: reply(Call("finish", {}), 6),
        ])
        original = provider.create

        def create(**kwargs):
            if provider.calls < len(steps):
                return original(**kwargs)
            capture(provider.calls, kwargs)
            provider.calls += 1
            return next(dynamic)()

        provider.chat.completions.create = create
        result = agent.run()
        self.assertTrue(result["success"], result)
        pick = min if direction == "minimize" else max
        self.assertEqual(agent.store.best["score"], pick(cv_scores["call_2"], cv_scores["call_4"]),
                         "backend must keep the candidate with the better CV score")
        check = subprocess.run([sys.executable, str(ROOT / "qa" / "check_submission.py"), str(folder)],
                               capture_output=True, text=True)
        report = json.loads(check.stdout)
        self.assertTrue(report["format_ok"], report)
        return report, cv_scores

    def test_regression_with_outliers_and_dates(self):
        report, cv = self.solve("reg_outliers", "reg", "rmse", "minimize", {"numeric_columns": ["site_eui"]})
        self.assertLess(report["metric"]["rmse"], 1.3 * min(cv.values()), "test RMSE far from CV RMSE")

    def test_multiclass_string_labels_and_string_ids(self):
        import pandas as pd
        labels = sorted(pd.read_csv(self.root / "multiclass_imbalanced" / "train.csv")["attack_cat"].unique())
        report, _ = self.solve("multiclass_imbalanced", "clf", "macro_f1", "maximize",
                               {"allowed_values": {"attack_cat": labels}})
        self.assertGreater(report["metric"]["macro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
