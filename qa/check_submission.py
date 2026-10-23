"""Проверка сабмита «как на платформе»: формат против sample_submission + метрика против скрытых ответов.

    python qa/check_submission.py qa_tasks/reg_outliers
Код возврата 0 — формат валиден. Метрика печатается, если есть qa_tasks/_answers/<name>.csv.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def check_format(task_dir):
    task_dir = Path(task_dir).resolve()
    problems = []
    sub_path = task_dir / "submission.csv"
    if not sub_path.exists():
        return ["submission.csv отсутствует"], None
    sample = pd.read_csv(task_dir / "sample_submission.csv")
    sub = pd.read_csv(sub_path)
    if list(sub.columns) != list(sample.columns):
        problems.append(f"колонки {list(sub.columns)} != {list(sample.columns)}")
    if len(sub) != len(sample):
        problems.append(f"строк {len(sub)} != {len(sample)}")
    if "id" in sub and "id" in sample:
        if sub["id"].astype(str).tolist() != sample["id"].astype(str).tolist():
            problems.append("id или их порядок не совпадают с sample_submission")
        if sub["id"].duplicated().any():
            problems.append("дубликаты id")
    if sub.isna().any().any():
        problems.append(f"NaN: {sub.isna().sum().to_dict()}")
    target = sample.columns[-1]
    if target in sub and pd.api.types.is_numeric_dtype(sample[target]) and not pd.api.types.is_numeric_dtype(sub[target]):
        problems.append(f"{target}: ожидались числа, получено {sub[target].dtype}")
    return problems, sub


def score(task_dir, sub):
    task_dir = Path(task_dir).resolve()
    answers = task_dir.parent / "_answers" / f"{task_dir.name}.csv"
    if not answers.exists() or sub is None:
        return None
    truth = pd.read_csv(answers)
    target = truth.columns[-1]
    merged = truth.merge(sub, on="id", suffixes=("_true", "_pred"))
    y, p = merged[f"{target}_true"], merged[f"{target}_pred"]
    from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
    kind = json.loads((answers.with_suffix(".json")).read_text())["kind"] if answers.with_suffix(".json").exists() \
        else task_dir.name
    if kind.startswith("reg"):
        return {"rmse": float(np.sqrt(mean_squared_error(y, p)))}
    if "auc" in kind:
        return {"roc_auc": float(roc_auc_score(y, p)),
                "warning": "предсказаны жёсткие метки 0/1 — AUC сильно занижен" if set(p.unique()) <= {0, 1} else None}
    return {"macro_f1": float(f1_score(y.astype(str), p.astype(str), average="macro"))}


if __name__ == "__main__":
    folder = sys.argv[1]
    problems, sub = check_format(folder)
    print(json.dumps({"task": folder, "format_ok": not problems, "problems": problems,
                      "metric": score(folder, sub)}, ensure_ascii=False))
    sys.exit(1 if problems else 0)
