"""Генератор синтетических задач в формате платформы (task.md + train/test/data_dictionary/sample_submission).

Скрытые ответы кладутся в qa_tasks/<name>/../_answers/<name>.csv — агент их не видит.
Задачи специально «неудобные», чтобы проверить универсальность агента на private-этапе.

    python qa/gen_tasks.py qa_tasks            # все задачи
    python qa/gen_tasks.py qa_tasks --small    # маленькие (быстрый smoke)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _write(root, name, train, test, target, answers, task_md, dictionary, sample_fill):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    train.to_csv(folder / "train.csv", index=False)
    test.to_csv(folder / "test.csv", index=False)
    pd.DataFrame(dictionary, columns=["feature", "role", "dtype", "unit", "description_ru"]).to_csv(
        folder / "data_dictionary.csv", index=False)
    sample = pd.DataFrame({"id": test["id"], target: sample_fill})
    sample.to_csv(folder / "sample_submission.csv", index=False)
    (folder / "task.md").write_text(task_md, encoding="utf-8")
    hidden = root / "_answers"
    hidden.mkdir(exist_ok=True)
    pd.DataFrame({"id": test["id"], target: answers}).to_csv(hidden / f"{name}.csv", index=False)


def _dictionary(df, target, descriptions=None):
    rows = []
    for column in df.columns:
        role = "id" if column == "id" else "target" if column == target else "feature"
        rows.append([column, role, str(df[column].dtype), "", (descriptions or {}).get(column, f"Признак {column}")])
    return rows


def regression_outliers(rng, n):
    """Регрессия, выбросы в таргете, пропуски, категориальные с высокой кардинальностью, дата."""
    x1 = rng.normal(size=n)
    x2 = rng.uniform(0, 10, size=n)
    city = rng.choice([f"city_{i}" for i in range(300)], size=n)
    kind = rng.choice(["Офис", "Склад", "Жилой", None], size=n, p=[.4, .3, .25, .05])
    built = pd.to_datetime("1950-01-01") + pd.to_timedelta(rng.integers(0, 70 * 365, size=n), unit="D")
    city_effect = pd.Series(rng.normal(0, 3, 300), index=[f"city_{i}" for i in range(300)])
    y = 50 + 8 * x1 + 3 * x2 ** 1.3 + city_effect[city].to_numpy() + (kind == "Склад") * 15 \
        - (built.year.to_numpy() - 1950) * 0.2 + rng.normal(0, 4, size=n)
    y[rng.random(n) < 0.01] *= 8  # выбросы
    x1[rng.random(n) < 0.15] = np.nan
    df = pd.DataFrame({"id": np.arange(n) + 100000, "floor_area": x2, "energy_star": x1, "city": city,
                       "building_type": kind, "built_date": built.strftime("%Y-%m-%d"), "site_eui": y})
    return df, "site_eui"


def multiclass_imbalanced(rng, n):
    """Многоклассовая, сильный дисбаланс, строковые метки с пробелами/кириллицей, строковые id."""
    classes = np.array(["Normal", "DoS", "Exploits", "Backdoor", "Worms", "Атака нулевого дня"])
    probs = np.array([.55, .2, .15, .06, .03, .01])
    y_idx = rng.choice(len(classes), size=n, p=probs)
    centers = rng.normal(0, 2.5, size=(len(classes), 8))
    X = centers[y_idx] + rng.normal(0, 1.6, size=(n, 8))
    proto = np.where(y_idx % 2 == 0, "tcp", rng.choice(["udp", "icmp", "tcp"], size=n))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df.insert(0, "id", [f"rec-{i:07d}" for i in rng.permutation(n)])
    df["proto"] = proto
    df["attack_cat"] = classes[y_idx]
    return df, "attack_cat"


def binary_numeric_labels(rng, n):
    """Бинарная классификация: таргет 0/1 int, метрика ROC-AUC (ожидаются вероятности!)."""
    X = rng.normal(size=(n, 12))
    logit = X[:, 0] * 1.5 - X[:, 1] + 0.5 * X[:, 2] * X[:, 3]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(12)])
    df.insert(0, "id", np.arange(n))
    df["default_flag"] = y
    return df, "default_flag"


def make(root, small):
    rng = np.random.default_rng(42)
    size = 1500 if small else 30000
    specs = [
        ("reg_outliers", regression_outliers, "RMSE", "регрессия", None),
        ("multiclass_imbalanced", multiclass_imbalanced, "Macro F1", "многоклассовая классификация", None),
        ("binary_auc", binary_numeric_labels, "ROC-AUC (в сабмите — вероятность класса 1)",
         "бинарная классификация", 0.5),
    ]
    for name, builder, metric, kind, fill in specs:
        df, target = builder(rng, size)
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        cut = int(len(df) * 0.7)
        train, test_full = df.iloc[:cut], df.iloc[cut:]
        test = test_full.drop(columns=[target])
        fill_value = fill if fill is not None else (train[target].iloc[0])
        task = (f"# Задача: {kind}\n\nПо обучающей выборке train.csv предскажите `{target}` для test.csv.\n"
                f"Описание признаков — в data_dictionary.csv. Формат ответа — как в sample_submission.csv.\n\n"
                f"Метрика: {metric}.\n")
        _write(root, name, train, test, target, test_full[target].to_numpy(), task,
               _dictionary(df, target), fill_value)
        print("created", root / name, train.shape, test.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    make(Path(args.root), args.small)
