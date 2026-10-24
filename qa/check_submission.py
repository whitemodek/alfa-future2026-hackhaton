"""Validate a QA submission against its sample and optional hidden answers."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _read(path):
    # IDs are identifiers: preserve leading zeroes and textual values.
    return pd.read_csv(path, dtype={'id': str})


def check_format(task_dir):
    task_dir = Path(task_dir).resolve()
    sub_path = task_dir / 'submission.csv'
    if not sub_path.is_file():
        return ['submission.csv отсутствует'], None
    try:
        sample = _read(task_dir / 'sample_submission.csv')
        sub = _read(sub_path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        return [f'CSV не читается: {type(error).__name__}'], None
    problems = []
    if list(sub.columns) != list(sample.columns):
        problems.append(f'колонки {list(sub.columns)} != {list(sample.columns)}')
    if len(sub) != len(sample):
        problems.append(f'строк {len(sub)} != {len(sample)}')
    if 'id' in sub and 'id' in sample:
        if sub['id'].tolist() != sample['id'].tolist():
            problems.append('id или их порядок не совпадают с sample_submission')
        if sub['id'].duplicated().any():
            problems.append('дубликаты id')
    if sub.isna().any().any():
        problems.append('CSV содержит пропущенные значения')
    for name in sub.select_dtypes(include='number'):
        if not np.isfinite(sub[name].to_numpy()).all():
            problems.append(f'{name}: значения должны быть конечными')
    if len(sample.columns):
        target = sample.columns[-1]
        if target in sub and pd.api.types.is_numeric_dtype(sample[target]) and not pd.api.types.is_numeric_dtype(sub[target]):
            problems.append(f'{target}: ожидались числа, получено {sub[target].dtype}')
    return problems, sub


def score(task_dir, sub):
    task_dir = Path(task_dir).resolve()
    # Never compute a plausible score for a partial or malformed submission.
    problems, verified = check_format(task_dir)
    if problems or sub is None or verified is None or not sub.equals(verified):
        return None
    answers = task_dir.parent / '_answers' / f'{task_dir.name}.csv'
    if not answers.exists():
        return None
    truth = _read(answers)
    if len(truth) != len(sub) or 'id' not in truth or 'id' not in sub or truth['id'].duplicated().any():
        raise ValueError('Hidden answer IDs are invalid')
    if set(truth['id']) != set(sub['id']):
        raise ValueError('Hidden answer IDs do not match submission')
    target = truth.columns[-1]
    y = truth.set_index('id').loc[sub['id'], target].to_numpy()
    p = sub[target].to_numpy()
    from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
    metadata = answers.with_suffix('.json')
    kind = json.loads(metadata.read_text(encoding='utf-8-sig'))['kind'] if metadata.exists() else task_dir.name
    if kind.startswith('reg'):
        return {'rmse': float(np.sqrt(mean_squared_error(y, p)))}
    if 'auc' in kind:
        if not ((p >= 0) & (p <= 1)).all():
            raise ValueError('ROC-AUC QA expects probabilities in [0, 1]')
        return {'roc_auc': float(roc_auc_score(y, p)),
                'warning': 'Предсказаны жёсткие метки 0/1' if set(p) <= {0, 1} else None}
    return {'macro_f1': float(f1_score(y.astype(str), p.astype(str), average='macro'))}


if __name__ == '__main__':
    folder = sys.argv[1]
    problems, sub = check_format(folder)
    metric = None
    if not problems:
        try:
            metric = score(folder, sub)
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f'Проверка метрики: {type(error).__name__}: {error}')
    print(json.dumps({'task': folder, 'format_ok': not problems, 'problems': problems,
                      'metric': metric}, ensure_ascii=False))
    sys.exit(1 if problems else 0)
