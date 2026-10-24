"""Validate a QA submission against its sample and optional hidden answers."""
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


def _read(path):
    # IDs are identifiers: preserve leading zeroes and textual values.
    frame = pd.read_csv(path, dtype={'id': str})
    if 'id' in frame:
        frame['id'] = pd.read_csv(path, usecols=['id'], dtype=str, keep_default_na=False)['id']
    return frame


def _allowed_labels(task_dir, target):
    """Infer categorical label constraints using public training data only."""
    paths = [path for path in task_dir.rglob('train.csv') if not any(
        part.startswith(('.agent', '.ml', '.venv', '.git'))
        for part in path.relative_to(task_dir).parts)]
    if len(paths) != 1:
        return None
    header = pd.read_csv(paths[0], nrows=0)
    if target not in header:
        return None
    labels = pd.read_csv(paths[0], usecols=[target])[target]
    text_path = task_dir / 'task.md'
    text = text_path.read_text(encoding='utf-8-sig') if text_path.exists() else ''
    # A categorical training target does not constrain probability submissions.
    if re.search(r'roc[ _-]?auc|log[ _-]?loss|probabil|вероятност', text, re.I):
        return None
    classification = bool(re.search(r'classification|классификац|macro[ _-]?f1|accuracy', text, re.I))
    if not classification and pd.api.types.is_numeric_dtype(labels):
        return None
    # Class names are CSV lexemes, including numeric-looking names and NA/inf.
    return set(pd.read_csv(paths[0], usecols=[target], dtype=str,
                           keep_default_na=False)[target].tolist())


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
    target = sample.columns[-1] if len(sample.columns) else None
    allowed = None
    if target in sub:
        try:
            allowed = _allowed_labels(task_dir, target)
            if allowed is not None:
                sub[target] = pd.read_csv(sub_path, usecols=[target], dtype=str,
                                          keep_default_na=False)[target]
        except (OSError, ValueError, pd.errors.ParserError):
            problems.append('Не удалось прочитать публичные метки классов')
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
    for name in sub:
        if name == 'id':
            continue
        if name == target and allowed is not None:
            if not sub[name].isin(allowed).all() or sub[name].str.strip().eq('').any():
                problems.append(f'{name}: недопустимая метка класса')
            continue
        invalid = False
        for value in sub[name]:
            if not str(value).strip():
                invalid = True
                break
            try:
                if not math.isfinite(float(value)):
                    invalid = True
                    break
            except (TypeError, ValueError):
                pass
        if invalid:
            problems.append(f'{name}: пропущенное или неконечное предсказание')
    for name in sub.select_dtypes(include='number'):
        if not np.isfinite(sub[name].to_numpy()).all():
            problems.append(f'{name}: значения должны быть конечными')
    if len(sample.columns):
        target = sample.columns[-1]
        if (allowed is None and target in sub and pd.api.types.is_numeric_dtype(sample[target])
                and not pd.api.types.is_numeric_dtype(sub[target])):
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
    # Match class names without pandas collapsing NA or leading zeroes in answers.
    truth[target] = pd.read_csv(answers, usecols=[target], dtype=str,
                                keep_default_na=False)[target]
    y = truth.set_index('id').loc[sub['id'], target].to_numpy()
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
