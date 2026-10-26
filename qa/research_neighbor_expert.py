"""A fixed rank-distance neighbor expert, selected only on inner group data."""
import json
import os
from pathlib import Path
import time

import numpy as np

from qa.research_attack_specialist import FOLDS, sha256, guarded_splits, metrics


class NeighborExpert:
    def fit(self, x, y, categorical, deadline):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
        from sklearn.neighbors import NearestNeighbors
        from threadpoolctl import threadpool_limits
        if len(x) < 15 or len(x) > 200000:
            raise ValueError('Neighbor expert training size outside fixed bounds')
        numeric = [c for c in x if c not in categorical]
        self.transform = ColumnTransformer([
            ('numeric', make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                QuantileTransformer(n_quantiles=min(1000, len(x)), output_distribution='uniform',
                                    subsample=10000, random_state=42)), numeric),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32), categorical)],
            transformer_weights={'numeric': 1., 'categorical': 2.}, sparse_threshold=0)
        with threadpool_limits(limits=2):
            encoded = np.asarray(self.transform.fit_transform(x), dtype=np.float32)
            if encoded.size > 50_000_000 or not np.isfinite(encoded).all():
                raise ValueError('Neighbor matrix exceeds the finite dense-cell cap')
            self.model = NearestNeighbors(n_neighbors=15, algorithm='brute', metric='euclidean', n_jobs=2).fit(encoded)
        self.labels = np.asarray(y, dtype=int)
        counts = np.bincount(self.labels)
        if (counts == 0).any():
            raise ValueError('Neighbor training must preserve every class')
        self.class_weights = np.sqrt(counts.max() / counts)
        self.count = len(counts)
        if time.monotonic() >= deadline:
            raise TimeoutError('Neighbor training deadline exhausted')
        return self

    def predict_proba(self, x, deadline):
        from threadpoolctl import threadpool_limits
        probability = np.zeros((len(x), self.count))
        with threadpool_limits(limits=2):
            for start in range(0, len(x), 256):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Neighbor prediction deadline exhausted')
                frame = np.asarray(self.transform.transform(x.iloc[start:start + 256]), dtype=np.float32)
                distance, indices = self.model.kneighbors(frame)
                zero = distance <= 1e-8
                weights = np.where(zero.any(axis=1, keepdims=True), zero.astype(float), 1 / np.maximum(distance, 1e-8))
                labels = self.labels[indices]
                weights *= self.class_weights[labels]
                block = np.zeros((len(frame), self.count))
                np.add.at(block, (np.repeat(np.arange(len(frame)), labels.shape[1]), labels.ravel()), weights.ravel())
                probability[start:start + len(frame)] = block / block.sum(axis=1, keepdims=True)
        return probability


def main():
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'neighbor-expert-study'
    out.mkdir(exist_ok=False)
    protocol = dict(neighbors=15, numeric='Train-only median + uniform quantile transform',
        categorical='Train-only one-hot, fixed scale 2', class_weights='sqrt inverse training class count',
        blend_grid=[0., .25, .5], selection='Inner-only Macro F1, least intervention on ties',
        gate='Macro F1 gain >= .002 on BOTH historical outer folds before runtime follow-up',
        api_calls=0, platform_submissions=0, max_seconds=600, source_sha256=sha256(__file__),
        limitation='Known dataset and historical folds; exploratory cached-control comparison, not a platform score')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    started = time.monotonic()
    deadline = started + 600
    report = dict(protocol=protocol, trials=[], completed=False)
    import pandas as pd
    import ml_backend as b
    from sklearn.metrics import f1_score
    data = scratch / 'organizer/multiclass'
    p = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    x, _, cats = b._features(p)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    report['input_sha256'] = b._input_hashes(p)

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    for fold in FOLDS:
        with np.load(scratch / f'night-classification-results/{fold}-indices.npz', allow_pickle=False) as a:
            train, valid, it, iv = (a[k].copy() for k in ('train', 'valid', 'inner_train', 'inner_valid'))
        guards = guarded_splits(groups, train, valid, it, iv)
        inner, outer = [], []
        for rel in (f'night-classification-results/{fold}-lightgbm63_sqrt.npz',
                    f'night-classification-results/{fold}-xgboost6_sqrt.npz',
                    f'late-extra100-results/{fold}-raw100.npz'):
            with np.load(scratch / rel, allow_pickle=False) as a:
                key = 'validation_indices' if 'validation_indices' in a else 'val_indices'
                if not np.array_equal(a[key], valid) or not np.array_equal(a['inner_indices'], iv):
                    raise ValueError('Historical row alignment changed')
                inner.append(a['inner_probability'].copy())
                outer.append(a['probability'].copy())
        inner, outer = np.mean(inner, axis=0), np.mean(outer, axis=0)
        print('INNER', fold, flush=True)
        expert = NeighborExpert().fit(x.iloc[it], p['y'].iloc[it], cats, deadline)
        inner_expert = expert.predict_proba(x.iloc[iv], deadline)
        scores = [float(f1_score(p['y'].iloc[iv], ((1 - weight) * inner + weight * inner_expert).argmax(axis=1),
            labels=np.arange(len(p['classes'])), average='macro', zero_division=0)) for weight in protocol['blend_grid']]
        strength = protocol['blend_grid'][int(np.argmax(scores))]
        row = dict(fold=fold, guards=guards, inner_scores=scores, selected_strength=strength)
        report['trials'].append(row)
        save()
        print('OUTER', fold, strength, flush=True)
        if strength:
            full = NeighborExpert().fit(x.iloc[train], p['y'].iloc[train], cats, deadline)
            probability = full.predict_proba(x.iloc[valid], deadline)
            candidate = (1 - strength) * outer + strength * probability
        else:
            candidate = outer.copy()
        np.savez_compressed(out / f'{fold}-frozen.npz', probability=candidate, valid_indices=valid)
        row['base'] = metrics(p['y'].iloc[valid].to_numpy(), outer, p['classes'])
        row['candidate'] = metrics(p['y'].iloc[valid].to_numpy(), candidate, p['classes'])
        row['gain'] = row['candidate']['macro_f1'] - row['base']['macro_f1']
        save()
        print('RESULT', fold, row['gain'], flush=True)
    report.update(completed=True, passed=all(r['gain'] >= .002 for r in report['trials']))
    save()
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '2'
    main()
