"""Scratch-trained, BatchEnsemble-inspired MLP with train-only piecewise inputs.

This is our small CPU implementation, NOT the official TabM implementation
or a reproduction of its published results. No external weights/downloads.
"""
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from qa.research_attack_specialist import FOLDS, sha256, guarded_splits, metrics


class PiecewiseInputs:
    def fit(self, x, categorical):
        from sklearn.preprocessing import OneHotEncoder
        self.columns = list(x)
        self.categorical = list(categorical)
        self.numeric = [c for c in x if c not in categorical]
        values = x[self.numeric].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        self.median = np.nanmedian(values, axis=0)
        self.median[~np.isfinite(self.median)] = 0
        filled = np.where(np.isfinite(values), values, self.median)
        self.knots = []
        self.chunks = []
        for column in filled.T:
            knots = np.unique(np.quantile(column, np.linspace(0, 1, 9)))
            if len(knots) < 2:
                knots = np.array([knots[0], knots[0] + 1.])
            self.knots.append(knots)
            self.chunks.append(len(knots))  # intervals plus one missingness flag
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        self.encoder.fit(x[self.categorical].fillna('__MISSING__').astype(str))
        self.chunks.extend(len(c) for c in self.encoder.categories_)
        return self

    def transform(self, x):
        values = x[self.numeric].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        filled = np.where(~missing, values, self.median)
        parts = []
        for i, knots in enumerate(self.knots):
            interval = np.clip((filled[:, i, None] - knots[:-1]) / (knots[1:] - knots[:-1]), 0., 1.)
            parts.append(np.column_stack([interval, missing[:, i]]).astype(np.float32))
        parts.append(self.encoder.transform(x[self.categorical].fillna('__MISSING__').astype(str)))
        result = np.concatenate(parts, axis=1)
        if result.shape[1] != sum(self.chunks) or result.size > 60_000_000 or not np.isfinite(result).all():
            raise ValueError('Invalid or oversized piecewise feature matrix')
        return result


class EnsembleLinear(nn.Module):
    def __init__(self, inputs, outputs, members, chunks=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, inputs))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if chunks is None:
            r = torch.ones(members, inputs)
        else:
            signs = torch.randint(0, 2, (members, len(chunks))).float() * 2 - 1
            r = torch.repeat_interleave(signs, torch.tensor(chunks), dim=1)
        self.r = nn.Parameter(r)
        self.s = nn.Parameter(torch.ones(members, outputs))
        self.bias = nn.Parameter(torch.zeros(members, outputs))

    def forward(self, x):
        return nn.functional.linear(x * self.r, self.weight) * self.s + self.bias


class BatchMLP(nn.Module):
    def __init__(self, chunks, count, members=8, width=128):
        super().__init__()
        self.members = members
        self.layers = nn.Sequential(EnsembleLinear(sum(chunks), width, members, chunks), nn.ReLU(),
            EnsembleLinear(width, width, members), nn.ReLU(), EnsembleLinear(width, width, members),
            nn.ReLU(), EnsembleLinear(width, count, members))

    def forward(self, x):
        return self.layers(x[:, None, :].expand(-1, self.members, -1))


class LearnedExpert:
    def fit(self, x, y, categorical, deadline, valid_x=None, valid_y=None, fixed_epochs=None):
        from sklearn.metrics import f1_score
        torch.manual_seed(42)
        self.inputs = PiecewiseInputs().fit(x, categorical)
        encoded = torch.from_numpy(self.inputs.transform(x))
        labels = torch.tensor(np.asarray(y), dtype=torch.long)
        counts = np.bincount(np.asarray(y, dtype=int))
        if (counts <= 0).any():
            raise ValueError('Training must preserve all neural classes')
        self.count = len(counts)
        self.model = BatchMLP(self.inputs.chunks, self.count)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.002, weight_decay=.0003)
        loss = nn.CrossEntropyLoss(weight=torch.tensor(np.sqrt(counts.max() / counts), dtype=torch.float32))
        validation = torch.from_numpy(self.inputs.transform(valid_x)) if valid_x is not None else None
        best, best_state, best_epoch, stale = -np.inf, None, 0, 0
        self.curve = []
        for epoch in range(1, (fixed_epochs or 100) + 1):
            self.model.train()
            complete = True
            for batch in torch.randperm(len(encoded)).split(256):
                if time.monotonic() >= deadline:
                    complete = False
                    break
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(encoded[batch])
                objective = loss(logits.reshape(-1, self.count), labels[batch].repeat_interleave(self.model.members))
                if not torch.isfinite(objective):
                    raise ValueError('Nonfinite neural loss')
                objective.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                optimizer.step()
            if not complete:
                if fixed_epochs:
                    raise TimeoutError('Exact neural refit did not complete selected epochs')
                break
            if fixed_epochs:
                self.epochs = epoch
                continue
            if epoch == 1 or epoch % 2 == 0:
                probability = self._probability(validation)
                score = float(f1_score(valid_y, probability.argmax(axis=1), labels=np.arange(self.count),
                                       average='macro', zero_division=0))
                self.curve.append(dict(epoch=epoch, macro_f1=score))
                print('EPOCH', epoch, score, flush=True)
                if score > best + 1e-7:
                    best, best_epoch, stale = score, epoch, 0
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    stale += 1
                if stale >= 12:
                    break
        if not fixed_epochs:
            if best_state is None:
                raise TimeoutError('No complete neural epoch/checkpoint')
            self.model.load_state_dict(best_state)
            self.epochs = best_epoch
        return self

    def _probability(self, encoded):
        self.model.eval()
        output = []
        with torch.no_grad():
            for batch in encoded.split(512):
                output.append(torch.softmax(self.model(batch), dim=-1).mean(dim=1).numpy())
        return np.concatenate(output)

    def predict_proba(self, x):
        return self._probability(torch.from_numpy(self.inputs.transform(x)))


def main():
    import ml_backend as b
    from sklearn.metrics import f1_score
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'batch-mlp-study'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    protocol = dict(folds=FOLDS, members=8, width=128, hidden_layers=3, quantile_intervals=8,
        learning_rate=.002, weight_decay=.0003, batch_size=256, max_epochs=100,
        patience=12, eval_every=2, blend_grid=[0., .25, .5],
        gate='Macro F1 gain >=.002 on BOTH outer folds before runtime follow-up',
        api_calls=0, platform_submissions=0, pretrained_downloads=False,
        source_sha256=sha256(__file__), inspiration='https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html',
        limitation='Own CPU BatchEnsemble-inspired implementation, not official TabM; known dataset/folds. '
                   'Epochs/blend selected on inner groups only; exact outer refit, no outer tuning.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    started = time.monotonic()
    report = dict(protocol=protocol, trials=[], completed=False)
    data = scratch / 'organizer/multiclass'
    p = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    x, _, cats = b._features(p)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

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
                    raise ValueError('Cached neural control alignment changed')
                inner.append(a['inner_probability'].copy())
                outer.append(a['probability'].copy())
        inner, outer = np.mean(inner, axis=0), np.mean(outer, axis=0)
        print('INNER', fold, flush=True)
        model = LearnedExpert().fit(x.iloc[it], p['y'].iloc[it].to_numpy(), cats, time.monotonic() + 500,
                                    x.iloc[iv], p['y'].iloc[iv].to_numpy())
        probability = model.predict_proba(x.iloc[iv])
        scores = [float(f1_score(p['y'].iloc[iv], ((1 - w) * inner + w * probability).argmax(axis=1),
            labels=np.arange(len(p['classes'])), average='macro', zero_division=0)) for w in protocol['blend_grid']]
        strength = protocol['blend_grid'][int(np.argmax(scores))]
        row = dict(fold=fold, guards=guards, selected_epochs=model.epochs, curve=model.curve,
                   inner_scores=scores, selected_strength=strength)
        report['trials'].append(row)
        save()
        print('OUTER', fold, strength, model.epochs, flush=True)
        if strength:
            full = LearnedExpert().fit(x.iloc[train], p['y'].iloc[train].to_numpy(), cats,
                                       time.monotonic() + 700, fixed_epochs=model.epochs)
            candidate = (1 - strength) * outer + strength * full.predict_proba(x.iloc[valid])
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
    os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'
    main()
