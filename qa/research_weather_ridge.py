"""Fixed within-entity linear weather-response model, softly blended with control.

No hyperparameter search or early stopping: alpha100 and a .25 blend are
declared before evaluation. Only strictly past entity targets define residuals.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from qa.research_weather_delta import causal_feature_deltas
from qa.research_building_delta import residual_history
from qa.research_attack_specialist import sha256


class WeatherRidge:
    def frame(self, raw, lag, delta, weather):
        frame = pd.DataFrame({'lag': lag,
            'log_area': np.log1p(pd.to_numeric(raw['floor_area'], errors='coerce').clip(lower=0)),
            'rating': pd.to_numeric(raw['energy_star_rating'], errors='coerce').replace(-1, np.nan),
            'built': pd.to_numeric(raw['year_built'], errors='coerce').replace(0, np.nan)}, index=raw.index)
        for i, name in enumerate(weather):
            frame['delta_' + name] = delta[:, i]
        for column in ('State_Factor', 'building_class', 'facility_type'):
            frame[column] = raw[column].fillna('__MISSING__').astype(str)
        return frame

    def encode(self, frame, fit=False):
        from scipy import sparse
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.pipeline import make_pipeline
        categories = ['State_Factor', 'building_class', 'facility_type']
        if fit:
            self.numeric_names = [name for name in frame if name not in categories]
            self.numeric = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                                         StandardScaler())
            self.category = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            number = self.numeric.fit_transform(frame[self.numeric_names])
            category = self.category.fit_transform(frame[categories])
        else:
            number = self.numeric.transform(frame[self.numeric_names])
            category = self.category.transform(frame[categories])
        parts = [sparse.csr_matrix(number), category]
        for name in ('delta_heating_degree_days_c', 'delta_cooling_degree_days_c', 'delta_avg_temp_c'):
            if name in self.numeric_names:
                index = self.numeric_names.index(name)
                parts.append(category.multiply(number[:, index, None]))
        return sparse.hstack(parts, format='csr')

    def fit(self, frame, residual):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=100., solver='lsqr', tol=1e-5, max_iter=2000)
        self.model.fit(self.encode(frame, fit=True), residual)
        if self.model.n_iter_ is not None and np.max(self.model.n_iter_) >= 2000:
            raise RuntimeError('Weather ridge did not converge')
        return self

    def predict(self, frame):
        return self.model.predict(self.encode(frame))


def main():
    import re
    import ml_backend as b
    from ml_history import entity_columns
    from threadpoolctl import threadpool_limits
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'weather-ridge-study'
    out.mkdir(exist_ok=False)
    data = scratch / 'organizer/regression'
    p = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    p['validation'] = dict(strategy='time', diagnostics={'time_column': 'Year_Factor'})
    columns, rationale = entity_columns(p, {})
    weather = [c for c in p['features'] if c not in columns and c != 'Year_Factor'
               and re.search('temp|degree|days|precip|snow|wind|fog', c, re.I)]
    protocol = dict(periods=[5, 4], alpha=100, blend=.25, grid_search=False, early_stopping=False,
        numeric='lag, log area, rating, built, strictly causal weather differences; train-only imputation/scaling',
        interactions='Fixed weather-change x state/class/facility indicators for HDD/CDD/mean temperature',
        entity_columns=columns, weather_columns=weather, rationale=rationale,
        gate='Relative RMSE gain >= .001 on BOTH complete periods before runtime follow-up',
        api_calls=0, platform_submissions=0, max_seconds=300,
        source_sha256={n: sha256(root / n) for n in ('qa/research_weather_ridge.py', 'qa/research_weather_delta.py')},
        limitation='Historical, previously inspected temporal folds; not a blind estimate or platform score')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    started = time.monotonic()
    report = dict(protocol=protocol, records=[], completed=False)
    forecasts = []
    for period in protocol['periods']:
        print('FIT', period, flush=True)
        train = np.flatnonzero(p['train']['Year_Factor'].to_numpy() < period)
        valid = np.flatnonzero(p['train']['Year_Factor'].to_numpy() == period)
        history = residual_history(p['train'].iloc[train], p['y'].iloc[train], p['train'].iloc[valid],
                                   columns, 'Year_Factor', time.monotonic() + 60)
        delta, query_delta = causal_feature_deltas(p['train'].iloc[train], p['train'].iloc[valid],
            columns, 'Year_Factor', weather, time.monotonic() + 60)
        a, v = history['train_available'], history['query_available']
        model = WeatherRidge()
        train_frame = model.frame(p['train'].iloc[train], history['train_lag'], delta, weather)
        query_frame = model.frame(p['train'].iloc[valid], history['query_lag'], query_delta, weather)
        with threadpool_limits(limits=2):
            model.fit(train_frame.iloc[np.flatnonzero(a)], history['residual_target'][a])
            forecast = history['query_lag'][v] + model.predict(query_frame.iloc[np.flatnonzero(v)])
        cache = scratch / f'building-delta-report-artifacts/period{period}.npz'
        with np.load(cache, allow_pickle=False) as old:
            if not np.array_equal(old['outer_validation_indices'], valid):
                raise ValueError('Cached control indices changed')
            base = old['baseline'].copy()
        candidate = base.copy()
        candidate[v] = .75 * base[v] + .25 * forecast
        np.testing.assert_array_equal(candidate[~v], base[~v])
        np.savez_compressed(out / f'period{period}-frozen.npz', prediction=candidate, valid_indices=valid)
        forecasts.append((period, valid, base, candidate, sha256(cache)))
    # Both predictions are frozen before scoring either outer period.
    for period, valid, base, candidate, cache_hash in forecasts:
        truth = p['y'].iloc[valid].to_numpy()
        old = float(np.sqrt(np.mean((base - truth) ** 2)))
        new = float(np.sqrt(np.mean((candidate - truth) ** 2)))
        row = dict(period=period, control_rmse=old, candidate_rmse=new, relative_gain=(old - new) / old,
                   control_cache_sha256=cache_hash)
        report['records'].append(row)
        print('RESULT', row, flush=True)
    report.update(completed=True, passed=all(r['relative_gain'] >= .001 for r in report['records']),
                  seconds=round(time.monotonic() - started, 3))
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '2'
    main()
