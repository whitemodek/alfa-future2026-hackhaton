"""Row-local flow features have no fitted/test-label state and survive refit."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ml_backend as fixtures
import ml_backend as b
from ml_flow import add_flow_features


class FlowFeatureTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(spkts=[0., 10., 1e308], dpkts=[0., 2., 0.],
            sbytes=[0., 500., 1e308], dbytes=[0., 50., 0.], proto=['tcp', 'udp', None],
            service=['http', '-', None], state=['FIN', 'INT', None]))

    def test_zero_and_extreme_counts_do_not_produce_infinity(self):
        transformed, categorical = add_flow_features(self.frame())
        derived = [c for c in transformed if c.startswith('__flow_') and c not in categorical]
        self.assertTrue(np.isfinite(transformed[derived].to_numpy()).all())
        self.assertEqual(transformed['__flow_packet_asymmetry'].iloc[0], 0.)
        self.assertEqual(len(categorical), 3)

    def test_features_are_row_local_order_invariant_and_ignore_targets(self):
        frame = self.frame()
        original = frame.copy(deep=True)
        joined, _ = add_flow_features(frame)
        single = pd.concat([add_flow_features(frame.iloc[[i]])[0] for i in range(len(frame))])
        pd.testing.assert_frame_equal(joined, single)
        pd.testing.assert_frame_equal(frame, original)
        permuted, _ = add_flow_features(frame.iloc[::-1])
        pd.testing.assert_frame_equal(joined, permuted.iloc[::-1])
        first, _ = add_flow_features(frame.assign(target=0))
        second, _ = add_flow_features(frame.assign(target=999999))
        derived = [c for c in first if c.startswith('__flow_')]
        pd.testing.assert_frame_equal(first[derived], second[derived])

    def test_missing_schema_and_reserved_column_collisions_fail(self):
        with self.assertRaises(ValueError):
            add_flow_features(self.frame().drop(columns='proto'))
        with self.assertRaises(ValueError):
            add_flow_features(self.frame().assign(__flow_fake=1))


class FlowRuntimeTests(unittest.TestCase):
    setUp = fixtures.MLBackendTests.setUp
    envelope = fixtures.MLBackendTests.envelope

    def add_inputs(self):
        fixtures.MLBackendTests.classify(self, ('a', 'b'))
        for name in ('train.csv', 'test.csv'):
            frame = pd.read_csv(self.root / name)
            index = np.arange(len(frame))
            frame['spkts'], frame['dpkts'] = index % 11, index % 7
            frame['sbytes'], frame['dbytes'] = (index % 11) * 100, (index % 7) * 70
            frame['proto'] = np.where(index % 2, 'tcp', 'udp')
            frame['service'], frame['state'] = 'http', 'FIN'
            frame.to_csv(self.root / name, index=False)

    def test_all_native_families_reproduce_and_refit_flow_representations(self):
        self.add_inputs()
        for family in ('lightgbm', 'xgboost', 'catboost'):
            with self.subTest(family=family):
                result = b.run(self.envelope(action='experiment', model=family, flow_features=True, iterations=20, refit=False))
                final = b.run(self.envelope(action='refit', manifest_path=result['manifest_path']))
                self.assertTrue(final['final'])
                self.assertEqual(final['score'], result['score'])
                self.assertEqual(final['validation_id'], result['validation_id'])

    def test_raw_and_enriched_models_can_share_a_validation_and_ensemble(self):
        self.add_inputs()
        raw = b.run(self.envelope(action='experiment', model='lightgbm', iterations=20, refit=False))
        enriched = b.run(self.envelope(action='experiment', model='xgboost', flow_features=True, iterations=20, refit=False))
        self.assertEqual(raw['validation_id'], enriched['validation_id'])
        result = b.run(self.envelope(action='ensemble', manifest_paths=[raw['manifest_path'], enriched['manifest_path']]))
        self.assertEqual(result['status'], 'ok')
        final = b.run(self.envelope(action='refit', manifest_path=result['manifest_path']))
        self.assertTrue(final['final'])
        self.assertEqual(final['score'], result['score'])
