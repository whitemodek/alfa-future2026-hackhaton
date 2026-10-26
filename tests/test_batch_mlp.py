"""Train-only inputs and fixed-epoch refit of the experimental neural expert."""
import importlib.util
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Optional platform Torch package is unavailable')
class BatchMLPTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame({'x': np.arange(24, dtype=float), 'c': ['a', 'b'] * 12})

    def test_piecewise_inputs_are_bounded_and_queries_do_not_change_knots_or_vocabulary(self):
        from qa.research_batch_mlp import PiecewiseInputs
        data = self.data()
        encoder = PiecewiseInputs().fit(data, ['c'])
        knots = [a.copy() for a in encoder.knots]
        category = [a.copy() for a in encoder.encoder.categories_]
        encoded = encoder.transform(pd.DataFrame({'x': [np.nan, -1000, 1000], 'c': ['new', 'a', 'b']}))
        self.assertTrue(np.isfinite(encoded).all())
        self.assertTrue(((encoded >= 0) & (encoded <= 1)).all())
        self.assertEqual(encoded.shape[1], sum(encoder.chunks))
        for a, b in zip(knots, encoder.knots):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(category, encoder.encoder.categories_):
            np.testing.assert_array_equal(a, b)

    def test_member_axis_and_gradients_are_finite(self):
        import torch
        from qa.research_batch_mlp import BatchMLP
        torch.set_num_threads(1)
        model = BatchMLP([3, 2], count=3, members=4, width=16)
        logits = model(torch.ones(7, 5))
        self.assertEqual(tuple(logits.shape), (7, 4, 3))
        labels = torch.arange(7) % 3
        loss = torch.nn.CrossEntropyLoss()(logits.reshape(-1, 3), labels.repeat_interleave(4))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_fixed_epoch_refit_requires_no_validation_answers(self):
        import torch
        from qa.research_batch_mlp import LearnedExpert
        torch.set_num_threads(1)
        x = self.data()
        y = np.arange(len(x)) % 2
        with patch.object(LearnedExpert, '_probability', side_effect=AssertionError('No validation in exact refit')):
            model = LearnedExpert().fit(x, y, ['c'], time.monotonic() + 15, fixed_epochs=1)
        self.assertEqual(model.epochs, 1)
        probabilities = model.predict_proba(x.iloc[:3])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        with self.assertRaises(TimeoutError):
            LearnedExpert().fit(x, y, ['c'], time.monotonic() - 1, fixed_epochs=1)
