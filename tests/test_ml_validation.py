import unittest

import numpy as np
import pandas as pd

from ml_validation import plan_validation, time_values


def problem(frame, test=None, labels=None, text=""):
    return dict(train=frame, test=frame.copy() if test is None else test,
                features=list(frame), y=pd.Series(np.arange(len(frame)) if labels is None else labels),
                classes=None if labels is None else sorted(set(labels)), text=text)


class ValidationPlannerTests(unittest.TestCase):
    def test_many_unsorted_iso_dates_use_complete_future_holdout(self):
        dates = pd.date_range('2025-01-01', periods=200, tz='UTC')
        order = np.random.default_rng(42).permutation(200)
        frame = pd.DataFrame({'event_date': dates.strftime('%Y-%m-%d')[order],
                              'x': np.arange(200)})
        p = problem(frame, pd.DataFrame({'event_date': ['2025-08-01'], 'x': [999]}),
                    text='event date; predict future observations')
        plan = plan_validation(p)
        self.assertEqual(plan['strategy'], 'time')
        a = pd.to_datetime(frame.iloc[plan['train_indices']].event_date)
        b = pd.to_datetime(frame.iloc[plan['valid_indices']].event_date)
        self.assertLess(a.max(), b.min())
        self.assertEqual(len(b), 40)

    def test_more_than_one_hundred_numeric_timestamps_are_temporal(self):
        p = problem(pd.DataFrame({'timestamp': np.arange(200)}),
                    pd.DataFrame({'timestamp': [200, 201]}), text='timestamp of observation')
        plan = plan_validation(p)
        self.assertEqual(plan['strategy'], 'time')
        np.testing.assert_array_equal(plan['valid_indices'], np.arange(160, 200))

    def test_invalid_or_mixed_dates_are_not_guessed(self):
        for bad in ('2025-02-30', 'not-a-date', '01/02/2025', None):
            dates = pd.date_range('2025-01-01', periods=20).strftime('%Y-%m-%d').tolist()
            dates[3] = bad
            p = problem(pd.DataFrame({'event_date': dates, 'x': np.arange(20)}),
                        pd.DataFrame({'event_date': ['2026-01-01'], 'x': [99]}), text='event date')
            with self.subTest(bad=bad):
                self.assertEqual(plan_validation(p)['strategy'], 'random')
                self.assertIsNone(time_values(p['train'].event_date))

    def test_explicit_iso_offsets_sort_by_instant_not_lexical_text(self):
        dates = ['2025-01-01T00:00:00Z'] * 4 + ['2025-01-01T01:30:00+02:00'] * 4
        dates += ['2025-01-01T00:30:00Z'] * 2
        p = problem(pd.DataFrame({'event_date': dates}))
        plan = plan_validation(p, {'split': 'time', 'time_column': 'event_date'})
        np.testing.assert_array_equal(plan['valid_indices'], [8, 9])
        parsed = time_values(p['train'].event_date)
        self.assertLess(parsed.iloc[plan['train_indices']].max(),
                        parsed.iloc[plan['valid_indices']].min())

    def test_iso_time_split_preserves_future_new_class_safety(self):
        dates = pd.date_range('2025-01-01', periods=200).strftime('%Y-%m-%d')
        p = problem(pd.DataFrame({'event_date': dates}),
                    pd.DataFrame({'event_date': ['2026-01-01']}),
                    labels=[0] * 160 + [1] * 40, text='event date')
        plan = plan_validation(p)
        self.assertEqual(plan['strategy'], 'time')
        self.assertIsNone(plan['train_indices'])

    def test_time_diagnostics_are_bounded_for_many_unique_periods(self):
        p = problem(pd.DataFrame({'timestamp': np.arange(1000)}),
                    pd.DataFrame({'timestamp': [1001]}), text='timestamp of observation')
        plan = plan_validation(p)
        self.assertEqual(plan['diagnostics']['validation_period_count'], 200)
        self.assertEqual(len(plan['diagnostics']['validation_periods']), 100)

    def test_target_cannot_define_time_or_group_folds(self):
        p = problem(pd.DataFrame({'x': range(20), 'target': range(20)}))
        p['target'] = 'target'
        p['features'] = ['x']
        for request in ({'time_column': 'target'}, {'group_columns': ['target']}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                plan_validation(p, request)

    def test_group_columns_may_use_an_id_excluded_from_features(self):
        frame = pd.DataFrame({'patient_id': np.repeat(range(10), 2), 'x': range(20)})
        p = problem(frame)
        p['features'] = ['x']
        plan = plan_validation(p, {'group_columns': ['patient_id']})
        a = set(frame.iloc[plan['train_indices']].patient_id)
        b = set(frame.iloc[plan['valid_indices']].patient_id)
        self.assertFalse(a & b)

    def test_future_new_class_omits_score_instead_of_leaking_future_rows(self):
        p = problem(pd.DataFrame({'year': [1] * 8 + [2] * 2}),
                    pd.DataFrame({'year': [3]}), labels=[0] * 8 + [1] * 2,
                    text='year of observation')
        plan = plan_validation(p)
        self.assertEqual(plan['strategy'], 'time')
        self.assertIsNone(plan['train_indices'])
        self.assertIsNone(plan['valid_indices'])
        with self.assertRaises(ValueError):
            plan_validation(p, {'split': 'time', 'time_column': 'year'})

    def test_unsorted_future_period_is_complete_holdout(self):
        p = problem(pd.DataFrame({"year_code": [1, 5, 2, 4, 3] * 4, "x": range(20)}),
                    pd.DataFrame({"year_code": [6, 6], "x": [99, 100]}), text="year of observation")
        plan = plan_validation(p)
        self.assertEqual(plan["strategy"], "time")
        self.assertEqual(set(p["train"].iloc[plan["valid_indices"]].year_code), {5})
        self.assertTrue((p["train"].iloc[plan["train_indices"]].year_code < 5).all())

    def test_duplicate_vectors_never_cross_folds(self):
        p = problem(pd.DataFrame({"x": np.repeat(range(20), 3)}), labels=[0, 1, 1] * 20)
        plan = plan_validation(p)
        self.assertEqual(plan["strategy"], "group")
        self.assertFalse(set(p["train"].iloc[plan["train_indices"]].x) &
                         set(p["train"].iloc[plan["valid_indices"]].x))

    def test_model_seed_does_not_change_validation(self):
        p = problem(pd.DataFrame({"x": np.repeat(range(20), 3)}), labels=[0, 1, 1] * 20)
        a = plan_validation(p, {"random_seed": 1})
        b = plan_validation(p, {"random_seed": 99})
        np.testing.assert_array_equal(a["valid_indices"], b["valid_indices"])

    def test_arbitrary_numeric_shift_does_not_imply_time(self):
        p = problem(pd.DataFrame({"temperature": range(20)}),
                    pd.DataFrame({"temperature": [99, 100]}), text="observed year")
        self.assertEqual(plan_validation(p)["strategy"], "random")

    def test_explicit_random_overrides_duplicate_policy(self):
        p = problem(pd.DataFrame({"x": np.repeat(range(20), 3)}), labels=[0, 1, 1] * 20)
        self.assertEqual(plan_validation(p, {"split": "random"})["strategy"], "random")

    def test_group_keeps_rare_class_in_training(self):
        p = problem(pd.DataFrame({"x": np.repeat(range(10), 2)}), labels=[1] + [0] * 19)
        plan = plan_validation(p, {"split": "group"})
        self.assertIn(0, plan["train_indices"])

    def test_explicit_time_without_column_uses_order(self):
        p = problem(pd.DataFrame({"x": range(20)}))
        np.testing.assert_array_equal(plan_validation(p, {"split": "time"})["valid_indices"], [16, 17, 18, 19])

    def test_impossible_group_split_fails(self):
        p = problem(pd.DataFrame({"x": [1] * 20}))
        with self.assertRaises(ValueError):
            plan_validation(p, {"split": "group"})

    def test_impossible_automatic_group_returns_unscored_plan(self):
        p = problem(pd.DataFrame({"x": [1] * 20}))
        plan = plan_validation(p)
        self.assertIsNone(plan["valid_indices"])
        self.assertIn("No independent holdout", plan["rationale"])

    def test_automatic_group_cannot_remove_only_class_examples(self):
        p = problem(pd.DataFrame({"x": [0, 0, 1, 1, 2, 2]}), labels=[0, 0, 1, 1, 2, 2])
        self.assertIsNone(plan_validation(p)["valid_indices"])

    def test_group_columns_select_group_in_auto_mode(self):
        p = problem(pd.DataFrame({"subject": np.repeat(range(10), 2), "x": range(20)}))
        plan = plan_validation(p, {"group_columns": ["subject"]})
        self.assertEqual(plan["strategy"], "group")
        self.assertFalse(set(p["train"].iloc[plan["train_indices"]].subject) &
                         set(p["train"].iloc[plan["valid_indices"]].subject))

    def test_conflicting_options_rejected(self):
        p = problem(pd.DataFrame({"x": range(20)}))
        for request in ({"split": "random", "time_column": "x"},
                        {"split": "time", "group_columns": ["x"]},
                        {"time_column": "x", "group_by_features": True}):
            with self.assertRaises(ValueError):
                plan_validation(p, request)

    def test_seen_diagnostics_explain_duplicate_grouping(self):
        p = problem(pd.DataFrame({"x": np.repeat(range(20), 3)}))
        plan = plan_validation(p)
        self.assertEqual(plan["diagnostics"]["test_seen_fraction"], 1)
        self.assertEqual(plan["diagnostics"]["validation_seen_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
