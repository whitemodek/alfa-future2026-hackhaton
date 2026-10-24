import unittest

import numpy as np
import pandas as pd

from ml_validation import plan_validation


def problem(frame, test=None, labels=None, text=""):
    return dict(train=frame, test=frame.copy() if test is None else test,
                features=list(frame), y=pd.Series(np.arange(len(frame)) if labels is None else labels),
                classes=None if labels is None else sorted(set(labels)), text=text)


class ValidationPlannerTests(unittest.TestCase):
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
