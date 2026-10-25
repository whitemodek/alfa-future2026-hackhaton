"""Independent grouped outer tests and immutable input/answer boundaries."""
from contextlib import nullcontext
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qa.transfer_benchmark import outer_split, prepare_case, run_case


class TransferSplitTests(unittest.TestCase):
    def test_mixed_label_duplicates_are_never_shared_between_outer_folds(self):
        # Each repeated feature vector has both labels, so row stratification
        # would leak a memorisable key even though class counts look correct.
        frame = pd.DataFrame({'key': np.repeat(np.arange(30), 4), 'constant': 1})
        target = pd.Series(np.tile([0, 0, 1, 1], 30))
        original = frame.copy(deep=True)
        original_target = target.copy(deep=True)
        train, test, metadata = outer_split(frame, target, True)
        self.assertFalse(set(frame.iloc[train].key) & set(frame.iloc[test].key))
        self.assertEqual(set(target.iloc[train]), {0, 1})
        self.assertEqual(set(target.iloc[test]), {0, 1})
        self.assertEqual(set(train) | set(test), set(range(len(frame))))
        self.assertFalse(set(train) & set(test))
        again_train, again_test, _ = outer_split(frame, target, True)
        np.testing.assert_array_equal(train, again_train)
        np.testing.assert_array_equal(test, again_test)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_series_equal(target, original_target)
        self.assertEqual(metadata['method'], 'stratified_group')

    def test_rare_class_confined_to_one_group_cannot_leak_to_preserve_classes(self):
        frame = pd.DataFrame({'key': np.repeat(np.arange(10), 5)})
        target = pd.Series(np.repeat([0] * 9 + [1], 5))
        with self.assertRaisesRegex(ValueError, 'preserve every class'):
            outer_split(frame, target, True)


class TransferImmutabilityTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / '.test-temp'
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        source = self.root / 'source.csv'
        pd.DataFrame({'x': np.arange(100), 'target': np.arange(100) * 2}).to_csv(source, index=False)
        self.spec = dict(name='contract_case', path=str(source), target='target',
                         problem_type='regression', metric='rmse', source_url='https://example.org/dataset')
        self.folder = prepare_case(self.spec, self.root)

    def test_preparation_is_frozen_and_outer_labels_live_outside_worker_inputs(self):
        before = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        self.assertEqual(set(before), {'train.csv', 'test.csv', 'sample_submission.csv', 'task.md'})
        self.assertNotIn('target', pd.read_csv(self.folder / 'test.csv'))
        self.assertTrue((self.folder.parent / '_answers' / 'contract_case.csv').is_file())
        self.assertEqual(prepare_case(self.spec, self.root), self.folder)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.folder.iterdir()})

    def test_added_input_or_subdirectory_is_rejected_before_it_can_reach_worker(self):
        for name in ['hidden_labels.csv', 'private']:
            path = self.folder / name
            if name.endswith('.csv'):
                path.write_text('id,target\n1,12345\n', encoding='utf-8')
            else:
                path.mkdir()
            try:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Frozen benchmark input'):
                    prepare_case(self.spec, self.root)
            finally:
                path.unlink() if path.is_file() else path.rmdir()

    def test_changed_hidden_answer_csv_and_metadata_are_rejected(self):
        for suffix in ['csv', 'json']:
            path = self.folder.parent / '_answers' / ('contract_case.' + suffix)
            before = path.read_bytes()
            path.write_bytes(before + b'\n')
            try:
                with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, 'Frozen hidden answer'):
                    prepare_case(self.spec, self.root)
            finally:
                path.write_bytes(before)

    def test_changed_original_source_or_frozen_train_is_rejected(self):
        source = Path(self.spec['path'])
        content = source.read_bytes()
        source.write_bytes(content + b'\n')
        try:
            with self.assertRaisesRegex(ValueError, 'Frozen dataset/split'):
                prepare_case(self.spec, self.root)
        finally:
            source.write_bytes(content)
        path = self.folder / 'train.csv'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'Frozen benchmark input'):
            prepare_case(self.spec, self.root)

    def test_mutated_worker_copy_is_rejected_even_when_source_stays_frozen(self):
        archive = self.root / 'unused-solution.zip'
        archive.write_bytes(b'Worker is mocked; no archive execution or training')
        worker = self.root / 'worker'

        def changed_worker(_archive, folder, _limit):
            data = worker / 'data'
            shutil.copytree(folder, data)
            train = data / 'train.csv'
            train.write_bytes(train.read_bytes() + b'\n')
            return dict(work_dir=str(worker), success=True)

        with patch('qa.transfer_benchmark.finish_provider',
                   return_value=nullcontext(('http://unused.local/v1', []))), \
                patch('qa.transfer_benchmark.run_task', side_effect=changed_worker):
            with self.assertRaisesRegex(ValueError, 'copied worker input changed'):
                run_case(self.spec, self.root, archive, 30, 1)

    def test_reserved_answer_directory_and_unsupported_metrics_cannot_be_prepared(self):
        with self.assertRaisesRegex(ValueError, 'safe identifiers'):
            prepare_case(dict(self.spec, name='_answers'), self.root)
        with self.assertRaisesRegex(ValueError, 'evaluator supports'):
            prepare_case(dict(self.spec, name='wrong_metric', metric='log_loss'), self.root)


if __name__ == '__main__':
    unittest.main()
