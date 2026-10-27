"""A small real fit verifies new-data opt-in without retraining project models."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from test_history import stream
from train import train
from model_sharing import check_model


class ExpandedTrainingTests(unittest.TestCase):
    def test_expanded_training_without_future_test_is_explicit_and_shareable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = stream(500).copy()
            data['is_fraud'] = (np.arange(len(data)) % 7 == 0).astype(int)
            data['trans_num'] = [f'id-{i}' for i in range(len(data))]
            data.to_csv(root / 'fraudTrain.csv', index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'full original'):
                    train(root, root / 'original_mode')
                metadata, results = train(root, root / 'new_experiment',
                    candidates_to_fit=('Logistic Regression',), expanded_data=True)
            self.assertIsNone(metadata['data']['test_rows'])
            self.assertEqual(metadata['data']['dataset_mode'], 'expanded')
            self.assertFalse(metadata['data']['test_labels_loaded'])
            self.assertEqual(metadata['data']['training_rows'] + metadata['data']['validation_rows'], 500)
            self.assertEqual(check_model(root / 'new_experiment')['model'], 'Logistic Regression')
