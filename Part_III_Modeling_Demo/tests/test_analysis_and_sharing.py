import copy
import json
import platform
import sys
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_errors
from core import required_columns, validate_input
from history import HISTORY_VERSION, add_history
from model_sharing import check_model, record_model_files, source_hash
from shared.project_data import sha256
from train import make_pipeline
from test_history import stream


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = validate_input(stream(200), history=True)
        cls.raw['is_fraud'] = (np.arange(200) % 7 == 0).astype(int)
        cls.fit, cls.review = cls.raw.iloc[:160].copy(), cls.raw.iloc[160:].copy()
        cls.pipeline = make_pipeline(LogisticRegression(solver='liblinear'), history=True)
        cls.pipeline.fit(cls.fit.join(add_history(cls.fit)), cls.fit.is_fraud)
        cls.environment = {'python': platform.python_version(), **{
            key: version(package) for key, package in [('pandas', 'pandas'), ('numpy', 'numpy'),
                ('scipy', 'scipy'), ('scikit_learn', 'scikit-learn'), ('joblib', 'joblib')]}}

    def fixture(self, directory):
        raw = directory / 'data'
        artifacts = directory / 'model'
        raw.mkdir(); artifacts.mkdir()
        self.raw.to_csv(raw / 'fraudTrain.csv', index=False)
        locked = {'history_features': True, 'history_version': HISTORY_VERSION,
                  'selected_model': 'Logistic Regression', 'threshold': .5,
                  'feature_count': len(self.pipeline['preprocess'].get_feature_names_out()),
                  'environment': self.environment,
                  'data': {'train_sha256': sha256(raw / 'fraudTrain.csv'),
                           'training_rows': 160, 'validation_rows': 40}}
        saved = {'pipeline': self.pipeline, 'threshold': .5, 'metadata': locked}
        joblib.dump(saved, artifacts / 'selected_model.joblib')
        (artifacts / 'experiment.json').write_text(json.dumps(locked), encoding='utf-8')
        self.review[required_columns(True)].to_csv(artifacts / 'demo_transactions.csv', index=False)
        self.fit[required_columns(True)].to_csv(artifacts / 'demo_history.csv', index=False)
        (artifacts / 'validation_comparison.csv').write_text('model,precision\nfixture,0.5\n')
        (artifacts / 'validation_pr.png').write_bytes(b'fixture-not-an-image')
        return raw, artifacts, locked

    def test_error_analysis_uses_recorded_validation_split(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, artifacts, _ = self.fixture(Path(temp))
            result = analyze_errors.validation_errors(artifacts, raw)
            self.assertEqual(len(result), 40)
            self.assertTrue({'score', 'flag', 'is_fraud'}.issubset(result.columns))

    def test_analysis_rejects_stale_version_before_reading_csv(self):
        saved = {'metadata': {'history_version': 'strict-timestamp-groups-v2'}, 'pipeline': self.pipeline}
        with patch.object(analyze_errors, 'load_bundle', return_value=saved), \
                patch.object(analyze_errors, 'development_split') as read:
            with self.assertRaisesRegex(ValueError, 'version'):
                analyze_errors.validation_errors('unused', 'unused')
            read.assert_not_called()

    def test_analysis_rejects_changed_source_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, artifacts, locked = self.fixture(Path(temp))
            original = (raw / 'fraudTrain.csv').read_bytes()
            (raw / 'fraudTrain.csv').write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                analyze_errors.validation_errors(artifacts, raw)
            (raw / 'fraudTrain.csv').write_bytes(original)
            altered = copy.deepcopy(locked)
            altered['threshold'] = .9
            (artifacts / 'experiment.json').write_text(json.dumps(altered), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'metadata'):
                analyze_errors.validation_errors(artifacts, raw)

    def test_shared_model_checks_copy_environment_and_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            _, artifacts, locked = self.fixture(Path(temp))
            record_model_files(artifacts, locked, self.pipeline, self.review, self.fit)
            result = check_model(artifacts)
            self.assertEqual(result['verified_demo_rows'], 40)
            self.assertEqual(result['history_version'], HISTORY_VERSION)
            with (artifacts / 'selected_model.joblib').open('ab') as file:
                file.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                check_model(artifacts)

    def test_source_hash_allows_windows_and_mac_line_endings(self):
        with tempfile.TemporaryDirectory() as temp:
            one, two = Path(temp) / 'one.py', Path(temp) / 'two.py'
            one.write_bytes(b'x = 1\r\ny = 2\r\n')
            two.write_bytes(b'x = 1\ny = 2\n')
            self.assertEqual(source_hash(one), source_hash(two))


if __name__ == '__main__':
    unittest.main()
