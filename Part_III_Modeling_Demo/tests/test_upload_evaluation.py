"""Uploaded-label evaluation must align after score sorting and never affect scores."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import app as web
import test_project as fixtures


class UploadEvaluationTests(unittest.TestCase):
    def test_confusion_counts_follow_original_row_numbers(self):
        labels = pd.Series([1, 0, 1, 0])
        result = pd.DataFrame({'row_number': [4, 1, 3, 2], 'review_flag': [False, True, False, True]})
        report = web.upload_evaluation(labels, result)
        for field in ['true_positives', 'false_positives', 'false_negatives', 'true_negatives']:
            self.assertEqual(report[field], 1)
        for field in ['precision', 'recall', 'f1', 'accuracy']:
            self.assertEqual(report[field], .5)
        self.assertEqual(result.evaluation_result.tolist(), ['Correctly cleared', 'Correct alert', 'Missed fraud', 'False alert'])

    def test_undefined_precision_and_recall_are_not_fake_zeros(self):
        result = pd.DataFrame({'row_number': [1, 2], 'review_flag': [False, False]})
        report = web.upload_evaluation(pd.Series([0, 0]), result)
        self.assertIsNone(report['precision'])
        self.assertIsNone(report['recall'])
        self.assertIsNone(report['f1'])
        self.assertEqual(report['accuracy'], 1)

    def test_upload_labels_are_optional_validated_and_do_not_change_predictions(self):
        fixtures.ProjectTests.setUpClass()
        with patch.object(web, '_bundle', fixtures.ProjectTests.saved):
            client = TestClient(web.app)
            raw = fixtures.ProjectTests.frame.copy()
            def upload(frame):
                return client.post('/api/analyze', files={'file': ('review.csv', frame.to_csv(index=False))})
            plain = upload(raw).json()
            self.assertIsNone(plain['evaluation'])
            zero, one = upload(raw.assign(is_fraud=0)).json(), upload(raw.assign(is_fraud=1)).json()
            for labeled in [zero, one]:
                np.testing.assert_array_equal([row['model_score'] for row in plain['rows']],
                                              [row['model_score'] for row in labeled['rows']])
                self.assertEqual(sum(labeled['evaluation'][key] for key in
                    ['true_positives', 'false_positives', 'false_negatives', 'true_negatives']), len(raw))
            self.assertEqual(one['evaluation']['actual_fraud'], len(raw))
            for bad in [2, 'fraud', None]:
                response = upload(raw.assign(is_fraud=bad))
                self.assertEqual(response.status_code, 400)
                self.assertNotIn('rows', response.json())
