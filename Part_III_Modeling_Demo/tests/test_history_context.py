import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as web
from core import REQUIRED, score_transactions, validate_input
from history import HISTORY_VERSION, add_history, history_with_context, validate_context_period
from train import make_pipeline
from test_history import stream


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = validate_input(stream(200), history=True)
        cls.context, cls.review = cls.raw.iloc[:100], cls.raw.iloc[100:]
        pipeline = make_pipeline(LogisticRegression(solver='liblinear'), history=True)
        pipeline.fit(cls.raw.join(add_history(cls.raw)), np.arange(200) % 7 == 0)
        cls.saved = {'pipeline': pipeline, 'threshold': .5, 'metadata': {'history_version': HISTORY_VERSION}}

    def test_simultaneous_rows_see_only_earlier_timestamp(self):
        frame = self.raw.iloc[:4].copy()
        frame['cc_num'] = 'same-card'
        frame['trans_date_trans_time'] = ['2020-01-01 00:00:00', '2020-01-01 01:00:00',
                                        '2020-01-01 01:00:00', '2020-01-01 02:00:00']
        frame['amt'] = [10., 20., 100., 30.]
        frame['merchant'] = ['old', 'new', 'new', 'new']
        frame['category'] = ['old', 'new', 'new', 'new']
        features = add_history(frame)
        self.assertEqual(features.hist_prior_count.tolist(), [0, 1, 1, 3])
        self.assertEqual(features.hist_count_24h.tolist(), [0, 1, 1, 3])
        np.testing.assert_allclose(features.hist_amt_to_prior_mean.iloc[1:3], [2, 10])
        self.assertEqual(features.hist_new_merchant.tolist(), [1, 1, 1, 0])
        pd.testing.assert_frame_equal(features, add_history(frame.iloc[::-1]).reindex(frame.index))
        modified = frame.copy()
        modified.loc[2, 'amt'] = 10000.
        pd.testing.assert_series_equal(features.iloc[1], add_history(modified).iloc[1])

    def test_all_simultaneous_rows_have_no_prior_history(self):
        frame = self.raw.iloc[:3].copy()
        frame['cc_num'] = 'same'
        frame['trans_date_trans_time'] = '2020-01-01 00:00:00'
        features = add_history(frame)
        self.assertTrue(features.hist_prior_count.eq(0).all())
        self.assertTrue(features.hist_amt_to_prior_mean.eq(1).all())
        self.assertTrue(features.hist_dist_from_prev_km.eq(0).all())

    def test_upload_and_validation_scores_are_identical_with_same_context(self):
        expected = self.saved['pipeline'].predict_proba(self.review[REQUIRED].join(
            history_with_context(self.context, self.review)))[:, 1]
        actual = score_transactions(self.review, self.saved, self.context).sort_values('row_number')
        np.testing.assert_allclose(expected, actual.model_score, rtol=0, atol=0)
        self.assertNotIn('cc_num', actual.columns)

    def test_future_and_overlapping_context_rejected(self):
        for context in [self.review, self.raw.iloc[190:]]:
            with self.assertRaises(ValueError):
                validate_context_period(context, self.review)
        with self.assertRaises(ValueError):
            score_transactions(self.review, self.saved)

    def test_explicit_cold_start_and_unknown_card(self):
        empty = self.context.iloc[:0]
        cold = score_transactions(self.review, self.saved, empty)
        self.assertTrue(cold.prior_transactions.eq(0).any())
        unknown = self.review.iloc[:1].assign(cc_num='new-card')
        result = score_transactions(unknown, self.saved, self.context)
        self.assertEqual(result.iloc[0].customer_status, 'No prior history')

    def test_labels_and_unused_fields_do_not_affect_history_scores(self):
        a = score_transactions(self.review.assign(is_fraud=0), self.saved,
                               self.context.assign(is_fraud=0, first='PRIVATE'))
        b = score_transactions(self.review.assign(is_fraud=1), self.saved,
                               self.context.assign(is_fraud=1, first='OTHER'))
        np.testing.assert_array_equal(a.model_score, b.model_score)

    def test_history_api_is_session_scoped_and_replacement_failure_clears_context(self):
        with patch.dict(web._bundles, {'history': self.saved}, clear=True), patch.dict(web._contexts, {}, clear=True):
            one, two = TestClient(web.app), TestClient(web.app)
            context_csv = self.context.to_csv(index=False)
            review_csv = self.review.to_csv(index=False)
            response = one.post('/api/history/context', files={'file': ('history.csv', context_csv)})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(one.get('/api/history/context').json()['loaded'])
            self.assertFalse(two.get('/api/history/context').json()['loaded'])
            denied = two.post('/api/analyze?model=history', files={'file': ('new.csv', review_csv)})
            self.assertEqual(denied.status_code, 400)
            valid = one.post('/api/analyze?model=history', files={'file': ('new.csv', review_csv)})
            self.assertEqual(valid.status_code, 200, valid.text)
            repeated = one.post('/api/analyze?model=history', files={'file': ('new.csv', review_csv)})
            self.assertEqual(valid.json(), repeated.json())
            overlap = one.post('/api/analyze?model=history', files={'file': ('same.csv', context_csv)})
            self.assertEqual(overlap.status_code, 400)
            invalid = one.post('/api/history/context', files={'file': ('bad.csv', 'amt\n-1\n')})
            self.assertEqual(invalid.status_code, 400)
            self.assertFalse(one.get('/api/history/context').json()['loaded'])
            self.assertNotIn('rows', invalid.json())
            two.post('/api/history/cold-start')
            allowed = two.post('/api/analyze?model=history', files={'file': ('new.csv', review_csv)})
            self.assertEqual(allowed.status_code, 200, allowed.text)
            two.delete('/api/history/context')
            self.assertFalse(two.get('/api/history/context').json()['loaded'])

    def test_multiple_history_files_deduplicate_ids_and_keep_distinct_purchases(self):
        history = self.context.copy()
        history['trans_num'] = [f'id-{i}' for i in range(len(history))]
        with patch.dict(web._contexts, {}, clear=True):
            client = TestClient(web.app)
            result = client.post('/api/history/context', files=[
                ('file', ('first.csv', history.iloc[:70].to_csv(index=False))),
                ('file', ('second.csv', history.iloc[50:].to_csv(index=False)))])
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()['rows'], 100)
            self.assertEqual(result.json()['duplicates_removed'], 20)
            self.assertEqual(len(result.json()['files']), 2)
            conflict = history.iloc[:1].assign(amt=9999)
            bad = client.post('/api/history/context', files=[
                ('file', ('first.csv', history.to_csv(index=False))),
                ('file', ('conflict.csv', conflict.to_csv(index=False)))])
            self.assertEqual(bad.status_code, 400)
            self.assertFalse(client.get('/api/history/context').json()['loaded'])
            same = history.iloc[:1].copy()
            separate = same.assign(trans_num='another-transaction')
            result = client.post('/api/history/context', files=[
                ('file', ('one.csv', same.to_csv(index=False))),
                ('file', ('two.csv', separate.to_csv(index=False)))])
            self.assertEqual(result.json()['rows'], 2)

    def test_sequential_history_uploads_append_and_invalid_addition_keeps_history(self):
        history = self.context.copy()
        history['trans_num'] = [f'id-{i}' for i in range(len(history))]
        with patch.dict(web._contexts, {}, clear=True):
            client = TestClient(web.app)
            first = client.post('/api/history/context?append=true', files={
                'file': ('first.csv', history.iloc[:70].to_csv(index=False))})
            self.assertEqual(first.json()['rows'], 70)
            second = client.post('/api/history/context?append=true', files={
                'file': ('second.csv', history.iloc[50:].to_csv(index=False))})
            self.assertEqual(second.status_code, 200, second.text)
            self.assertEqual(second.json()['rows'], 100)
            self.assertEqual(second.json()['duplicates_removed'], 20)
            self.assertEqual(len(second.json()['files']), 2)
            bad = client.post('/api/history/context?append=true', files={
                'file': ('bad.csv', 'amt\nwrong\n')})
            self.assertEqual(bad.status_code, 400)
            self.assertEqual(client.get('/api/history/context').json()['rows'], 100)
            client.delete('/api/history/context')
            self.assertFalse(client.get('/api/history/context').json()['loaded'])

    def test_expired_context_is_not_reused(self):
        with patch.dict(web._contexts, {}, clear=True):
            client = TestClient(web.app)
            client.post('/api/history/cold-start')
            with patch('app.time.monotonic', return_value=web.time.monotonic() + web.CONTEXT_TTL + 1):
                self.assertFalse(client.get('/api/history/context').json()['loaded'])

    def test_original_csv_with_unnamed_export_index_is_accepted(self):
        with patch.dict(web._contexts, {}, clear=True):
            client = TestClient(web.app)
            result = client.post('/api/history/context', files={'file': (
                'fraudTrain.csv', self.context.to_csv(index=True))})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()['rows'], len(self.context))

    def test_new_models_cannot_silently_reopen_observed_test(self):
        import evaluate as evaluation
        with patch.object(evaluation, 'load_bundle', return_value={'metadata': {'selected_model': 'Gradient Boosting'}}), \
                patch.object(evaluation.pd, 'read_csv') as read:
            with self.assertRaisesRegex(ValueError, 'already observed'):
                evaluation.evaluate(artifacts=Path(__file__).parent / 'not_created')
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
