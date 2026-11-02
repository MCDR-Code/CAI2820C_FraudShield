
# Discover the project from any supported entry point; retain saved-model imports.
import sys as _sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / 'project_paths.py').is_file())
if str(_project) not in _sys.path:
    _sys.path.insert(0, str(_project))
from project_paths import PROJECT_ROOT
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as web
from test_history import stream


class HistoryFileTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web.app)
        self.frame = stream(20).assign(trans_num=[f't{i}' for i in range(20)])

    def test_unload_overlapping_file_preserves_other_file_rows(self):
        raw = self.frame.to_csv(index=False)
        result = self.client.post('/api/history/context', files=[
            ('file', ('first.csv', raw)), ('file', ('second.csv', raw))])
        self.assertEqual(result.status_code, 200, result.text)
        details = result.json()['file_details']
        self.assertEqual(len(details), 2)
        result = self.client.delete('/api/history/files/' + details[0]['id'])
        self.assertEqual(result.json()['rows'], 20)
        self.assertEqual(result.json()['files'], ['second.csv'])
        self.assertEqual(self.client.delete('/api/history/files/' + details[0]['id']).status_code, 404)
        result = self.client.delete('/api/history/files/' + details[1]['id'])
        self.assertFalse(result.json()['loaded'])

    def test_server_history_reads_only_allowlisted_predictors_and_preserves_on_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'transactions_2025.csv'
            self.frame.assign(is_fraud='invalid labels intentionally ignored').to_csv(path, index=False)
            with patch.object(web, 'SERVER_HISTORY', {'2025': path}):
                result = self.client.post('/api/history/server-context')
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()['rows'], 20)
            with patch.object(web, 'SERVER_HISTORY', {'2025': Path(temp) / 'missing.csv'}):
                self.assertEqual(self.client.post('/api/history/server-context').status_code, 400)
            self.assertEqual(self.client.get('/api/history/context').json()['rows'], 20)
        self.assertNotIn('2026', web.SERVER_HISTORY)


if __name__ == '__main__':
    unittest.main()
