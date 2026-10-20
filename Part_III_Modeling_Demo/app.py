"""Local-only demo. Uploads are validated and scored in memory, never saved."""

import csv
import io
import json
from pathlib import Path
from threading import Lock

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from core import MAX_BYTES, MAX_ROWS, REQUIRED, ROOT, load_bundle, score_transactions, validate_input

app = FastAPI(title='FraudShield', docs_url=None, redoc_url=None)
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
_bundle = None
_lock = Lock()


def bundle():
    global _bundle
    with _lock:
        if _bundle is None:
            _bundle = load_bundle()
    return _bundle


def analyze(df, source):
    if len(df) > MAX_ROWS:
        raise ValueError(f'Maximum batch size is {MAX_ROWS:,} transactions.')
    validate_input(df)
    result = score_transactions(df, bundle())
    return {'source': source, 'rows': result.to_dict('records'),
            'total': len(result), 'flagged': int(result.review_flag.sum()),
            'flagged_amount': float(result.loc[result.review_flag, 'amount_usd'].sum()),
            'threshold': bundle()['threshold']}


@app.get('/')
def index():
    return FileResponse(ROOT / 'static/index.html')


@app.get('/api/metadata')
def metadata():
    try:
        saved = bundle()['metadata']
        artifacts = ROOT / 'artifacts'
        test = artifacts / 'final_test.json'
        return {'model': saved['selected_model'], 'threshold': saved['threshold'],
                'features': saved['feature_count'], 'created': saved['created_utc'],
                'data': saved['data'], 'limitations': saved['limitations'],
                'validation': pd.read_csv(artifacts / 'validation_comparison.csv').to_dict('records'),
                'final_test': json.loads(test.read_text()) if test.exists() else None,
                'required_columns': REQUIRED}
    except FileNotFoundError as exc:
        return JSONResponse({'error': str(exc)}, status_code=503)


@app.post('/api/sample')
def sample():
    try:
        df = pd.read_csv(ROOT / 'artifacts/demo_transactions.csv')
        return analyze(df, 'Validation demo sample (1,000 transactions)')
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


@app.post('/api/analyze')
def upload(file: UploadFile = File(...)):
    try:
        filename = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
        if not filename.lower().endswith('.csv'):
            raise ValueError('Choose a .csv file.')
        raw = file.file.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('CSV exceeds the 20 MB upload limit.')
        if not raw:
            raise ValueError('The uploaded file is empty.')
        text = raw.decode('utf-8-sig')
        header = next(csv.reader(io.StringIO(text)), [])
        if not header or len(set(header)) != len(header):
            raise ValueError('CSV column names must be nonempty and unique.')
        df = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS+1, dtype={'cc_num': 'string'})
        return analyze(df, filename)
    except (ValueError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, FileNotFoundError) as exc:
        # An invalid upload never substitutes the validation sample.
        return JSONResponse({'error': str(exc)}, status_code=400)
    finally:
        file.file.close()


@app.get('/api/validation-pr.png')
def validation_chart():
    path = ROOT / 'artifacts/validation_pr.png'
    if not path.exists():
        return JSONResponse({'error': 'No validation chart is available.'}, status_code=404)
    return FileResponse(path, media_type='image/png')
