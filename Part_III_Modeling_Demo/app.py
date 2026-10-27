"""Local demo with separate models and browser-session history held only in RAM."""

import csv
import io
import json
import secrets
import time
from threading import Lock

import pandas as pd
from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core import MAX_BYTES, MAX_ROWS, ROOT, bundle_uses_history, load_bundle, required_columns, score_transactions, validate_input
from history import HISTORY_VERSION

app = FastAPI(title='FraudShield', docs_url=None, redoc_url=None)
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
MODEL_NAMES = {'baseline': 'Random Forest baseline', 'boosting': 'Gradient Boosting',
               'history': 'Gradient Boosting + customer history'}
MAX_HISTORY_ROWS = 2_000_000
MAX_HISTORY_BYTES = 512 * 1024 * 1024
CONTEXT_TTL = 3600
_bundle = None
_bundles = {}
_contexts = {}
_lock = Lock()


def artifact_dir(model):
    if model not in MODEL_NAMES:
        raise ValueError('Unknown model.')
    directory = ROOT / 'artifacts' / ('history_v3' if model == 'history' else model)
    if model == 'baseline' and not directory.exists():
        return ROOT / 'artifacts'
    return directory


def bundle(model='baseline'):
    global _bundle
    with _lock:
        if model == 'baseline' and _bundle is not None:
            return _bundle
        if model not in _bundles:
            saved = load_bundle(artifact_dir(model) / 'selected_model.joblib')
            if model == 'history' and (saved['metadata'].get('history_version') != HISTORY_VERSION
                                       or saved['pipeline']['features'].history != 'precomputed'):
                raise ValueError('History artifact is incompatible. Train the corrected history model first.')
            _bundles[model] = saved
        return _bundles[model]


def read_upload(file, history=False):
    try:
        filename = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
        if not filename.lower().endswith('.csv'):
            raise ValueError('Choose a .csv file.')
        limit = MAX_HISTORY_BYTES if history else MAX_BYTES
        raw = file.file.read(limit + 1)
        if len(raw) > limit:
            raise ValueError(f'CSV exceeds the {limit // (1024 * 1024)} MB limit.')
        if not raw:
            raise ValueError('The uploaded file is empty.')
        text = raw.decode('utf-8-sig')
        header = next(csv.reader(io.StringIO(text)), [])
        # Sparkov's original CSV exports its row index with an empty first heading.
        # Pandas names that unused field Unnamed: 0; other blank headings remain errors.
        if header and header[0] == '':
            header[0] = 'Unnamed: 0'
        if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
            raise ValueError('CSV column names must be nonempty and unique.')
        maximum = MAX_HISTORY_ROWS if history else MAX_ROWS
        frame = pd.read_csv(io.StringIO(text), nrows=maximum + 1, dtype={'cc_num': 'string'})
        if len(frame) > maximum:
            raise ValueError(f'Maximum batch size is {maximum:,} transactions.')
        return frame, filename
    finally:
        file.file.close()


def session_context(request):
    token = request.cookies.get('fraudshield_history')
    with _lock:
        for key in list(_contexts):
            if time.monotonic() - _contexts[key]['time'] > CONTEXT_TTL:
                del _contexts[key]
        return _contexts.get(token)


def context_summary(context):
    if context is None:
        return {'loaded': False}
    frame = context['frame']
    return {'loaded': True, 'source': context['source'], 'rows': len(frame),
            'cards': int(frame.cc_num.nunique()), 'cold_start': frame.empty,
            'start': None if frame.empty else str(frame.trans_date_trans_time.min()),
            'end': None if frame.empty else str(frame.trans_date_trans_time.max())}


def save_context(request, response, frame, source):
    # Replace this browser's context; never persist uploads or share another session's history.
    old = request.cookies.get('fraudshield_history')
    token = secrets.token_urlsafe(24)
    with _lock:
        if old:
            _contexts.pop(old, None)
        if len(_contexts) >= 4:
            oldest = min(_contexts, key=lambda key: _contexts[key]['time'])
            del _contexts[oldest]
        context = {'frame': frame, 'source': source, 'time': time.monotonic()}
        _contexts[token] = context
    response.set_cookie('fraudshield_history', token, httponly=True, samesite='strict', max_age=CONTEXT_TTL)
    return context_summary(context)


def analyze(df, source, model='baseline', context=None):
    if len(df) > MAX_ROWS:
        raise ValueError(f'Maximum batch size is {MAX_ROWS:,} transactions.')
    saved = bundle(model)
    validate_input(df, history=bundle_uses_history(saved))
    result = score_transactions(df, saved, context=context)
    answer = {'source': source, 'rows': result.to_dict('records'), 'model_id': model,
              'total': len(result), 'flagged': int(result.review_flag.sum()),
              'flagged_amount': float(result.loc[result.review_flag, 'amount_usd'].sum()),
              'threshold': saved['threshold']}
    if model == 'history':
        answer['no_prior_history'] = int(result.prior_transactions.eq(0).sum())
    return answer


ERRORS = (ValueError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, FileNotFoundError)


@app.get('/')
def index():
    return FileResponse(ROOT / 'static/index.html')


@app.get('/api/models')
def models():
    return [{'id': key, 'label': value, 'history': key == 'history',
             'available': (artifact_dir(key) / 'selected_model.joblib').exists()} for key, value in MODEL_NAMES.items()]


@app.get('/api/metadata')
def metadata(model: str = 'baseline'):
    try:
        saved = bundle(model)['metadata']
        artifacts = artifact_dir(model)
        test = artifacts / 'final_test.json'
        return {'model': saved['selected_model'], 'model_id': model, 'threshold': saved['threshold'],
                'features': saved['feature_count'], 'created': saved['created_utc'],
                'data': saved['data'], 'limitations': saved['limitations'],
                'validation': pd.read_csv(artifacts / 'validation_comparison.csv').to_dict('records'),
                'final_test': json.loads(test.read_text()) if model == 'baseline' and test.exists() else None,
                'required_columns': required_columns(bundle_uses_history(bundle(model)))}
    except ERRORS as exc:
        return JSONResponse({'error': str(exc)}, status_code=503)


@app.post('/api/sample')
def sample(model: str = 'baseline'):
    try:
        if model == 'history':
            raise ValueError('Use the paired history example in the Customer history tab.')
        df = pd.read_csv(artifact_dir(model) / 'demo_transactions.csv')
        return analyze(df, f'Validation demo sample ({len(df):,} transactions)', model)
    except ERRORS as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


@app.post('/api/analyze')
def upload(request: Request, file: UploadFile = File(...), model: str = 'baseline'):
    try:
        df, filename = read_upload(file)
        context = session_context(request) if model == 'history' else None
        return analyze(df, filename, model, None if context is None else context['frame'])
    except ERRORS as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


@app.get('/api/history/context')
def context_status(request: Request):
    return context_summary(session_context(request))


@app.post('/api/history/context')
def upload_context(request: Request, response: Response, file: UploadFile = File(...)):
    # Clear prior context on a failed replacement rather than silently keep using it.
    previous = request.cookies.get('fraudshield_history')
    with _lock:
        _contexts.pop(previous, None)
    try:
        frame, filename = read_upload(file, history=True)
        frame = validate_input(frame, history=True)[required_columns(True)]
        return save_context(request, response, frame, filename)
    except ERRORS as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


@app.delete('/api/history/context')
def clear_context(request: Request, response: Response):
    with _lock:
        _contexts.pop(request.cookies.get('fraudshield_history'), None)
    response.delete_cookie('fraudshield_history')
    return {'loaded': False}


@app.post('/api/history/cold-start')
def cold_start(request: Request, response: Response):
    return save_context(request, response, pd.DataFrame(columns=required_columns(True)), 'No prior history selected')


@app.post('/api/history/sample-context')
def sample_context(request: Request, response: Response):
    try:
        frame = pd.read_csv(artifact_dir('history') / 'demo_history.csv', dtype={'cc_num': 'string'})
        return save_context(request, response, validate_input(frame, history=True)[required_columns(True)],
                            'Earlier training-period example history')
    except ERRORS as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


@app.post('/api/history/sample')
def history_sample(request: Request):
    try:
        frame = pd.read_csv(artifact_dir('history') / 'demo_transactions.csv', dtype={'cc_num': 'string'})
        context = session_context(request)
        return analyze(frame, f'Validation example ({len(frame):,} transactions)', 'history',
                       None if context is None else context['frame'])
    except ERRORS as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


@app.get('/api/validation-pr.png')
def validation_chart(model: str = 'baseline'):
    try:
        path = artifact_dir(model) / 'validation_pr.png'
        if not path.exists():
            raise FileNotFoundError('No validation chart is available.')
        return FileResponse(path, media_type='image/png')
    except ERRORS as exc:
        return JSONResponse({'error': str(exc)}, status_code=404)
