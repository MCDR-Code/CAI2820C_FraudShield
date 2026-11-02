"""Local demo with separate models and browser-session history held only in RAM."""

import base64
import csv
import io
import json
import os
import secrets
import time
from threading import Lock

import pandas as pd
from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core import ROOT, bundle_uses_history, load_bundle, required_columns, score_transactions, validate_input
from history import HISTORY_VERSION

app = FastAPI(title='FraudShield', docs_url=None, redoc_url=None)

# Remote presentation mode: set FRAUDSHIELD_PASSWORD to require a password on every request
# (HTTP Basic auth, any user name). Unset = local-only use, no password, as before.
DEMO_PASSWORD = os.environ.get('FRAUDSHIELD_PASSWORD') or None


@app.middleware('http')
async def require_password(request: Request, call_next):
    if DEMO_PASSWORD is None:
        return await call_next(request)
    header = request.headers.get('authorization', '')
    if header.startswith('Basic '):
        try:
            supplied = base64.b64decode(header[6:]).decode('utf-8').split(':', 1)[1]
        except Exception:
            supplied = ''
        if secrets.compare_digest(supplied.encode(), DEMO_PASSWORD.encode()):
            return await call_next(request)
    return Response('Password required', status_code=401, headers={'WWW-Authenticate': 'Basic realm="FraudShield"'})

app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
MODEL_NAMES = {'baseline': 'Random Forest (comparison)', 'boosting': 'Gradient Boosting, transaction-only (comparison)',
               'history': 'Gradient Boosting + customer history (final model)'}
CONTEXT_TTL = 3600
_bundle = None
_bundles = {}
_contexts = {}
_lock = Lock()


def artifact_dir(model):
    if model not in MODEL_NAMES:
        raise ValueError('Unknown model.')
    # Final models trained on the corrected (fixed-sampler) data; see Generator_Fix.md.
    expanded = ROOT / 'artifacts' / 'expanded_fixed_run1'
    selected = expanded / {'history': 'history_v3', 'boosting': 'transaction_models', 'baseline': 'rf_baseline'}[model]
    if (selected / 'selected_model.joblib').is_file():
        return selected
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
                raise ValueError('History artifact is incompatible. Install the verified history v3 artifact.')
            _bundles[model] = saved
        return _bundles[model]


def read_upload(file, history=False):
    try:
        filename = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
        if not filename.lower().endswith('.csv'):
            raise ValueError('Choose a .csv file.')
        # Read directly from FastAPI's spooled upload. Avoid retaining a full
        # bytes copy and a second decoded string alongside a large dataframe.
        with io.TextIOWrapper(file.file, encoding='utf-8-sig', newline='') as text:
            header = next(csv.reader(text), [])
            # Original Sparkov exports its unused row index with a blank heading.
            if header and header[0] == '':
                header[0] = 'Unnamed: 0'
            if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
                raise ValueError('CSV column names must be nonempty and unique.')
            text.seek(0)
            frame = pd.read_csv(text, dtype={'cc_num': 'string'})
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
            'file_details': [{'id': item['id'], 'name': item['name'], 'rows': len(item['frame'])} for item in context.get('parts', [])],
            'files': context.get('files', []), 'duplicates_removed': context.get('duplicates_removed', 0),
            'cards': int(frame.cc_num.nunique()), 'cold_start': frame.empty,
            'start': None if frame.empty else str(frame.trans_date_trans_time.min()),
            'end': None if frame.empty else str(frame.trans_date_trans_time.max())}


def save_context(request, response, frame, source, files=None, duplicates_removed=0, parts=None):
    # Replace this browser's context; never persist uploads or share another session's history.
    old = request.cookies.get('fraudshield_history')
    token = secrets.token_urlsafe(24)
    with _lock:
        if old:
            _contexts.pop(old, None)
        if len(_contexts) >= 4:
            oldest = min(_contexts, key=lambda key: _contexts[key]['time'])
            del _contexts[oldest]
        context = {'frame': frame, 'source': source, 'time': time.monotonic(),
                   'files': files or [], 'duplicates_removed': duplicates_removed,
                   'parts': parts or ([] if frame.empty else [{'id': secrets.token_urlsafe(12), 'name': source, 'frame': frame}])}
        _contexts[token] = context
    response.set_cookie('fraudshield_history', token, httponly=True, samesite='strict', max_age=CONTEXT_TTL)
    return context_summary(context)


def upload_evaluation(labels, result):
    """Evaluate AFTER scoring; align labels by original row position, not score order."""
    truth = labels.iloc[result.row_number.to_numpy() - 1].to_numpy(dtype=int)
    flags = result.review_flag.to_numpy(dtype=bool)
    tp = int(((truth == 1) & flags).sum())
    fp = int(((truth == 0) & flags).sum())
    fn = int(((truth == 1) & ~flags).sum())
    tn = int(((truth == 0) & ~flags).sum())
    result['actual_is_fraud'] = truth
    result['evaluation_result'] = pd.Series(truth * 2 + flags.astype(int)).map({
        0: 'Correctly cleared', 1: 'False alert', 2: 'Missed fraud', 3: 'Correct alert'}).to_numpy()
    return {'rows': len(result), 'actual_fraud': tp + fn, 'actual_legitimate': tn + fp,
            'true_positives': tp, 'false_positives': fp, 'false_negatives': fn, 'true_negatives': tn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'accuracy': (tp + tn) / len(result)}


def analyze(df, source, model='baseline', context=None):
    labels = None
    if 'is_fraud' in df:
        labels = pd.to_numeric(df['is_fraud'], errors='coerce')
        if not labels.isin([0, 1]).all():
            raise ValueError('is_fraud must contain a 0 or 1 for every transaction to evaluate this file.')
    saved = bundle(model)
    validate_input(df, history=bundle_uses_history(saved))
    result = score_transactions(df, saved, context=context)
    evaluation = upload_evaluation(labels, result) if labels is not None else None
    answer = {'source': source, 'rows': result.to_dict('records'), 'model_id': model,
              'evaluation': evaluation,
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
                'final_test': json.loads(test.read_text()) if test.exists() else None,
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


def combine_history(files, previous=None):
    """Combine a selected set; transaction IDs prevent double-counted history."""
    entries = list(previous.get('parts', [])) if previous else []
    if previous and not entries and not previous['frame'].empty:
        entries = [{'id': secrets.token_urlsafe(12), 'name': previous['source'], 'frame': previous['frame']}]
    columns = required_columns(True)
    for upload_file in files:
        frame, name = read_upload(upload_file, history=True)
        validated = validate_input(frame, history=True)
        keep = columns + (['trans_num'] if 'trans_num' in validated else [])
        entries.append({'id': secrets.token_urlsafe(12), 'name': name, 'frame': validated[keep]})
    return merge_history_parts(entries)


def merge_history_parts(entries):
    columns = required_columns(True)
    parts = [item['frame'] for item in entries]
    names = [item['name'] for item in entries]
    combined = pd.concat(parts, ignore_index=True)
    if 'trans_num' in combined and combined.trans_num.notna().all() and combined.trans_num.astype(str).str.strip().ne('').all():
        # Identical IDs with changed model inputs are conflicts, not duplicates.
        unique = combined.drop_duplicates(subset=['trans_num'] + columns)
        if unique.trans_num.duplicated().any():
            raise ValueError('History files contain conflicting transactions with the same trans_num.')
        combined = unique
    elif len(parts) > 1 and combined.duplicated(subset=columns).any():
        raise ValueError('History files contain identical rows without complete transaction IDs. Include trans_num to distinguish duplicates from separate purchases.')
    removed = sum(len(part) for part in parts) - len(combined)
    combined = combined.sort_values('trans_date_trans_time', kind='stable').reset_index(drop=True)
    keep = columns + (['trans_num'] if 'trans_num' in combined else [])
    return combined[keep], ', '.join(names), names, removed, entries


@app.post('/api/history/context')
def upload_context(request: Request, response: Response, file: list[UploadFile] = File(...), append: bool = False):
    # Clear prior context on a failed replacement rather than silently keep using it.
    previous_context = session_context(request) if append else None
    previous = request.cookies.get('fraudshield_history')
    if not append:
        with _lock:
            _contexts.pop(previous, None)
    try:
        frame, filename, names, removed, parts = combine_history(file, previous_context)
        return save_context(request, response, frame, filename, files=list(dict.fromkeys(names)), duplicates_removed=removed, parts=parts)
    except ERRORS as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)
    finally:
        for upload_file in file:
            upload_file.file.close()


# Only these known project files can be loaded from the server. No client paths.
SERVER_HISTORY = {
    str(year): ROOT / 'data' / ('synthetic_2021_fixed' if year == 2021 else 'synthetic_2022_2026_fixed') / f'transactions_{year}.csv'
    for year in range(2021, 2026)
}


@app.post('/api/history/server-context')
def server_context(request: Request, response: Response):
    try:
        entries = []
        for year, path in SERVER_HISTORY.items():
            # Read predictor/history fields and transaction IDs only, never labels.
            columns = required_columns(True) + ['trans_num']
            frame = pd.read_csv(path, dtype={'cc_num': 'string'}, usecols=columns)
            frame = validate_input(frame, history=True)[columns]
            entries.append({'id': secrets.token_urlsafe(12), 'name': path.name, 'frame': frame})
        frame, source, names, removed, parts = merge_history_parts(entries)
        return save_context(request, response, frame, source, files=names, duplicates_removed=removed, parts=parts)
    except ERRORS as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


# Reserved 2026 test file, scored on the server so presenters need not upload 250 MB.
SERVER_REVIEW = ROOT / 'data' / 'synthetic_2022_2026_fixed' / 'transactions_2026.csv'


@app.post('/api/history/server-review')
def server_review(request: Request):
    try:
        context = session_context(request)
        if context is None:
            raise ValueError('Load the server history 2021-2025 first.')
        frame = pd.read_csv(SERVER_REVIEW, dtype={'cc_num': 'string'})
        return analyze(frame, f'{SERVER_REVIEW.name} (server copy, {len(frame):,} transactions)', 'history', context['frame'])
    except ERRORS as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


@app.delete('/api/history/files/{file_id}')
def unload_history_file(file_id: str, request: Request, response: Response):
    previous = session_context(request)
    if previous is None:
        return JSONResponse({'error': 'History session expired. Load history again.'}, status_code=400)
    entries = previous.get('parts', [])
    remaining = [item for item in entries if item['id'] != file_id]
    if len(remaining) == len(entries):
        return JSONResponse({'error': 'History file not found.'}, status_code=404)
    if not remaining:
        return clear_context(request, response)
    frame, source, names, removed, parts = merge_history_parts(remaining)
    return save_context(request, response, frame, source, files=names, duplicates_removed=removed, parts=parts)


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
