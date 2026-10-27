const $ = id => document.getElementById(id);
const count = n => Number(n).toLocaleString('en-US');
const pct = n => `${(100 * n).toFixed(2)}%`;
let rows = [], busy = false, view = 'review', metadataRequest = 0;
const emptyBatch = () => ({data:null,error:'',status:'No batch loaded',flaggedOnly:false});
const batches = {baseline:emptyBatch(),boosting:emptyBatch(),history:emptyBatch()};
const batchKey = () => view === 'history' ? 'history' : $('review-model').value;
const activeModel = () => view === 'history' ? 'history' : view === 'model' ? $('evaluation-model').value : $('review-model').value;
const cell = (row, value, cls = '') => {const td = document.createElement('td'); td.textContent = value; td.className = cls; row.appendChild(td);};
function clearBatch() {
  batches[batchKey()] = emptyBatch(); displayBatch();
}
function showEvaluation(data) {
  $('upload-evaluation').hidden=!data;
  const evaluated=data?.evaluation;
  $('evaluation-values').hidden=!evaluated;
  $('actual-heading').hidden=$('outcome-heading').hidden=!evaluated;
  if(!data)return;
  $('evaluation-note').textContent=evaluated
    ? `Compared with is_fraud in this file: ${count(evaluated.actual_fraud)} fraud and ${count(evaluated.actual_legitimate)} legitimate transactions. These are this upload's results, separate from saved validation results. Accuracy alone can be misleading when fraud is rare. N/A means the measure has no applicable cases.`
    : 'This file has no is_fraud labels. Predictions are available, but their correctness cannot be measured from this upload.';
  if(!evaluated)return;
  for(const key of ['precision','recall','f1','accuracy'])$('upload-'+key).textContent=evaluated[key]===null?'N/A':pct(evaluated[key]);
  for(const [short,key] of [['tp','true_positives'],['fp','false_positives'],['fn','false_negatives'],['tn','true_negatives']])$('upload-'+short).textContent=count(evaluated[key]);
}
function displayBatch() {
  const state=batches[batchKey()],data=state.data;
  rows=data?.rows || [];
  $('flagged-only').checked=state.flaggedOnly;
  $('status').textContent=state.status;
  $('error').textContent=state.error;$('error').hidden=!state.error;
  $('total').textContent=data?count(data.total):'—';
  $('flagged').textContent=data?count(data.flagged):'—';
  $('amount').textContent=data?data.flagged_amount.toLocaleString('en-US',{style:'currency',currency:'USD'}):'—';
  $('threshold').textContent=data?data.threshold.toFixed(4):'—';
  $('download').disabled=$('history-download').disabled=!rows.length;
  showEvaluation(data);render();
}
function render() {
  const filtered = $('flagged-only').checked ? rows.filter(r => r.review_flag) : rows;
  $('prior-heading').hidden = view !== 'history';
  $('queue').replaceChildren();
  for (const item of filtered.slice(0,100)) {
    const tr = document.createElement('tr');
    cell(tr,item.row_number); cell(tr,item.transaction_time); cell(tr,item.category);
    cell(tr,Number(item.amount_usd).toLocaleString('en-US',{style:'currency',currency:'USD'}),'number');
    cell(tr,item.model_score.toFixed(4),'number'); cell(tr,item.review_flag ? 'Review' : 'Not flagged',item.review_flag ? 'flag' : 'ok');
    if(view === 'history') cell(tr,item.prior_transactions ? count(item.prior_transactions) : 'New / no prior history');
    if(batches[batchKey()].data?.evaluation){cell(tr,item.actual_is_fraud===1?'Fraud':'Legitimate');cell(tr,item.evaluation_result);}
    $('queue').appendChild(tr);
  }
  if(!filtered.length){const tr=document.createElement('tr');cell(tr,rows.length ? 'No matching transactions' : 'No transactions','empty');tr.firstChild.colSpan=(view==='history'?7:6)+(batches[batchKey()].data?.evaluation?2:0);$('queue').appendChild(tr);}
  $('shown').textContent=filtered.length ? `${count(Math.min(100,filtered.length))} of ${count(filtered.length)}` : '';
}
function showContext(data) {
  $('history-status').textContent = !data.loaded ? 'No history loaded' : data.cold_start ? 'No prior history selected · cold start' : `${data.source} · ${count(data.rows)} transactions · ${count(data.cards)} cards · through ${data.end} UTC`+(data.duplicates_removed?` · ${count(data.duplicates_removed)} duplicate transactions removed`:'');
}
async function operation(url,form,context=false,method='POST') {
  if(busy)return;
  const key=batchKey();
  if(!context)clearBatch();
  busy=true;$('status').textContent=context?'Updating history…':'Scoring transactions…';
  const controls=[...document.querySelectorAll('button,input[type=file],select')];
  const prior=controls.map(c=>c.disabled);controls.forEach(c=>c.disabled=true);
  try {
    const response=await fetch(url,{method,...(form?{body:form}:{})});
    const data=await response.json();if(!response.ok)throw new Error(data.error || 'Could not process this file.');
    if(context){showContext(data);batches.history=emptyBatch();displayBatch();}
    else {
      batches[key].data=data;batches[key].error='';
      batches[key].status=data.source+(data.no_prior_history===undefined?'':` · ${count(data.no_prior_history)} transactions without prior card history`);
      displayBatch();
    }
  } catch(error) {
    batches[key].error=error.message;
    if(!context)batches[key].status='Batch not analyzed';
    displayBatch();
    if(context)showContext(await fetch('/api/history/context').then(r=>r.json()).catch(()=>({loaded:false})));
  } finally {
    busy=false;controls.forEach((c,i)=>c.disabled=prior[i]);
    $('download').disabled=$('history-download').disabled=!rows.length;
    document.querySelectorAll('input[type=file]').forEach(c=>c.value='');
  }
}
function upload(id,url,context=false){$(id).onchange=()=>{if($(id).files.length){const form=new FormData();for(const file of $(id).files)form.append('file',file);operation(typeof url==='function'?url():url,form,context);}};}
upload('upload',()=>`/api/analyze?model=${$('review-model').value}`);
upload('history-upload','/api/history/context?append=true',true);
upload('history-review-upload','/api/analyze?model=history');
$('sample').onclick=()=>operation(`/api/sample?model=${$('review-model').value}`);
$('history-context-sample').onclick=()=>operation('/api/history/sample-context',null,true);
$('cold-start').onclick=()=>operation('/api/history/cold-start',null,true);
$('history-clear').onclick=()=>operation('/api/history/context',null,true,'DELETE');
$('history-sample').onclick=()=>operation('/api/history/sample');
$('clear').onclick=$('history-batch-clear').onclick=clearBatch;$('flagged-only').onchange=()=>{batches[batchKey()].flaggedOnly=$('flagged-only').checked;render();};
function download(){
  const columns=['row_number','transaction_time','category','amount_usd','model_score','review_flag',...(view==='history'?['prior_transactions','customer_status']:[]),...(batches[batchKey()].data?.evaluation?['actual_is_fraud','evaluation_result']:[])];
  const escape=value=>{let s=String(value);if(/^[=+@\-\t\r]/.test(s.trimStart()))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
  const csv=[columns.join(','),...rows.map(r=>columns.map(c=>escape(r[c])).join(','))].join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8;'}));
  const a=document.createElement('a');a.href=url;a.download=`fraudshield_${activeModel()}_scored.csv`;a.click();URL.revokeObjectURL(url);
}
$('download').onclick=$('history-download').onclick=download;
for(const name of ['review','history','model'])$(name+'-tab').onclick=()=>{
  if(busy)return;view=name;displayBatch();
  for(const other of ['review','history','model']){$(other+'-view').hidden=other!==name;$(other+'-tab').classList.toggle('active',other===name);$(other+'-tab').setAttribute('aria-selected',String(other===name));}
  $('batch-view').hidden=name==='model';
  if(name==='history')fetch('/api/history/context').then(r=>r.json()).then(showContext);
  loadMetadata();
};
const detail=(id,label,value)=>{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;$(id).append(dt,dd);};
async function loadMetadata(){
  const request=++metadataRequest,model=activeModel();
  $('model-error').hidden=true;
  for(const id of ['comparison','test-results','run-record','limitations'])$(id).replaceChildren();
  document.querySelector('.chart-layout img').hidden=true;
  try{
    const response=await fetch(`/api/metadata?model=${model}`),data=await response.json();if(request!==metadataRequest)return;
    if(!response.ok)throw new Error(data.error);
    $('model-label').textContent=data.model+(model==='history'?' + customer history':'')+(model!=='baseline'?' · validation only':'');
    $('training-size').textContent=`${count(data.data.training_rows)} training / ${count(data.data.validation_rows)} validation`;
    for(const id of ['comparison','test-results','run-record','limitations'])$(id).replaceChildren();
    for(const r of data.validation){const tr=document.createElement('tr');cell(tr,r.model);for(const k of ['precision','recall','f1','average_precision'])cell(tr,pct(r[k]),'number');cell(tr,count(r.fp),'number');$('comparison').appendChild(tr);}
    if(data.final_test){for(const [k,label]of[['precision','Precision'],['recall','Recall'],['f1','F1'],['average_precision','Average precision']])detail('test-results',label,pct(data.final_test[k]));detail('test-results','False alerts',count(data.final_test.fp));detail('test-results','Missed fraud',count(data.final_test.fn));detail('test-results','Preliminary targets',data.final_test.targets_met?'Met':'Not met');}
    else detail('test-results','Status','Not evaluated for this model · validation only');
    detail('run-record','Features',data.features);detail('run-record','Threshold',data.threshold.toFixed(6));detail('run-record','Created',new Date(data.created).toLocaleString());
    for(const text of data.limitations){const li=document.createElement('li');li.textContent=text;$('limitations').appendChild(li);}
    document.querySelector('.chart-layout img').src=`/api/validation-pr.png?model=${model}`;
    document.querySelector('.chart-layout img').hidden=false;
  }catch(error){if(request!==metadataRequest)return;$('model-label').textContent='Model unavailable';$('model-error').textContent=error.message;$('model-error').hidden=false;}
}
$('review-model').onchange=()=>{displayBatch();loadMetadata();};$('evaluation-model').onchange=loadMetadata;
loadMetadata();
