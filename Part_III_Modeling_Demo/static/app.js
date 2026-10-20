const $ = id => document.getElementById(id);
const count = n => Number(n).toLocaleString('en-US');
const pct = n => `${(100 * n).toFixed(2)}%`;
let rows = [], source = '', busy = false;
const cell = (row, value, cls = '') => {const td = document.createElement('td'); td.textContent = value; td.className = cls; row.appendChild(td);};
function clearBatch() {
  rows = []; source = ''; $('download').disabled = true;
  for (const id of ['total','flagged','amount','threshold']) $(id).textContent = '—';
  $('status').textContent = 'No batch loaded'; $('error').hidden = true; render();
}
function render() {
  const filtered = $('flagged-only').checked ? rows.filter(r => r.review_flag) : rows;
  $('queue').replaceChildren();
  for (const item of filtered.slice(0, 100)) {
    const tr = document.createElement('tr');
    cell(tr, item.row_number); cell(tr, item.transaction_time); cell(tr, item.category);
    cell(tr, Number(item.amount_usd).toLocaleString('en-US',{style:'currency',currency:'USD'}), 'number');
    cell(tr, item.model_score.toFixed(4), 'number'); cell(tr, item.review_flag ? 'Review' : 'Not flagged', item.review_flag ? 'flag' : 'ok');
    $('queue').appendChild(tr);
  }
  if (!filtered.length) {const tr=document.createElement('tr'); cell(tr, rows.length ? 'No matching transactions' : 'No transactions','empty'); tr.firstChild.colSpan=6; $('queue').appendChild(tr);}
  $('shown').textContent = filtered.length ? `${count(Math.min(100,filtered.length))} of ${count(filtered.length)}` : '';
}
async function analyze(url, form) {
  if (busy) return;
  clearBatch(); busy = true; $('status').textContent = 'Scoring transactions…';
  $('sample').disabled = true; $('clear').disabled = true; $('upload').disabled = true;
  try {
    const response = await fetch(url, {method:'POST', ...(form ? {body:form} : {})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not analyze this batch.');
    rows=data.rows; source=data.source; $('status').textContent=source;
    $('total').textContent=count(data.total); $('flagged').textContent=count(data.flagged);
    $('amount').textContent=data.flagged_amount.toLocaleString('en-US',{style:'currency',currency:'USD'});
    $('threshold').textContent=data.threshold.toFixed(4); $('download').disabled=false; render();
  } catch(error) {$('status').textContent='Batch not analyzed'; $('error').textContent=error.message; $('error').hidden=false;}
  finally {busy=false; $('sample').disabled=false; $('clear').disabled=false; $('upload').disabled=false; $('upload').value='';}
}
$('upload').addEventListener('change', () => {if($('upload').files.length){const form=new FormData();form.append('file',$('upload').files[0]);analyze('/api/analyze',form);}});
$('sample').onclick=()=>analyze('/api/sample'); $('clear').onclick=clearBatch; $('flagged-only').onchange=render;
$('download').onclick=()=>{
  const columns=['row_number','transaction_time','category','amount_usd','model_score','review_flag'];
  const escape = value => {let s=String(value); if (/^[=+@\-\t\r]/.test(s.trimStart())) s="'"+s; return '"'+s.replaceAll('"','""')+'"';};
  const csv=[columns.join(','),...rows.map(r=>columns.map(c=>escape(r[c])).join(','))].join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8;'}));
  const a=document.createElement('a');a.href=url;a.download='fraudshield_scored.csv';a.click();URL.revokeObjectURL(url);
};
for (const name of ['review','model']) $(name+'-tab').onclick=()=>{
  for (const other of ['review','model']) {$(other+'-view').hidden=other!==name;$(other+'-tab').classList.toggle('active',other===name);$(other+'-tab').setAttribute('aria-selected',String(other===name));}
};
const detail=(id,label,value)=>{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;$(id).append(dt,dd);};
async function loadMetadata(){
  try{
    const response=await fetch('/api/metadata'); const data=await response.json();if(!response.ok)throw new Error(data.error);
    $('model-label').textContent=data.model; $('training-size').textContent=`${count(data.data.training_rows)} training / ${count(data.data.validation_rows)} validation`;
    for(const r of data.validation){const tr=document.createElement('tr');cell(tr,r.model);for(const k of ['precision','recall','f1','average_precision'])cell(tr,pct(r[k]),'number');cell(tr,count(r.fp),'number');$('comparison').appendChild(tr);}
    if(data.final_test){for(const [k,label] of [['precision','Precision'],['recall','Recall'],['f1','F1'],['average_precision','Average precision']])detail('test-results',label,pct(data.final_test[k]));detail('test-results','False alerts',count(data.final_test.fp));detail('test-results','Missed fraud',count(data.final_test.fn));detail('test-results','Preliminary targets',data.final_test.targets_met ? 'Met' : 'Not met');}
    else detail('test-results','Status','Not evaluated');
    detail('run-record','Features',data.features); detail('run-record','Test rows',count(data.data.test_rows));detail('run-record','Threshold',data.threshold.toFixed(6));detail('run-record','Created',new Date(data.created).toLocaleString());
    for(const text of data.limitations){const li=document.createElement('li');li.textContent=text;$('limitations').appendChild(li);}
  }catch(error){$('error').textContent=error.message;$('error').hidden=false;}
}
loadMetadata();
