'use strict';
const $ = id => document.getElementById(id);
let state = null, selectedLayer = null, pending = false, inspectSequence = 0;
const openEvents = new Set();
const number = value => Number.isFinite(Number(value)) ? Number(value).toLocaleString() : '—';
const phases = {idle:'AWAITING PATIENT',loading:'LOADING PATIENT',ready:'PATIENT READY',operating:'SURGERY IN PROGRESS',comparing:'COMPARING OUTPUTS',reverting:'REVERTING',error:'ATTENTION REQUIRED'};
function setError(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
async function request(path, body) {
  const response = await fetch(path, body === undefined ? {cache:'no-store'} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const text = await response.text(); let result;
  try { result = text ? JSON.parse(text) : {}; } catch { throw new Error(text.slice(0, 300) || `Request failed (${response.status})`); }
  if (!response.ok || result.ok === false || result.error) throw new Error(typeof result.error === 'string' ? result.error : `Request failed (${response.status})`);
  return result;
}
async function refresh() {
  try { const next = await request('/api/state'); state = next; $('connection').textContent = 'CONNECTED'; $('connection-light').classList.add('connected'); render(); }
  catch (error) { $('connection').textContent = 'DISCONNECTED'; $('connection-light').classList.remove('connected'); setError(`Unable to read backend state: ${error.message}`); }
}
function render() {
  const p = state.patient, loaded = !!p?.loaded, busy = !!state.busy || pending;
  $('phase').textContent = phases[state.phase] || String(state.phase || 'UNKNOWN').toUpperCase();
  setError(state.error || '');
  $('model').textContent = p?.model || 'Qwen3-0.6B'; $('loaded-pill').textContent = loaded ? 'LOADED' : 'NOT LOADED'; $('loaded-pill').classList.toggle('loaded', loaded);
  $('parameters').textContent = p ? number(p.parameters) : '—'; $('device').textContent = p?.device || '—'; $('changed').textContent = p ? number(p.changed_parameters) : '—'; $('revision').textContent = p?.revision != null ? String(p.revision) : '—';
  const percent = Number(p?.changed_percent) || 0; $('percent').textContent = `${percent.toFixed(4)}%`; $('progress').style.width = `${Math.max(0,Math.min(100,percent))}%`;
  const limits = state.limits || {}; $('resources').textContent = `${state.tool_calls || 0} / ${limits.max_tool_calls ?? 12} CALLS · ${state.experiments || 0} / ${limits.max_experiments ?? 3} EDITS`;
  $('load').disabled = busy || loaded; $('surgery').disabled = busy || !loaded || !$('objective').value.trim(); $('compare').disabled = busy || !loaded; $('accept').disabled = busy || !loaded || !(Number(p?.changed_parameters) > 0); $('revert').disabled = !loaded && !state.busy;
  $('objective').disabled = busy; $('max-tools').disabled = busy; $('max-experiments').disabled = busy;
  $('active-tensor').hidden = !state.active_tensor;
  $('active-tensor').textContent = state.active_tensor ? `ACTIVE TENSOR / ${state.active_tensor}` : '';
  renderLayers(p); renderActivity(state.activity || []); renderComparison(state.comparison);
}
function layerMatches(tensor, layer) { return new RegExp(`(?:^|\\.)layers\\.${layer}(?:\\.|$)`).test(tensor || ''); }
function renderLayers(p) {
  const count = Number(p?.layers) > 0 ? Number(p.layers) : 28; $('layer-count').textContent = `${count} LAYERS`;
  const container = $('layers'); container.replaceChildren();
  for (let i=0; i<count; i++) { const button = document.createElement('button'); button.className = 'layer'; button.textContent = String(i).padStart(2,'0'); button.title = `Inspect layer ${i}`; button.setAttribute('aria-label', `Inspect layer ${i}`); button.setAttribute('aria-pressed', String(selectedLayer === i)); button.disabled = !p?.loaded || !!state.busy || pending; button.classList.toggle('selected', selectedLayer === i); button.classList.toggle('active', layerMatches(state.active_tensor, i)); button.classList.toggle('modified', (p?.modified_tensors || []).some(t => layerMatches(t,i))); button.addEventListener('click', () => inspect(i)); container.append(button); }
}
function renderActivity(events) {
  $('activity-count').textContent = `${events.length} EVENTS`; const container = $('activity'); container.replaceChildren();
  if (!events.length) { const empty = document.createElement('div'); empty.className='empty-log'; const icon=document.createElement('span'); icon.textContent='⌁'; const message=document.createElement('p'); message.textContent='Waiting for activity.'; const small=document.createElement('small'); small.textContent='Procedure events will appear here.'; message.append(document.createElement('br'),small); empty.append(icon,message); container.append(empty); return; }
  for (const event of events.slice(-30).reverse()) { const row=document.createElement('div'); row.className='event'; if(event.kind==='error') row.classList.add('error'); const kind=document.createElement('span'); kind.className='event-kind'; kind.textContent=event.kind || 'event'; const content=document.createElement('div');
    const message=document.createElement('span'); message.className='event-message'; message.textContent=event.message || ''; content.append(message);
    if (event.data != null && (typeof event.data !== 'object' || Object.keys(event.data).length)) {
      const details=document.createElement('details'); details.className='event-details';
      details.open = openEvents.has(event.id);
      details.addEventListener('toggle', () => {
        if (details.open) openEvents.add(event.id);
        else openEvents.delete(event.id);
      });
      const summary=document.createElement('summary'); summary.textContent='VIEW DETAILS';
      const pre=document.createElement('pre');
      let serialized;
      try { serialized=JSON.stringify(event.data, null, 2); } catch { serialized=String(event.data); }
      pre.textContent=serialized.length > 12000 ? `${serialized.slice(0,12000)}\n… [truncated]` : serialized;
      details.append(summary,pre); content.append(details);
    }
    row.append(kind,content); container.append(row); }
}
function renderComparison(comparison) {
  $('comparison-section').hidden = !comparison; const root=$('comparison-results'); root.replaceChildren(); if(!comparison) return;
  const before=Array.isArray(comparison.baseline)?comparison.baseline:[], after=Array.isArray(comparison.working)?comparison.working:[];
  for(let i=0;i<Math.max(before.length,after.length);i++) { const a=before[i] || {}, b=after[i] || {}; const prompt=document.createElement('div'); prompt.className='prompt'; prompt.textContent=`PROMPT ${String(i+1).padStart(2,'0')}  /  ${a.prompt || b.prompt || ''}`; root.append(prompt); const grid=document.createElement('div'); grid.className='comparison-grid'; for(const [label,item] of [['BEFORE / BASELINE',a],['AFTER / WORKING',b]]) { const card=document.createElement('div'); card.className='comparison-item'; const title=document.createElement('h4'); title.textContent=label; const output=document.createElement('p'); output.textContent=item.output ?? 'No output available.'; const meta=document.createElement('div'); meta.className='output-meta'; const details=[]; if(item.tokens != null) details.push(`${item.tokens} tokens`); if(item.elapsed_ms != null) details.push(`${number(item.elapsed_ms)} ms`); if(item.hit_token_limit) details.push('token limit reached'); if(item.repetition_rate != null) details.push(`repetition ${item.repetition_rate}`); meta.textContent=details.join(' · '); card.append(title,output,meta); grid.append(card); } root.append(grid); }
}
async function action(path, body={}) { pending=true; if(state) render(); setError(''); try { await request(path,body); await refresh(); } catch(error) { setError(error.message); } finally { pending=false; if(state) { const currentError=$('error').textContent; render(); if(currentError && !state.error) setError(currentError); } } }
async function inspect(layer) { selectedLayer=layer; const sequence=++inspectSequence; renderLayers(state.patient); $('inspector-title').textContent=`Layer ${String(layer).padStart(2,'0')}`; $('inspect-message').textContent='Inspecting tensor catalog…'; $('inspect-results').replaceChildren(); try { const result=await request('/api/inspect',{layer}); if(sequence!==inspectSequence)return; const candidates = Array.isArray(result) ? result : (result.tensors || result.catalog || result.parameters || result.items || result.result?.tensors || []); let rows=Array.isArray(candidates) ? candidates : Object.entries(candidates).map(([name,value]) => ({name,...(typeof value==='object' && value !== null ? value : {value})})); if(!rows.length && result && typeof result==='object') { const inner=result.result || result; rows=Object.entries(inner).filter(([key])=>!['ok','layer'].includes(key)).map(([name,value])=>({name,value})); } $('inspect-message').textContent=rows.length ? `Showing ${Math.min(rows.length,30)} of ${rows.length} catalog entries.` : 'No tensor entries returned for this layer.'; if(!rows.length)return; const table=document.createElement('table'); table.className='tensor-table'; const head=document.createElement('tr'); for(const label of ['TENSOR','DETAILS']) {const th=document.createElement('th'); th.textContent=label; head.append(th);} table.append(head); for(const item of rows.slice(0,30)) {const row=document.createElement('tr'); const name=document.createElement('td'); const details=document.createElement('td'); if(typeof item==='string'){name.textContent=item;details.textContent='—';} else {name.textContent=String(item.name || item.tensor || item.key || 'Tensor'); const rest=Object.fromEntries(Object.entries(item).filter(([key])=>!['name','tensor','key'].includes(key))); details.textContent=Object.keys(rest).length ? JSON.stringify(rest) : '—';} row.append(name,details);table.append(row);} $('inspect-results').append(table); } catch(error) { if(sequence===inspectSequence) $('inspect-message').textContent=`Inspection failed: ${error.message}`; } }
$('load').addEventListener('click',()=>action('/api/load')); $('surgery').addEventListener('click',()=>action('/api/surgery',{objective:$('objective').value.trim(),max_tool_calls:Number($('max-tools').value),max_experiments:Number($('max-experiments').value)})); $('compare').addEventListener('click',()=>action('/api/compare')); $('accept').addEventListener('click',()=>action('/api/accept')); $('revert').addEventListener('click',()=>action('/api/revert')); $('objective').addEventListener('input',()=>{if(state)render()});
refresh(); setInterval(refresh,1000);
