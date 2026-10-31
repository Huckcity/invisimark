(() => {
  'use strict';
  const $ = (sel, root = document) => root.querySelector(sel);

  function h(tag, props, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (k === 'class') node.className = v; else if (k === 'text') node.textContent = v; else node.setAttribute(k, v);
    }
    for (const kid of kids) if (kid != null) node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return node;
  }

  // ---- tabs ---------------------------------------------------------------------------------
  const tabs = { mark: $('#tab-mark'), trace: $('#tab-trace') };
  const panels = { mark: $('#panel-mark'), trace: $('#panel-trace') };
  function showTab(name) {
    for (const k of Object.keys(tabs)) {
      const on = k === name;
      tabs[k].setAttribute('aria-selected', on ? 'true' : 'false');
      tabs[k].tabIndex = on ? 0 : -1;
      panels[k].hidden = !on;
    }
  }
  for (const k of Object.keys(tabs)) {
    tabs[k].addEventListener('click', () => showTab(k));
    tabs[k].addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { const other = k === 'mark' ? 'trace' : 'mark'; showTab(other); tabs[other].focus(); }
    });
  }

  // ---- talking to the server ----------------------------------------------------------------
  function send(method, url, body, onUpload) {
    return new Promise((resolve) => {
      const xhr = new XMLHttpRequest();
      xhr.open(method, url);
      if (onUpload) xhr.upload.onprogress = (e) => { if (e.lengthComputable) onUpload(e.loaded / e.total); };
      xhr.onload = () => {
        let json = {};
        try { json = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
        resolve({ ok: xhr.status >= 200 && xhr.status < 300, status: xhr.status, json });
      };
      xhr.onerror = () => resolve({ ok: false, status: 0, json: { detail: 'The connection failed. Check your network and try again.' } });
      xhr.send(body || null);
    });
  }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // ---- one runner per panel -----------------------------------------------------------------
  function runner(kind, render) {
    const form = $(`#${kind}-form`), status = $(`#${kind}-status`), error = $(`#${kind}-error`), result = $(`#${kind}-result`);
    const bar = $('.bar > i', status), stage = $('.stage', status), button = $('button.go', form);
    let token = 0;

    function show(text, pct) {
      status.hidden = false;
      stage.textContent = text;
      if (pct != null) bar.style.width = `${Math.round(pct * 100)}%`;
    }
    function fail(text) {
      status.hidden = true; result.hidden = true;
      error.textContent = text.charAt(0).toUpperCase() + text.slice(1); error.hidden = false;
      button.disabled = false;
    }
    async function follow(id) {
      const mine = ++token;
      history.replaceState(null, '', `#job=${id}&kind=${kind}`);
      for (;;) {
        const r = await send('GET', `/api/jobs/${id}`);
        if (mine !== token) return;
        if (!r.ok) { history.replaceState(null, '', location.pathname); return fail(r.status === 404 ? 'That job has expired and been deleted.' : (r.json.detail || 'Something went wrong.')); }
        const job = r.json;
        if (job.status === 'error') { history.replaceState(null, '', location.pathname); return fail(job.error || 'That did not work.'); }
        if (job.status === 'done') {
          status.hidden = true; button.disabled = false;
          result.replaceChildren(); render(result, job, id); result.hidden = false;
          result.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
          return;
        }
        const waiting = job.status === 'queued' ? `Waiting for a free worker (${job.queue_position || 1} ahead of you or running now)` : job.stage;
        show(waiting, job.status === 'queued' ? 0.02 : Math.max(job.progress, 0.03));
        await sleep(job.status === 'queued' ? 2500 : 1500);
      }
    }
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      error.hidden = true; result.hidden = true;
      button.disabled = true;
      show('Uploading…', 0);
      const r = await send('POST', `/api/${kind}`, new FormData(form), (f) => show(`Uploading… ${Math.round(f * 100)}%`, f * 0.5));
      if (!r.ok) return fail(r.json.detail || 'That did not work.');
      show('Checking the file…', 0.5);
      follow(r.json.id);
    });
    return { follow, form };
  }

  // ---- rendering results --------------------------------------------------------------------
  function dl(name, label, id, primary) {
    return h('a', { class: 'btn' + (primary ? ' primary' : ''), href: `/api/jobs/${id}/files/${name}`, download: name }, label);
  }

  function renderMark(root, job, id) {
    const r = job.result || {};
    root.append(h('p', { class: 'verdict named', text: 'Your marked copies are ready.' }));
    if (r.note) root.append(h('div', { class: 'callout', text: r.note }));
    if (r.robust === false) {
      root.append(h('div', { class: 'callout', text: 'This picture is too flat or dark to carry a mark that survives heavy changes. The copies can be traced when shared as they are or lightly recompressed, but not after heavy recompression or a screen recording.' }));
    }
    root.append(h('p', {}, `One folder per recipient (${(r.recipients || []).length}), inside a zip. Files stay here for a limited time; download them now.`));
    root.append(h('div', { class: 'actions' }, dl('copies.zip', 'Download the copies (zip)', id, true), dl('invisimark-case.json', 'Download the case file', id, false)));
    root.append(h('div', { class: 'callout bad' }, h('strong', { text: 'Keep the case file. ' }),
      'It holds your secret key, which this service has not stored and cannot recover. Without it a leak cannot be traced; anyone who has it can make accusations look convincing, so do not share it. It is also inside the zip.'));
  }

  function renderTrace(root, job) {
    const r = job.result || {};
    const named = (r.verdict || []).length > 0;
    root.append(h('p', { class: 'verdict ' + (named ? 'named' : (r.lead ? 'lead' : '')), text: r.summary || 'Done.' }));
    const rows = r.matches || [];
    if (rows.length) {
      const segs = r.mode === 'segments';
      const head = h('tr', {}, h('th', { text: 'Recipient' }), h('th', { text: segs ? 'Segments matching' : 'Score' }), segs ? h('th', { text: 'Chance an unrelated person matches this well' }) : null);
      const table = h('table', {}, h('thead', {}, head), h('tbody', {}, ...rows.map((m) => h('tr', {},
        h('td', { text: m.recipient }),
        h('td', { text: segs ? `${m.agree} of ${m.clear}` : String(m.z) }),
        segs ? h('td', { text: `1 in ${Number(m.odds).toLocaleString()}` }) : null))));
      root.append(table);
    }
    if (r.segments_total) root.append(h('p', { class: 'hint', text: `Read ${r.segments_read} of ${r.segments_total} segments of the video.` }));
    root.append(h('div', { class: 'callout', text: 'A result is evidence, not proof. Check it against how and when you sent each copy before acting on it.' }));
  }

  const marker = runner('mark', renderMark);
  const tracer = runner('trace', renderTrace);

  // ---- small conveniences: what was chosen, is it too big, how many recipients ---------------------
  let limits = { max_mb: 300, max_recipients: 100 };
  fetch('/api/limits').then((r) => r.json()).then((l) => { limits = l; }).catch(() => {});
  const size = (n) => (n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);

  function watchFile(inputId, metaId) {
    const input = $(inputId), meta = $(metaId);
    if (!input || !meta) return;
    input.addEventListener('change', () => {
      const f = input.files[0];
      meta.classList.remove('bad');
      if (!f) { meta.hidden = true; return; }
      meta.hidden = false;
      if (f.size > limits.max_mb * 1048576) {
        meta.classList.add('bad');
        meta.textContent = `${f.name} is ${size(f.size)}; the limit is ${limits.max_mb} MB.`;
      } else {
        meta.textContent = `${f.name} \u00b7 ${size(f.size)}`;
      }
    });
  }
  watchFile('#mark-file', '#mark-filemeta');
  watchFile('#trace-file', '#trace-filemeta');

  const names = $('#mark-names'), count = $('#mark-count');
  if (names && count) {
    names.addEventListener('input', () => {
      const n = new Set(names.value.split('\n').map((x) => x.trim()).filter(Boolean)).size;
      count.hidden = n === 0;
      count.classList.toggle('bad', n > limits.max_recipients);
      count.textContent = n > limits.max_recipients ? `${n} recipients; the limit is ${limits.max_recipients}.` : `${n} recipient${n === 1 ? '' : 's'}`;
    });
  }
  for (const [formId, inputId] of [['#mark-form', '#mark-file'], ['#trace-form', '#trace-file']]) {
    const form = $(formId), input = $(inputId);
    if (form && input) form.addEventListener('submit', (e) => {
      const f = input.files[0];
      if (f && f.size > limits.max_mb * 1048576) { e.preventDefault(); e.stopImmediatePropagation(); }
    }, true);
  }

  // /#trace opens the trace tab
  if (location.hash === '#trace') showTab('trace');

  // resume a job after a refresh
  const m = /job=([\w-]+)&kind=(mark|trace)/.exec(location.hash);
  if (m) { showTab(m[2]); (m[2] === 'mark' ? marker : tracer).follow(m[1]); }
})();
