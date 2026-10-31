(() => {
  'use strict';
  const $ = (sel) => document.querySelector(sel);
  const form = $('#check-form'), status = $('#check-status'), error = $('#check-error'), result = $('#check-result');
  const bar = $('#check-status .bar > i'), stage = $('#check-status .stage'), button = $('#check-form button.go');
  const input = $('#check-file'), meta = $('#check-filemeta');

  function h(tag, props, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (k === 'class') node.className = v; else if (k === 'text') node.textContent = v; else node.setAttribute(k, v);
    }
    for (const kid of kids) if (kid != null) node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return node;
  }
  const show = (text, pct) => { status.hidden = false; stage.textContent = text; bar.style.width = `${Math.round(pct * 100)}%`; };
  function fail(text) {
    status.hidden = true; result.hidden = true; button.disabled = false;
    error.textContent = text; error.hidden = false;
  }
  function upload(body) {
    return new Promise((resolve) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/check');
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) show(`Uploading… ${Math.round(e.loaded / e.total * 100)}%`, e.loaded / e.total * 0.5); };
      xhr.onload = () => {
        let json = {};
        try { json = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
        resolve({ ok: xhr.status >= 200 && xhr.status < 300, json });
      };
      xhr.onerror = () => resolve({ ok: false, json: { detail: 'The connection failed. Check your network and try again.' } });
      xhr.send(body);
    });
  }

  function render(job) {
    const r = job.result || {};
    result.replaceChildren();
    result.append(h('p', { class: 'verdict ' + (r.ai_marked ? 'lead' : ''), text: r.ai_marked ? 'This file carries an AI marking.' : 'No AI marking found.' }));
    for (const line of r.summary || []) result.append(h('p', { text: line }));
    const rows = [['SHA-256 of the file', r.sha256]];
    if (r.watermark) rows.push(['Watermark scores', `sync ${r.watermark.sync_score}, id ${r.watermark.id_score}`]);
    const m = r.metadata || {};
    for (const [label, value] of [['Signer', m.signer], ['Issuer', m.issuer], ['Signed at', m.signed_at], ['Provider (as stated)', m.provider], ['System (as stated)', m.system], ['Model (as stated)', m.model]]) {
      if (value) rows.push([label, value]);
    }
    result.append(h('details', {}, h('summary', { text: 'Details' }),
      h('table', {}, h('tbody', {}, ...rows.map(([k, v]) => h('tr', {}, h('th', { scope: 'row', text: k }), h('td', { text: v })))))));
    if ((job.files || []).includes('invisimark-result.json')) {
      result.append(h('div', { class: 'actions' }, h('a', { class: 'btn', href: `/api/jobs/${job.id}/files/invisimark-result.json`, download: 'invisimark-result.json' }, 'Download the signed result')));
    }
    result.hidden = false;
  }

  input.addEventListener('change', () => {
    const f = input.files[0];
    meta.hidden = !f;
    if (f) meta.textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} MB`;
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    error.hidden = true; result.hidden = true; button.disabled = true;
    show('Uploading…', 0);
    const posted = await upload(new FormData(form));
    if (!posted.ok) return fail(posted.json.detail || 'That did not work.');
    const id = posted.json.id;
    for (;;) {
      const r = await fetch(`/api/jobs/${id}`).then((x) => x.json().then((json) => ({ ok: x.ok, json }))).catch(() => ({ ok: false, json: {} }));
      if (!r.ok) return fail(r.json.detail || 'That check has expired.');
      if (r.json.status === 'error') return fail(r.json.error || 'That did not work.');
      if (r.json.status === 'done') {
        status.hidden = true; button.disabled = false;
        render(r.json);
        return;
      }
      show(r.json.status === 'queued' ? 'Waiting for a free worker…' : 'Looking for marks…', 0.6);
      await new Promise((done) => setTimeout(done, 1500));
    }
  });
})();
