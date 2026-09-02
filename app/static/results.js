// Results page: polls the run, renders the 1:1 check banner, the grouped result
// cards, and drives the side-by-side / overlay / difference comparison viewer.
(() => {
  const RUN = window.RUN_ID;
  const $ = (id) => document.getElementById(id);

  const statusText = $('status-text');
  const spinner = document.querySelector('.spinner');
  const headline = $('headline');
  const counts = $('counts');
  const groupsEl = $('groups');

  // ------------------------------------------------------------------ polling
  let timer = null;

  const poll = async () => {
    try {
      const res = await fetch(`/api/run/${RUN}`);
      const run = await res.json();
      statusText.textContent = run.status_text || run.status;
      renderEngines(run.engines);
      if (run.complete) {
        clearInterval(timer);
        spinner.classList.add('done');
        if (run.status === 'error') {
          headline.hidden = false;
          headline.className = 'headline zero';
          headline.textContent = run.error || 'The run failed.';
          return;
        }
        renderReport(run.report);
      }
    } catch (err) {
      statusText.textContent = `Poll failed: ${err.message}`;
    }
  };

  timer = setInterval(poll, 900);
  poll();

  // ------------------------------------------------------------------ engines
  const renderEngines = (engines) => {
    if (!engines || !engines.length) return;
    const card = $('engines-card');
    card.hidden = false;
    const rows = engines.map((e) => `
      <tr>
        <td>${esc(e.display_name)}</td>
        <td>${e.count}</td>
        <td>${e.elapsed_ms ?? 0} ms</td>
        <td class="${e.error ? 'err' : ''}">${e.error ? esc(e.error) : 'ok'}</td>
      </tr>`).join('');
    $('engines-table').innerHTML =
      `<thead><tr><th>Engine</th><th>Results</th><th>Time</th><th>Status</th></tr></thead><tbody>${rows}</tbody>`;
  };

  // ------------------------------------------------------------------- report
  const renderReport = (report) => {
    if (!report) return;
    const c = report.counts || {};
    const exact = (report.groups || []).find((g) => g.group === 'exact');
    const edited = (report.groups || []).find((g) => g.group === 'edited_resized');
    const oneToOne = (exact ? exact.count : 0) + (edited ? edited.count : 0);

    headline.hidden = false;
    headline.className = 'headline' + (oneToOne ? '' : ' zero');
    headline.textContent = oneToOne
      ? `FOUND ${oneToOne} TRUE 1:1 ${oneToOne === 1 ? 'MATCH' : 'MATCHES'}`
      : 'NO 1:1 MATCH FOUND';

    counts.hidden = false;
    counts.innerHTML = [
      ['Results collected', c.raw_results ?? 0],
      ['Unique URLs', c.unique_urls ?? 0],
      ['Compared', c.compared ?? 0],
      ['Deep-verified', c.verified ?? 0],
      ['1:1 matches', c.one_to_one ?? 0],
      ['Crops', c.crops ?? 0],
      ['Similar', c.similar ?? 0],
      ['Result cards', c.merged_cards ?? 0],
      ['Total time', `${((report.timings_ms || {}).total_ms ?? 0)} ms`],
    ].map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(String(v))}</dd></div>`).join('');

    renderOriginal(report.original);
    renderGroups(report.groups || []);
    renderTrace(report.stages || [], report.timings_ms || {});
  };

  const renderOriginal = (o) => {
    if (!o) return;
    $('original-card').hidden = false;
    $('original-img').src = `/api/run/${RUN}/original`;
    $('original-meta').innerHTML = [
      ['Dimensions', `${o.width} × ${o.height}`],
      ['Format', o.format],
      ['Bytes', o.bytes.toLocaleString()],
      ['SHA-256', `<span title="${esc(o.sha256)}">${esc(o.sha256.slice(0, 24))}…</span>`],
    ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  };

  const renderGroups = (groups) => {
    groupsEl.innerHTML = groups.map((g) => `
      <section class="group ${g.group}">
        <div class="group-head"><h2>${esc(g.title)}</h2><span class="n">${g.count}</span></div>
        <div class="grid">${g.results.map(card).join('')}</div>
      </section>`).join('');

    groupsEl.querySelectorAll('[data-compare]').forEach((btn) => {
      btn.addEventListener('click', () => openCompare(btn.dataset.compare, btn.dataset.title));
    });
  };

  // -------------------------------------------------------------- one card
  const card = (r) => {
    const engineChips = (r.engines || []).map((e) => `<span class="chip engine">✓ ${esc(e)}</span>`).join('');
    const pageChips = (r.pages || []).slice(0, 4)
      .map((p) => `<span class="chip page"><a href="${esc(p)}" target="_blank" rel="noopener">${esc(host(p))}</a></span>`)
      .join('');
    const engineLabels = (r.search_engine_labels || []).slice(0, 3)
      .map((l) => `<span class="chip">${esc(l)}</span>`).join('');
    const reasons = (r.verdict && r.verdict.reasons || []).map((x) => `<li>${esc(x)}</li>`).join('');
    const overlap = r.overlap != null
      ? `<p class="overlap">Estimated overlap: ${(r.overlap * 100).toFixed(0)}%</p>` : '';

    return `
    <article class="result">
      <div class="thumb"><img loading="lazy" src="/api/run/${RUN}/image/${esc(r.representative_cid)}" alt=""></div>
      <div class="body">
        <span class="badge ${esc(r.group)}">${esc(r.display)} <span class="conf">${r.confidence.toFixed(1)}% confidence</span></span>
        ${overlap}
        <div class="verdicts">
          <div><dt>Local verification</dt><dd class="local">${esc(r.display)} — ${r.confidence.toFixed(1)}%</dd></div>
          <div><dt>Search engine said</dt><dd>${engineLabels || '<em>—</em>'}</dd></div>
        </div>
        <div class="split"><h4>Found by</h4><div class="chips">${engineChips || '<span class="chip">—</span>'}</div></div>
        <div class="split"><h4>Found on</h4><div class="chips">${pageChips || '<span class="chip">—</span>'}</div></div>
        <div class="actions">
          <button class="primary" data-compare="${esc(r.representative_cid)}" data-title="${esc(r.display)}">Compare</button>
          <a href="${esc(r.best_url)}" target="_blank" rel="noopener"><button>Open source</button></a>
        </div>
        <details class="signals">
          <summary>Why this verdict (${(r.verdict && r.verdict.reasons || []).length} reasons)</summary>
          <ul class="reasons">${reasons}</ul>
          <pre>${esc(JSON.stringify(r.verdict, null, 2))}</pre>
        </details>
      </div>
    </article>`;
  };

  const renderTrace = (stages, timings) => {
    $('trace-card').hidden = false;
    const rows = stages.map((s) => `
      <tr><td>${esc(s.stage)}</td><td><pre style="margin:0;font-size:11.5px">${esc(JSON.stringify(rest(s)))}</pre></td></tr>`).join('');
    const trows = Object.entries(timings)
      .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v} ms</td></tr>`).join('');
    $('trace-table').innerHTML = `
      <thead><tr><th>Stage</th><th>Detail</th></tr></thead><tbody>${rows}</tbody>
      <thead><tr><th>Timing</th><th>Elapsed</th></tr></thead><tbody>${trows}</tbody>`;
  };

  const rest = ({ stage, ...others }) => others;
  const host = (u) => { try { return new URL(u).hostname; } catch { return u; } };
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ------------------------------------------------------ comparison viewer
  const modal = $('compare-modal');
  const holder = $('canvas-holder');
  const canvas = $('cmp-canvas');
  const imgA = $('cmp-a');
  const imgB = $('cmp-b');
  const ctx = canvas.getContext('2d');

  let mode = 'side';
  let zoom = 1;
  let natural = { w: 0, h: 0 };
  let cache = {};
  let currentCid = null;

  const load = (cid, view) => new Promise((resolve, reject) => {
    const key = `${cid}:${view}`;
    if (cache[key]) return resolve(cache[key]);
    const im = new Image();
    im.onload = () => { cache[key] = im; resolve(im); };
    im.onerror = reject;
    im.src = `/api/run/${RUN}/diff/${cid}?view=${view}`;
  });

  const openCompare = async (cid, title) => {
    currentCid = cid;
    modal.hidden = false;
    $('cmp-title').textContent = `ORIGINAL vs ${title}`;
    $('cmp-sub').textContent = 'Loading aligned comparison…';
    cache = {};
    setMode('side');
    try {
      const [a, b, diff, heat] = await Promise.all([
        load(cid, 'original'), load(cid, 'candidate'),
        load(cid, 'difference'), load(cid, 'heatmap'),
      ]);
      natural = { w: a.naturalWidth, h: a.naturalHeight };
      imgA.src = a.src;
      imgB.src = b.src;
      canvas.width = a.naturalWidth;
      canvas.height = a.naturalHeight;
      const meta = await (await fetch(`/api/run/${RUN}/compare/${cid}`)).json();
      $('cmp-foot').innerHTML =
        `Aligned by <code>${esc(meta.align_method)}</code> · ` +
        `original coverage <code>${(meta.coverage_orig * 100).toFixed(1)}%</code> · ` +
        `candidate coverage <code>${(meta.coverage_cand * 100).toFixed(1)}%</code> · ` +
        `${meta.size[0]} × ${meta.size[1]}`;
      $('cmp-sub').textContent = 'Both frames are registered before comparison, so a resize, crop or border does not show up as a false difference.';
      fit();
      draw();
    } catch (err) {
      $('cmp-sub').textContent = `Could not build the comparison: ${err.message}`;
    }
  };

  const draw = () => {
    if (!currentCid) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (mode === 'difference') {
      ctx.drawImage(cache[`${currentCid}:difference`], 0, 0);
    } else if (mode === 'heatmap') {
      ctx.drawImage(cache[`${currentCid}:heatmap`], 0, 0);
    } else if (mode === 'overlay') {
      ctx.drawImage(cache[`${currentCid}:candidate`], 0, 0);
      holder.style.setProperty('--blend', String($('blend').value / 100));
    }
  };

  const setMode = (m) => {
    mode = m;
    holder.className = `canvas-holder ${m}`;
    document.querySelectorAll('.modes button').forEach((b) =>
      b.classList.toggle('on', b.dataset.mode === m));
    $('overlay-wrap').hidden = m !== 'overlay';
    draw();
  };

  const applyZoom = () => {
    holder.style.transform = `scale(${zoom})`;
    $('zoom-val').textContent = `${Math.round(zoom * 100)}%`;
  };

  const fit = () => {
    const vp = $('viewport');
    const pad = 34;
    const width = mode === 'side' ? natural.w * 2 + pad : natural.w;
    if (!natural.w) return;
    zoom = Math.min((vp.clientWidth - pad) / width, (vp.clientHeight - pad) / natural.h, 1);
    applyZoom();
  };

  document.querySelectorAll('.modes button').forEach((b) =>
    b.addEventListener('click', () => { setMode(b.dataset.mode); fit(); }));
  $('blend').addEventListener('input', () => {
    $('blend-val').textContent = `${$('blend').value}%`;
    draw();
  });
  $('zoom-in').addEventListener('click', () => { zoom = Math.min(6, zoom * 1.25); applyZoom(); });
  $('zoom-out').addEventListener('click', () => { zoom = Math.max(0.1, zoom / 1.25); applyZoom(); });
  $('zoom-fit').addEventListener('click', fit);
  $('cmp-close').addEventListener('click', () => { modal.hidden = true; currentCid = null; });
  modal.addEventListener('click', (e) => { if (e.target === modal) modal.hidden = true; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') modal.hidden = true; });
})();
