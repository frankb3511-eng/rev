// Upload page: preview, drag & drop, submit, then redirect to the results run.
(() => {
  const form = document.getElementById('upload-form');
  const input = document.getElementById('image-input');
  const dz = document.getElementById('dropzone');
  const preview = document.getElementById('preview');
  const fileName = document.getElementById('file-name');
  const go = document.getElementById('go');
  const demo = document.getElementById('demo');

  const show = (file) => {
    if (!file) return;
    fileName.textContent = `${file.name} — ${(file.size / 1024).toFixed(0)} KiB`;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.hidden = false;
  };

  input.addEventListener('change', () => show(input.files[0]));

  ['dragenter', 'dragover'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    show(file);
  });

  // Demo mode is the honest fallback when no engine is reachable: the search is
  // synthetic, but every measurement after it is the real pipeline.
  const anyEngineChecked = () =>
    [...form.querySelectorAll('input[name=engines]')].some((c) => c.checked);
  const syncDemo = () => { if (!anyEngineChecked()) demo.checked = true; };
  form.querySelectorAll('input[name=engines]').forEach((c) => c.addEventListener('change', syncDemo));
  syncDemo();

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!input.files[0]) return;
    go.disabled = true;
    go.textContent = 'Starting…';
    try {
      const body = new FormData(form);
      const res = await fetch('/api/search', { method: 'POST', body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      window.location.href = data.results_url;
    } catch (err) {
      go.disabled = false;
      go.textContent = 'Search & verify';
      alert(`Could not start the search: ${err.message}`);
    }
  });
})();
