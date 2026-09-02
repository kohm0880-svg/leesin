(() => {
  const SUPPORTED = new Set(['.py', '.csv', '.tsv', '.txt']);
  let lastPythonName = '';
  let lastDataName = '';

  function ext(name) {
    const value = String(name || '').toLowerCase();
    const dot = value.lastIndexOf('.');
    return dot >= 0 ? value.slice(dot) : '';
  }

  function installStyles() {
    if (document.getElementById('mwFileInputStyles')) return;
    const style = document.createElement('style');
    style.id = 'mwFileInputStyles';
    style.textContent = `
      .mw-file-zone{border:1.5px dashed #94a3b8;border-radius:11px;background:#f8fafc;padding:14px 16px;margin:0 0 16px;transition:.15s ease;cursor:pointer}
      .mw-file-zone:hover,.mw-file-zone.is-dragging{border-color:#111827;background:#f1f5f9;box-shadow:inset 0 0 0 1px #11182718}
      .mw-file-main{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
      .mw-file-title{font-weight:800}
      .mw-file-hint{font-size:12px;color:#64748b;margin-top:3px}
      .mw-file-badges{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
      .mw-file-badge{display:inline-flex;align-items:center;gap:5px;border:1px solid #d9dfe8;border-radius:999px;background:#fff;padding:4px 8px;font-size:12px;color:#334155}
      .mw-file-message{margin-top:8px;font-size:12px;color:#475569}
      .mw-file-message.error{color:#991b1b}
    `;
    document.head.appendChild(style);
  }

  function kindFromFile(file, text) {
    const suffix = ext(file.name);
    if (suffix === '.py') return 'python';
    if (suffix === '.csv' || suffix === '.tsv') return 'data';
    if (suffix === '.txt') {
      return /^\s*(?:from\s+\S+\s+import|import\s+\S+|def\s+\w+\s*\()/m.test(text)
        ? 'python'
        : 'data';
    }
    return null;
  }

  function statusText() {
    const parts = [];
    if (lastPythonName) parts.push(`<span class="mw-file-badge">🐍 Python: ${escapeHtml(lastPythonName)}</span>`);
    if (lastDataName) parts.push(`<span class="mw-file-badge">▦ Data: ${escapeHtml(lastDataName)}</span>`);
    return parts.join('');
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
  }

  function setMessage(text, isError = false) {
    const box = document.getElementById('mwFileMessage');
    if (!box) return;
    box.textContent = text || '';
    box.className = `mw-file-message${isError ? ' error' : ''}`;
  }

  function renderBadges() {
    const box = document.getElementById('mwFileBadges');
    if (box) box.innerHTML = statusText();
  }

  async function readFiles(fileList) {
    const files = [...(fileList || [])];
    if (!files.length) return;

    const accepted = files.filter(file => SUPPORTED.has(ext(file.name)));
    if (!accepted.length) {
      setMessage('.py, .csv, .tsv, .txt 파일만 읽을 수 있습니다.', true);
      return;
    }

    let pythonPayload = null;
    let dataPayload = null;
    const rejected = files.filter(file => !SUPPORTED.has(ext(file.name))).map(file => file.name);

    for (const file of accepted) {
      let text;
      try {
        text = await file.text();
      } catch (error) {
        setMessage(`${file.name}을 읽지 못했습니다.`, true);
        return;
      }
      const kind = kindFromFile(file, text);
      if (kind === 'python') pythonPayload = {name: file.name, text};
      if (kind === 'data') dataPayload = {name: file.name, text};
    }

    // Loading a new .py means "new module source". Re-open a clean Workshop so a
    // previously loaded saved-module entry-function hint cannot leak into Prepare.
    if (pythonPayload && typeof window.openModuleWorkshop === 'function') {
      const preservedData = dataPayload?.text ?? document.getElementById('mwData')?.value ?? '';
      await window.openModuleWorkshop();
      await new Promise(resolve => setTimeout(resolve, 0));
      dataPayload = dataPayload || (preservedData ? {name: lastDataName || 'current data', text: preservedData} : null);
    }

    const code = document.getElementById('mwCode');
    const data = document.getElementById('mwData');
    if (!code || !data) return;

    if (pythonPayload) {
      code.value = pythonPayload.text;
      lastPythonName = pythonPayload.name;
      code.dispatchEvent(new Event('input', {bubbles: true}));
    }
    if (dataPayload) {
      data.value = dataPayload.text;
      lastDataName = dataPayload.name;
      data.dispatchEvent(new Event('input', {bubbles: true}));
    }

    renderBadges();
    const loaded = [pythonPayload?.name, dataPayload?.name].filter(Boolean).join(' + ');
    const suffix = rejected.length ? ` · 무시됨: ${rejected.join(', ')}` : '';
    setMessage(`${loaded || '파일'}을 읽었습니다.${suffix}`);

    if (code.value.trim() && data.value.trim()) {
      const prepareButton = document.getElementById('mwPrepareBtn');
      if (prepareButton) window.setTimeout(() => prepareButton.click(), 40);
    }
  }

  function enhanceWorkshop() {
    const code = document.getElementById('mwCode');
    const data = document.getElementById('mwData');
    if (!code || !data || document.getElementById('mwUnifiedFileZone')) return;

    const panel = code.closest('section.panel');
    if (!panel) return;

    const zone = document.createElement('div');
    zone.id = 'mwUnifiedFileZone';
    zone.className = 'mw-file-zone';
    zone.tabIndex = 0;
    zone.innerHTML = `
      <input id="mwUnifiedFileInput" type="file" multiple accept=".py,.csv,.tsv,.txt,text/x-python,text/csv,text/tab-separated-values,text/plain" hidden>
      <div class="mw-file-main">
        <div>
          <div class="mw-file-title">파일도 그냥 여기로 넣으세요</div>
          <div class="mw-file-hint">.py + .csv/.tsv를 같이 드래그하거나, 클릭해서 찾기 · 아래 칸에서는 기존처럼 복붙/수정 가능</div>
        </div>
        <button type="button" class="ghost" id="mwBrowseFilesBtn">Browse files</button>
      </div>
      <div class="mw-file-badges" id="mwFileBadges"></div>
      <div class="mw-file-message" id="mwFileMessage"></div>
    `;
    panel.insertBefore(zone, panel.firstChild);
    renderBadges();

    const input = document.getElementById('mwUnifiedFileInput');
    const browse = document.getElementById('mwBrowseFilesBtn');

    browse.addEventListener('click', event => {
      event.stopPropagation();
      input.click();
    });
    zone.addEventListener('click', event => {
      if (event.target === browse || browse.contains(event.target)) return;
      input.click();
    });
    zone.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        input.click();
      }
    });
    input.addEventListener('change', async () => {
      await readFiles(input.files);
      input.value = '';
    });

    for (const eventName of ['dragenter', 'dragover']) {
      zone.addEventListener(eventName, event => {
        event.preventDefault();
        event.stopPropagation();
        zone.classList.add('is-dragging');
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
      });
    }
    for (const eventName of ['dragleave', 'dragend']) {
      zone.addEventListener(eventName, event => {
        event.preventDefault();
        zone.classList.remove('is-dragging');
      });
    }
    zone.addEventListener('drop', async event => {
      event.preventDefault();
      event.stopPropagation();
      zone.classList.remove('is-dragging');
      await readFiles(event.dataTransfer?.files);
    });
  }

  installStyles();
  const observer = new MutationObserver(() => enhanceWorkshop());
  observer.observe(document.documentElement, {childList: true, subtree: true});
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', enhanceWorkshop);
  } else {
    enhanceWorkshop();
  }
})();
