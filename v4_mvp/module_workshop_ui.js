(() => {
  const workshopState = {
    modules: [],
    prepared: null,
    loadedModuleId: null,
    functionHint: '',
    preparing: false,
  };

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
  }

  async function workshopApi(path, options = {}) {
    const response = await fetch(path, {
      headers: {'Content-Type': 'application/json', ...(options.headers || {})},
      ...options,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    return data;
  }

  function installStyles() {
    if (document.getElementById('moduleWorkshopStyles')) return;
    const style = document.createElement('style');
    style.id = 'moduleWorkshopStyles';
    style.textContent = `
      .mw-top-btn{margin-left:auto;border:1px solid #475569;background:#1f2937;color:#fff;border-radius:8px;padding:7px 11px}
      .mw-layout{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(280px,.55fr);gap:18px;margin-top:18px}
      .mw-step{display:flex;align-items:center;gap:8px;margin:0 0 8px;font-weight:800}
      .mw-num{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:999px;background:#111827;color:#fff;font-size:12px}
      .mw-code,.mw-data{width:100%;min-height:210px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:13px;line-height:1.45;resize:vertical}
      .mw-data{min-height:150px}
      .mw-card{border:1px solid #d9dfe8;border-radius:10px;padding:12px;margin-top:10px;background:#fff}
      .mw-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:10px}
      .mw-chip{display:inline-block;border:1px solid #d9dfe8;background:#f8fafc;border-radius:999px;padding:4px 8px;font-size:12px;margin:3px 4px 3px 0}
      .mw-map-row{display:grid;grid-template-columns:minmax(110px,.45fr) 28px minmax(180px,1fr);gap:8px;align-items:center;padding:7px 0;border-bottom:1px solid #eef2f7}
      .mw-map-row:last-child{border-bottom:0}
      .mw-preview{overflow:auto;max-height:270px;border:1px solid #e5e7eb;border-radius:8px;margin-top:10px}
      .mw-preview table{margin:0}
      .mw-result{white-space:pre-wrap;word-break:break-word;background:#0f172a;color:#e2e8f0;padding:14px;border-radius:9px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:13px;max-height:360px;overflow:auto}
      .mw-module{border:1px solid #d9dfe8;border-radius:10px;padding:11px;margin-top:9px;background:#fff}
      .mw-module-title{font-weight:800}
      .mw-subtle{font-size:12px;color:#64748b}
      .mw-warning{border:1px solid #fde68a;background:#fffbeb;color:#92400e;border-radius:9px;padding:9px 11px;font-size:12px;margin-top:10px}
      .mw-ok{border:1px solid #bbf7d0;background:#f0fdf4;color:#166534;border-radius:9px;padding:9px 11px;font-size:13px;margin-top:10px}
      .mw-error{border:1px solid #fecaca;background:#fef2f2;color:#991b1b;border-radius:9px;padding:9px 11px;font-size:13px;margin-top:10px}
      .mw-empty{border:1px dashed #cbd5e1;border-radius:9px;padding:16px;color:#64748b;text-align:center}
      .mw-details{margin-top:12px;border-top:1px solid #eef2f7;padding-top:10px}
      .mw-details summary{cursor:pointer;font-weight:700;color:#334155}
      @media(max-width:1000px){.mw-layout{grid-template-columns:1fr}.mw-map-row{grid-template-columns:1fr}.mw-map-row .mw-arrow{display:none}}
    `;
    document.head.appendChild(style);
  }

  function installTopButton() {
    const topbar = document.querySelector('.topbar');
    if (!topbar || document.getElementById('moduleWorkshopBtn')) return;
    const button = document.createElement('button');
    button.id = 'moduleWorkshopBtn';
    button.className = 'mw-top-btn';
    button.textContent = 'Modules';
    button.addEventListener('click', () => openModuleWorkshop());
    topbar.appendChild(button);
  }

  async function loadModules() {
    try {
      const payload = await workshopApi('/api/module-workshop/modules');
      workshopState.modules = payload.modules || [];
    } catch (error) {
      workshopState.modules = [];
      console.warn('Module Workshop list failed:', error);
    }
  }

  function savedModulesHtml() {
    if (!workshopState.modules.length) {
      return '<div class="mw-empty">저장된 Module이 없습니다.<br>함수를 한 번 실행한 뒤 그대로 저장해보세요.</div>';
    }
    return workshopState.modules.map(item => `
      <div class="mw-module">
        <div class="mw-module-title">${esc(item.title)}</div>
        <div class="mw-subtle">${esc(item.entryFunction)} · v${esc(item.version || '0.1.0')}</div>
        ${item.description ? `<div style="margin-top:6px">${esc(item.description)}</div>` : ''}
        ${item.question ? `<div class="mw-subtle" style="margin-top:6px">Question: ${esc(item.question)}</div>` : ''}
        <div class="mw-toolbar"><button class="ghost" data-use-module="${esc(item.id)}">Use</button><button class="ghost" data-copy-module="${esc(item.id)}">Copy JSON</button></div>
      </div>
    `).join('');
  }

  function workshopShell(moduleItem = null) {
    const sampleCode = moduleItem?.code || `def average(values):\n    """Return the arithmetic mean of one numeric column."""\n    return sum(values) / len(values)`;
    const title = moduleItem?.title || '';
    const description = moduleItem?.description || '';
    const question = moduleItem?.question || '';
    const assumptions = (moduleItem?.assumptions || []).join('\n');
    const limits = (moduleItem?.limits || []).join('\n');
    return `
      <div class="breadcrumb">Module Workshop</div>
      <div class="section-head">
        <div>
          <h1 style="margin:0">Module Workshop</h1>
          <div class="muted">Paste → Map → Run → Save. 기존 분석 함수를 Leesin Module로 최대한 적은 입력으로 감쌉니다.</div>
        </div>
        <button class="ghost" id="mwBackBtn">← Project</button>
      </div>
      <div class="mw-warning">MVP runner는 제한된 Python subset을 별도 프로세스에서 실행합니다. 완전한 보안 sandbox는 아니므로 신뢰하는 로컬 코드만 사용하세요. 현재 <code>math</code>, <code>statistics</code>와 기본 Python 연산을 지원합니다.</div>
      <div class="mw-layout">
        <div>
          <section class="panel">
            <div class="mw-step"><span class="mw-num">1</span> Python 함수 붙여넣기</div>
            <textarea id="mwCode" class="mw-code" spellcheck="false">${esc(sampleCode)}</textarea>
            <div class="mw-step" style="margin-top:16px"><span class="mw-num">2</span> Data 붙여넣기</div>
            <div class="muted">Excel 표, TSV, CSV를 헤더째 그대로 Ctrl+V 하세요.</div>
            <textarea id="mwData" class="mw-data" spellcheck="false" placeholder="N\truntime_ms\n5\t0.001\n10\t0.003"></textarea>
            <div class="mw-toolbar">
              <button class="primary" id="mwPrepareBtn">Prepare</button>
              <span class="muted">붙여넣기 후에는 자동으로 Prepare도 시도합니다.</span>
            </div>
            <div id="mwPrepareMessage"></div>
          </section>
          <section class="panel" id="mwPreparedPanel" style="display:none">
            <div class="mw-step"><span class="mw-num">3</span> Mapping 확인</div>
            <div id="mwFunctionChooser"></div>
            <div id="mwDetected"></div>
            <div id="mwMapping"></div>
            <div id="mwPreview"></div>
            <div class="mw-toolbar"><button class="primary" id="mwRunBtn">Run</button></div>
          </section>
          <section class="panel" id="mwResultPanel" style="display:none">
            <div class="mw-step"><span class="mw-num">4</span> Result</div>
            <div id="mwResult"></div>
            <details class="mw-details">
              <summary>Optional details</summary>
              <div class="field"><label>Module title</label><input id="mwTitle" value="${esc(title)}" placeholder="비워두면 함수 이름"></div>
              <div class="field"><label>Description</label><textarea id="mwDescription">${esc(description)}</textarea></div>
              <div class="field"><label>Question (human-readable, optional)</label><input id="mwQuestion" value="${esc(question)}" placeholder="이 Module을 지금 왜 쓰는가"></div>
              <div class="field"><label>Assumptions (one per line)</label><textarea id="mwAssumptions">${esc(assumptions)}</textarea></div>
              <div class="field"><label>Limits (one per line)</label><textarea id="mwLimits">${esc(limits)}</textarea></div>
            </details>
            <div class="mw-toolbar"><button class="primary" id="mwSaveBtn">Save Module</button><span class="muted">Optional details를 비워도 저장됩니다.</span></div>
            <div id="mwSaveMessage"></div>
          </section>
        </div>
        <aside class="panel" style="border-right:1px solid var(--line);padding:16px;background:#fff">
          <div class="section-head"><h2>Saved Modules</h2><button class="ghost" id="mwPasteModuleBtn">Paste Module JSON</button></div>
          <div class="muted">저장한 함수는 Data만 바꿔 다시 사용할 수 있습니다. JSON 복붙으로 다른 사람에게 전달할 수도 있습니다.</div>
          <div id="mwModuleList">${savedModulesHtml()}</div>
        </aside>
      </div>
    `;
  }

  function tableHtml(data) {
    if (!data?.preview?.length) return '<div class="mw-empty">Preview 없음</div>';
    const columns = data.columns || Object.keys(data.preview[0] || {});
    return `
      <div class="muted" style="margin-top:12px">Detected ${esc(data.rowCount)} rows · delimiter=${esc(data.delimiter)}</div>
      <div class="mw-preview"><table><thead><tr>${columns.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead>
      <tbody>${data.preview.map(row => `<tr>${columns.map(c => `<td>${esc(row[c])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
    `;
  }

  function renderPrepared(payload) {
    workshopState.prepared = payload;
    const panel = document.getElementById('mwPreparedPanel');
    panel.style.display = '';
    document.getElementById('mwResultPanel').style.display = 'none';

    const functions = payload.functions || [];
    const selected = payload.selectedFunction || {};
    document.getElementById('mwFunctionChooser').innerHTML = functions.length > 1 ? `
      <div class="field"><label>Detected function</label><select id="mwFunctionSelect">${functions.map(fn => `<option value="${esc(fn.name)}" ${fn.name === selected.name ? 'selected' : ''}>${esc(fn.name)}</option>`).join('')}</select></div>
    ` : `<div class="muted">Detected function: <strong>${esc(selected.name)}</strong></div>`;

    const params = selected.parameters || [];
    document.getElementById('mwDetected').innerHTML = `
      <div style="margin-top:8px">${params.length ? params.map(p => `<span class="mw-chip">${esc(p.name)}${p.required ? '' : `=${esc(p.default)}`}</span>`).join('') : '<span class="muted">No inputs</span>'}</div>
      ${selected.docstring ? `<div class="muted" style="margin-top:6px">${esc(selected.docstring)}</div>` : ''}
    `;

    const columns = payload.data?.columns || [];
    const suggested = payload.suggestedMapping || {};
    const mappingRows = params.filter(p => !['var_positional', 'var_keyword'].includes(p.kind)).map(p => {
      const options = [
        `<option value="">-- choose --</option>`,
        `<option value="__rows__" ${suggested[p.name] === '__rows__' ? 'selected' : ''}>Whole table (rows)</option>`,
        ...columns.map(c => `<option value="${esc(c)}" ${suggested[p.name] === c ? 'selected' : ''}>Column: ${esc(c)}</option>`),
        ...(!p.required ? [`<option value="__default__" ${suggested[p.name] === '__default__' || !suggested[p.name] ? 'selected' : ''}>Use default (${esc(p.default)})</option>`] : []),
      ];
      return `<div class="mw-map-row"><div><strong>${esc(p.name)}</strong>${p.annotation ? `<div class="mw-subtle">${esc(p.annotation)}</div>` : ''}</div><div class="mw-arrow">←</div><select class="mw-map-select" data-param="${esc(p.name)}">${options.join('')}</select></div>`;
    }).join('');
    document.getElementById('mwMapping').innerHTML = mappingRows || '<div class="muted" style="margin-top:8px">이 함수에는 연결할 입력이 없습니다.</div>';
    document.getElementById('mwPreview').innerHTML = tableHtml(payload.data);

    const select = document.getElementById('mwFunctionSelect');
    if (select) select.addEventListener('change', () => prepare(select.value));
    document.getElementById('mwRunBtn').onclick = runPrepared;
  }

  function currentMapping() {
    const mapping = {};
    document.querySelectorAll('.mw-map-select').forEach(select => {
      mapping[select.dataset.param] = select.value;
    });
    return mapping;
  }

  async function prepare(functionName = '') {
    if (workshopState.preparing) return;
    const code = document.getElementById('mwCode')?.value || '';
    const dataText = document.getElementById('mwData')?.value || '';
    const message = document.getElementById('mwPrepareMessage');
    if (!code.trim() || !dataText.trim()) {
      if (message) message.innerHTML = '<div class="mw-error">함수와 Data를 둘 다 붙여넣으세요.</div>';
      return;
    }
    workshopState.preparing = true;
    if (message) message.innerHTML = '<div class="muted" style="margin-top:10px">Preparing…</div>';
    try {
      const payload = await workshopApi('/api/module-workshop/prepare', {
        method: 'POST',
        body: JSON.stringify({code, dataText, functionName}),
      });
      if (message) message.innerHTML = '<div class="mw-ok">함수와 표를 읽었습니다. 자동 Mapping을 확인하세요.</div>';
      renderPrepared(payload);
    } catch (error) {
      workshopState.prepared = null;
      document.getElementById('mwPreparedPanel').style.display = 'none';
      if (message) message.innerHTML = `<div class="mw-error">${esc(error.message)}</div>`;
    } finally {
      workshopState.preparing = false;
    }
  }

  async function runPrepared() {
    const prepared = workshopState.prepared;
    if (!prepared) return;
    const button = document.getElementById('mwRunBtn');
    const resultPanel = document.getElementById('mwResultPanel');
    const resultBox = document.getElementById('mwResult');
    button.disabled = true;
    button.textContent = 'Running…';
    resultPanel.style.display = '';
    resultBox.innerHTML = '<div class="muted">Running…</div>';
    try {
      const result = await workshopApi('/api/module-workshop/run', {
        method: 'POST',
        body: JSON.stringify({
          code: document.getElementById('mwCode').value,
          functionName: prepared.selectedFunction.name,
          dataText: document.getElementById('mwData').value,
          mapping: currentMapping(),
        }),
      });
      resultBox.innerHTML = `
        <div class="mw-ok">${esc(result.function)} · ${esc(result.rowCount)} rows · ${Number(result.executionMs || 0).toFixed(3)} ms</div>
        <div class="mw-result">${esc(JSON.stringify(result.result, null, 2))}</div>
      `;
      const title = document.getElementById('mwTitle');
      if (title && !title.value.trim()) title.value = prepared.selectedFunction.name;
    } catch (error) {
      resultBox.innerHTML = `<div class="mw-error">${esc(error.message)}</div>`;
    } finally {
      button.disabled = false;
      button.textContent = 'Run';
    }
  }

  async function saveCurrentModule() {
    const prepared = workshopState.prepared;
    if (!prepared) return;
    const message = document.getElementById('mwSaveMessage');
    const button = document.getElementById('mwSaveBtn');
    button.disabled = true;
    try {
      const saved = await workshopApi('/api/module-workshop/modules', {
        method: 'POST',
        body: JSON.stringify({
          code: document.getElementById('mwCode').value,
          functionName: prepared.selectedFunction.name,
          title: document.getElementById('mwTitle')?.value || '',
          description: document.getElementById('mwDescription')?.value || '',
          question: document.getElementById('mwQuestion')?.value || '',
          assumptions: document.getElementById('mwAssumptions')?.value || '',
          limits: document.getElementById('mwLimits')?.value || '',
        }),
      });
      await loadModules();
      document.getElementById('mwModuleList').innerHTML = savedModulesHtml();
      bindSavedModuleButtons();
      message.innerHTML = `<div class="mw-ok">Saved: ${esc(saved.title)} · ${esc(saved.id)}</div>`;
    } catch (error) {
      message.innerHTML = `<div class="mw-error">${esc(error.message)}</div>`;
    } finally {
      button.disabled = false;
    }
  }

  function sharePayload(item) {
    return {
      leesinModule: 1,
      title: item.title || item.entryFunction,
      description: item.description || '',
      question: item.question || '',
      assumptions: item.assumptions || [],
      limits: item.limits || [],
      entryFunction: item.entryFunction,
      code: item.code,
      version: item.version || '0.1.0',
    };
  }

  async function copyModuleJson(moduleId) {
    const item = workshopState.modules.find(module => module.id === moduleId);
    if (!item) return;
    const text = JSON.stringify(sharePayload(item), null, 2);
    try {
      await navigator.clipboard.writeText(text);
      alert('Module JSON을 클립보드에 복사했습니다.');
    } catch (error) {
      window.prompt('Copy Module JSON', text);
    }
  }

  async function pasteModuleJson() {
    let text = '';
    try {
      if (navigator.clipboard?.readText) text = await navigator.clipboard.readText();
    } catch (error) {
      // Browser permission can reject clipboard reads; prompt is the fallback.
    }
    if (!text.trim()) text = window.prompt('Paste Module JSON') || '';
    if (!text.trim()) return;
    try {
      const item = JSON.parse(text);
      if (!item || item.leesinModule !== 1 || !item.code || !item.entryFunction) {
        throw new Error('Leesin Module JSON 형식이 아닙니다.');
      }
      mountWorkshop({
        title: item.title || item.entryFunction,
        description: item.description || '',
        question: item.question || '',
        assumptions: Array.isArray(item.assumptions) ? item.assumptions : [],
        limits: Array.isArray(item.limits) ? item.limits : [],
        entryFunction: item.entryFunction,
        code: item.code,
        version: item.version || '0.1.0',
      }, null);
      document.getElementById('mwData').focus();
    } catch (error) {
      alert(error.message);
    }
  }

  function bindSavedModuleButtons() {
    document.querySelectorAll('[data-use-module]').forEach(button => {
      button.addEventListener('click', () => {
        const item = workshopState.modules.find(module => module.id === button.dataset.useModule);
        if (item) mountWorkshop(item, item.id);
      });
    });
    document.querySelectorAll('[data-copy-module]').forEach(button => {
      button.addEventListener('click', () => copyModuleJson(button.dataset.copyModule));
    });
    const pasteButton = document.getElementById('mwPasteModuleBtn');
    if (pasteButton) pasteButton.onclick = pasteModuleJson;
  }

  function schedulePrepareAfterPaste() {
    window.setTimeout(() => {
      const code = document.getElementById('mwCode')?.value || '';
      const data = document.getElementById('mwData')?.value || '';
      if (code.trim() && data.trim()) prepare(workshopState.functionHint || '');
    }, 80);
  }

  function mountWorkshop(moduleItem = null, moduleId = null) {
    workshopState.loadedModuleId = moduleId;
    workshopState.functionHint = moduleItem?.entryFunction || '';
    workshopState.prepared = null;
    const main = document.getElementById('mainView');
    if (!main) return;
    main.innerHTML = workshopShell(moduleItem);
    document.getElementById('mwPrepareBtn').addEventListener('click', () => prepare(moduleItem?.entryFunction || ''));
    document.getElementById('mwSaveBtn').onclick = saveCurrentModule;
    document.getElementById('mwBackBtn').addEventListener('click', () => {
      if (window.renderProjectHome) window.renderProjectHome();
      else history.back();
    });
    document.getElementById('mwCode').addEventListener('paste', schedulePrepareAfterPaste);
    document.getElementById('mwData').addEventListener('paste', schedulePrepareAfterPaste);
    bindSavedModuleButtons();
    if (moduleItem) document.getElementById('mwData').focus();
  }

  async function openModuleWorkshop(moduleId = null) {
    await loadModules();
    const moduleItem = moduleId ? workshopState.modules.find(item => item.id === moduleId) : null;
    mountWorkshop(moduleItem, moduleId);
  }

  window.openModuleWorkshop = openModuleWorkshop;

  installStyles();
  installTopButton();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      installStyles();
      installTopButton();
    });
  }
})();
