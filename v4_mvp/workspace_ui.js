(() => {
  const ws = {
    activeProjectId: null,
    project: null,
    workspace: null,
    selected: new Map(),
    lastSelectedKey: null,
    modules: [],
    moduleQuery: '',
    favoriteModules: new Set(JSON.parse(localStorage.getItem('leesin.favoriteModules') || '[]')),
    collapsed: JSON.parse(localStorage.getItem('leesin.explorerCollapsed') || '{}'),
    dragItems: [],
    flowStep: 'data',
  };

  const FLOW = [
    ['data', 'Data'],
    ['module', 'Module'],
    ['mapping', 'Mapping'],
    ['analysis', 'Analysis'],
    ['result', 'Result'],
    ['next', 'Next'],
  ];

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: {'Content-Type': 'application/json', ...(options.headers || {})},
      ...options,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    return data;
  }

  function installStyles() {
    if (document.getElementById('leesinWorkspaceStyles')) return;
    const style = document.createElement('style');
    style.id = 'leesinWorkspaceStyles';
    style.textContent = `
      :root{--leesin-left-width:${localStorage.getItem('leesin.leftWidth') || '290px'};--leesin-right-width:${localStorage.getItem('leesin.rightWidth') || '300px'}}
      .shell{grid-template-columns:var(--leesin-left-width) 5px minmax(0,1fr) 5px var(--leesin-right-width)!important;min-width:0}
      .shell>aside:first-child{grid-column:1;min-width:0;padding:12px;overflow:auto}
      .shell>main{grid-column:3;min-width:0;padding:0!important;overflow:auto;position:relative}
      .ws-resize{width:5px;background:#eef2f7;cursor:col-resize;position:relative;z-index:8}
      .ws-resize:hover,.ws-resize.dragging{background:#94a3b8}
      .ws-resize.left{grid-column:2}.ws-resize.right{grid-column:4}
      .ws-right{grid-column:5;border-left:1px solid var(--line);background:#fbfcfd;padding:12px;overflow:auto;min-width:0}
      body.ws-left-off .shell{grid-template-columns:0 0 minmax(0,1fr) 5px var(--leesin-right-width)!important}
      body.ws-left-off .shell>aside:first-child,body.ws-left-off .ws-resize.left{display:none}
      body.ws-right-off .shell{grid-template-columns:var(--leesin-left-width) 5px minmax(0,1fr) 0 0!important}
      body.ws-right-off .ws-right,body.ws-right-off .ws-resize.right{display:none}
      body.ws-left-off.ws-right-off .shell{grid-template-columns:0 0 minmax(0,1fr) 0 0!important}
      .ws-top-toggle{border:1px solid #475569;background:#1f2937;color:#fff;border-radius:8px;padding:7px 10px;font-size:12px}
      #moduleWorkshopBtn{display:none!important}
      .ws-main-pad{padding:22px}
      .ws-flow{position:sticky;top:0;z-index:7;background:#ffffffee;backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:10px 18px;display:flex;align-items:center;gap:5px;overflow:auto}
      .ws-flow-step{border:0;background:transparent;color:#94a3b8;font-size:12px;font-weight:700;padding:6px 8px;border-radius:7px;white-space:nowrap}
      .ws-flow-step.active{background:#111827;color:#fff}.ws-flow-step.done{color:#334155}.ws-flow-step.clickable{cursor:pointer}.ws-flow-arrow{color:#cbd5e1;font-size:12px}
      .ws-project-title{font-size:13px;font-weight:800;margin:12px 0 7px;display:flex;align-items:center;gap:6px}
      .ws-toolbar{display:flex;gap:5px;align-items:center;flex-wrap:wrap;margin:7px 0 9px}
      .ws-icon-btn{border:1px solid #d5dbe4;background:#fff;border-radius:7px;padding:5px 7px;font-size:12px;cursor:pointer}.ws-icon-btn:disabled{opacity:.35;cursor:default}
      .ws-tree{font-size:13px;user-select:none}.ws-section{margin-top:3px}.ws-section-head,.ws-folder-row,.ws-item-row{display:flex;align-items:center;gap:5px;min-height:29px;padding:3px 6px;border-radius:6px;white-space:nowrap;overflow:hidden}
      .ws-section-head,.ws-folder-row{font-weight:650;cursor:pointer}.ws-section-head:hover,.ws-folder-row:hover,.ws-item-row:hover{background:#f1f5f9}
      .ws-item-row{cursor:default}.ws-item-row.selected{background:#dbeafe;outline:1px solid #93c5fd}.ws-item-row.dragging{opacity:.45}.ws-folder-row.drop-target,.ws-section-head.drop-target{background:#dcfce7;outline:1px solid #86efac}
      .ws-caret{width:14px;text-align:center;color:#64748b;flex:0 0 14px}.ws-icon{width:18px;text-align:center;flex:0 0 18px}.ws-label{overflow:hidden;text-overflow:ellipsis}.ws-meta{margin-left:auto;color:#94a3b8;font-size:11px}
      .ws-depth-1{padding-left:18px}.ws-depth-2{padding-left:34px}.ws-depth-3{padding-left:50px}.ws-depth-4{padding-left:66px}.ws-depth-5{padding-left:82px}
      .ws-empty{color:#94a3b8;font-size:12px;padding:5px 8px 7px 27px}.ws-trash-name{text-decoration:line-through;color:#64748b}
      .ws-file-view{margin:22px}.ws-code-preview{white-space:pre;overflow:auto;max-height:65vh;background:#0f172a;color:#e2e8f0;border-radius:10px;padding:14px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;line-height:1.5}
      .ws-module-search{width:100%;margin:6px 0 9px}.ws-module-tabs{display:flex;gap:5px;margin-bottom:8px}.ws-module-card{border:1px solid var(--line);background:#fff;border-radius:9px;padding:10px;margin:7px 0}.ws-module-name{font-weight:800}.ws-module-author{font-size:11px;color:#64748b}.ws-module-desc{font-size:12px;color:#475569;margin-top:5px;line-height:1.4}.ws-module-actions{display:flex;gap:6px;margin-top:8px}.ws-star{border:0;background:transparent;font-size:17px;padding:0;cursor:pointer}.ws-right h2{margin:2px 0 6px;font-size:18px}
      .ws-right-head{display:flex;justify-content:space-between;align-items:center;gap:8px}.ws-right-section{font-size:12px;font-weight:800;color:#475569;margin:13px 0 5px;text-transform:uppercase;letter-spacing:.03em}
      .ws-file-input{display:none}
      @media(max-width:980px){:root{--leesin-left-width:240px;--leesin-right-width:260px}.ws-right{font-size:12px}}
    `;
    document.head.appendChild(style);
  }

  function installLayout() {
    const shell = document.querySelector('.shell');
    const left = shell?.querySelector(':scope > aside');
    const main = shell?.querySelector(':scope > main');
    if (!shell || !left || !main || document.getElementById('leesinRightModules')) return;

    const leftHandle = document.createElement('div');
    leftHandle.className = 'ws-resize left';
    const rightHandle = document.createElement('div');
    rightHandle.className = 'ws-resize right';
    const right = document.createElement('aside');
    right.className = 'ws-right';
    right.id = 'leesinRightModules';
    right.innerHTML = '<div class="muted">Modules loading…</div>';
    shell.insertBefore(leftHandle, main);
    shell.insertBefore(rightHandle, main.nextSibling);
    shell.appendChild(right);

    const mainView = document.getElementById('mainView');
    if (mainView) mainView.classList.add('ws-main-pad');
    const flow = document.createElement('div');
    flow.id = 'leesinFlowBar';
    flow.className = 'ws-flow';
    main.insertBefore(flow, mainView);

    setupResize(leftHandle, '--leesin-left-width', 'leesin.leftWidth', true);
    setupResize(rightHandle, '--leesin-right-width', 'leesin.rightWidth', false);
    installTopToggles();
    renderFlow();
  }

  function setupResize(handle, cssVar, storageKey, fromLeft) {
    handle.addEventListener('pointerdown', event => {
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      handle.classList.add('dragging');
      const shell = document.querySelector('.shell');
      const startX = event.clientX;
      const raw = getComputedStyle(document.documentElement).getPropertyValue(cssVar).trim();
      const startWidth = parseFloat(raw) || 280;
      const onMove = move => {
        const delta = fromLeft ? move.clientX - startX : startX - move.clientX;
        const width = Math.max(190, Math.min(520, startWidth + delta));
        document.documentElement.style.setProperty(cssVar, `${width}px`);
        localStorage.setItem(storageKey, `${width}px`);
      };
      const onUp = up => {
        handle.releasePointerCapture(up.pointerId);
        handle.classList.remove('dragging');
        handle.removeEventListener('pointermove', onMove);
        handle.removeEventListener('pointerup', onUp);
      };
      handle.addEventListener('pointermove', onMove);
      handle.addEventListener('pointerup', onUp);
    });
  }

  function installTopToggles() {
    const topbar = document.querySelector('.topbar');
    if (!topbar || document.getElementById('wsProjectsToggle')) return;
    const spacer = document.createElement('div');
    spacer.style.flex = '1';
    topbar.appendChild(spacer);
    const leftBtn = document.createElement('button');
    leftBtn.id = 'wsProjectsToggle';
    leftBtn.className = 'ws-top-toggle';
    leftBtn.textContent = '☰ Projects';
    const rightBtn = document.createElement('button');
    rightBtn.id = 'wsModulesToggle';
    rightBtn.className = 'ws-top-toggle';
    rightBtn.textContent = 'Modules ☰';
    leftBtn.onclick = () => {
      document.body.classList.toggle('ws-left-off');
      localStorage.setItem('leesin.leftOff', document.body.classList.contains('ws-left-off') ? '1' : '0');
    };
    rightBtn.onclick = () => {
      document.body.classList.toggle('ws-right-off');
      localStorage.setItem('leesin.rightOff', document.body.classList.contains('ws-right-off') ? '1' : '0');
    };
    topbar.appendChild(leftBtn);
    topbar.appendChild(rightBtn);
    if (localStorage.getItem('leesin.leftOff') === '1') document.body.classList.add('ws-left-off');
    if (localStorage.getItem('leesin.rightOff') === '1') document.body.classList.add('ws-right-off');
  }

  function renderFlow() {
    const box = document.getElementById('leesinFlowBar');
    if (!box) return;
    const activeIndex = Math.max(0, FLOW.findIndex(([id]) => id === ws.flowStep));
    box.innerHTML = FLOW.map(([id, label], index) => {
      const cls = index === activeIndex ? 'active' : index < activeIndex ? 'done' : '';
      const clickable = id === 'data' || id === 'module';
      const step = `<button class="ws-flow-step ${cls} ${clickable ? 'clickable' : ''}" data-flow="${id}">${esc(label)}</button>`;
      return index < FLOW.length - 1 ? step + '<span class="ws-flow-arrow">→</span>' : step;
    }).join('');
    box.querySelectorAll('[data-flow]').forEach(button => {
      button.addEventListener('click', () => {
        const id = button.dataset.flow;
        if (id === 'data' && typeof window.renderProjectHome === 'function') {
          window.renderProjectHome();
          setFlow('data');
        }
        if (id === 'module' && typeof window.openModuleWorkshop === 'function') {
          window.openModuleWorkshop();
          setFlow('module');
        }
      });
    });
  }

  function setFlow(step) {
    ws.flowStep = step;
    renderFlow();
  }

  function detectFlowFromMain() {
    const main = document.getElementById('mainView');
    if (!main) return;
    const breadcrumb = main.querySelector('.breadcrumb')?.textContent || '';
    const resultPanel = document.getElementById('mwResultPanel');
    const preparedPanel = document.getElementById('mwPreparedPanel');
    if (breadcrumb.includes('Module Workshop')) {
      if (resultPanel && getComputedStyle(resultPanel).display !== 'none' && document.getElementById('mwResult')?.textContent.trim()) setFlow('result');
      else if (preparedPanel && getComputedStyle(preparedPanel).display !== 'none') setFlow('mapping');
      else setFlow('module');
      return;
    }
    if (breadcrumb.includes('Analyses')) {
      setFlow('result');
      return;
    }
    if (breadcrumb.includes('Proposals')) {
      setFlow('next');
      return;
    }
    if (ws.activeProjectId) setFlow('data');
  }

  async function refreshProject(projectId = ws.activeProjectId) {
    if (!projectId) return;
    try {
      const [project, workspace] = await Promise.all([
        api(`/api/projects/${projectId}`),
        api(`/api/projects/${projectId}/workspace`),
      ]);
      ws.activeProjectId = projectId;
      ws.project = project;
      ws.workspace = workspace;
      renderExplorer();
    } catch (error) {
      console.warn('Workspace refresh failed:', error);
    }
  }

  function sectionOpen(id, defaultOpen = true) {
    return ws.collapsed[id] === undefined ? defaultOpen : !ws.collapsed[id];
  }

  function toggleSection(id) {
    ws.collapsed[id] = sectionOpen(id);
    localStorage.setItem('leesin.explorerCollapsed', JSON.stringify(ws.collapsed));
    renderExplorer();
  }

  function itemKey(type, id, trashId = '') {
    return trashId ? `trash:${trashId}` : `${type}:${id}`;
  }

  function selectedFolderId() {
    const values = [...ws.selected.values()];
    if (values.length !== 1) return null;
    return values[0].type === 'folder' ? values[0].id : null;
  }

  function rowHtml({type, id, name, depth = 1, icon = '📄', meta = '', trashId = '', draggable = false}) {
    const key = itemKey(type, id, trashId);
    const selected = ws.selected.has(key) ? ' selected' : '';
    const trash = trashId ? ' ws-trash-name' : '';
    return `<div class="ws-item-row ws-depth-${Math.min(depth, 5)}${selected}" data-ws-key="${esc(key)}" data-type="${esc(type)}" data-id="${esc(id)}" ${trashId ? `data-trash-id="${esc(trashId)}"` : ''} ${draggable ? 'draggable="true"' : ''}><span class="ws-caret"></span><span class="ws-icon">${icon}</span><span class="ws-label${trash}">${esc(name)}</span>${meta ? `<span class="ws-meta">${esc(meta)}</span>` : ''}</div>`;
  }

  function folderHtml(folder, foldersByParent, filesByParent, depth) {
    const key = `folder:${folder.id}`;
    const open = sectionOpen(key, true);
    const selected = ws.selected.has(key) ? ' selected' : '';
    let html = `<div class="ws-folder-row ws-depth-${Math.min(depth, 5)}${selected}" data-folder-id="${esc(folder.id)}" data-ws-key="${esc(key)}" data-type="folder" data-id="${esc(folder.id)}" draggable="true"><span class="ws-caret">${open ? '▾' : '▸'}</span><span class="ws-icon">📁</span><span class="ws-label">${esc(folder.name)}</span></div>`;
    if (open) {
      for (const child of foldersByParent.get(folder.id) || []) html += folderHtml(child, foldersByParent, filesByParent, depth + 1);
      for (const file of filesByParent.get(folder.id) || []) html += rowHtml({type:'file', id:file.id, name:file.name, depth:depth + 1, icon:fileIcon(file.name), meta:formatSize(file.size), draggable:true});
    }
    return html;
  }

  function fileIcon(name) {
    const lower = String(name || '').toLowerCase();
    if (lower.endsWith('.py')) return '🐍';
    if (lower.endsWith('.csv') || lower.endsWith('.tsv')) return '▦';
    if (lower.endsWith('.json')) return '{}';
    if (lower.endsWith('.md') || lower.endsWith('.txt')) return '📝';
    return '📄';
  }

  function formatSize(value) {
    const n = Number(value || 0);
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }

  function renderExplorer() {
    const tree = document.getElementById('projectTree');
    if (!tree || !ws.project || !ws.workspace) return;
    const folders = ws.workspace.folders || [];
    const files = ws.workspace.files || [];
    const foldersByParent = new Map();
    const filesByParent = new Map();
    for (const folder of folders) {
      const parent = folder.parentFolderId || '__root__';
      if (!foldersByParent.has(parent)) foldersByParent.set(parent, []);
      foldersByParent.get(parent).push(folder);
    }
    for (const file of files) {
      const parent = file.parentFolderId || '__root__';
      if (!filesByParent.has(parent)) filesByParent.set(parent, []);
      filesByParent.get(parent).push(file);
    }

    const filesOpen = sectionOpen('section:files', true);
    const dataOpen = sectionOpen('section:data', true);
    const analysisOpen = sectionOpen('section:analyses', false);
    const proposalOpen = sectionOpen('section:proposals', false);
    const trashOpen = sectionOpen('section:trash', false);

    let filesBody = '';
    for (const folder of foldersByParent.get('__root__') || []) filesBody += folderHtml(folder, foldersByParent, filesByParent, 1);
    for (const file of filesByParent.get('__root__') || []) filesBody += rowHtml({type:'file', id:file.id, name:file.name, depth:1, icon:fileIcon(file.name), meta:formatSize(file.size), draggable:true});
    if (!filesBody) filesBody = '<div class="ws-empty">Drop or add files here.</div>';

    const clusters = (ws.project.clusters || []).map(c => rowHtml({type:'cluster', id:c.id, name:c.name || c.filename, depth:1, icon:'▦', meta:c.filename || ''})).join('') || '<div class="ws-empty">No data clusters.</div>';
    const analyses = (ws.project.analyses || []).map(a => rowHtml({type:'analysis', id:a.id, name:a.outcome?.title || a.id, depth:1, icon:'📄', meta:a.outcome?.status_label || a.status || ''})).join('') || '<div class="ws-empty">No analyses.</div>';
    const proposals = (ws.project.proposals || []).map(p => rowHtml({type:'proposal', id:p.id, name:p.id, depth:1, icon:'➜', meta:p.status || ''})).join('') || '<div class="ws-empty">No proposals.</div>';
    const trash = (ws.workspace.trash || []).map(t => rowHtml({type:t.type, id:t.id, trashId:t.trashId, name:t.name, depth:1, icon:'🗑', meta:t.type})).join('') || '<div class="ws-empty">Trash is empty.</div>';

    tree.innerHTML = `
      <div class="ws-project-title">📁 <span>${esc(ws.project.title)}</span></div>
      <div class="ws-toolbar">
        <button class="ws-icon-btn" id="wsAddFiles">+ File</button>
        <button class="ws-icon-btn" id="wsNewFolder">+ Folder</button>
        <button class="ws-icon-btn" id="wsDeleteSelected" disabled>Delete</button>
        <button class="ws-icon-btn" id="wsRestoreSelected" style="display:none">Restore</button>
        <button class="ws-icon-btn" id="wsPurgeSelected" style="display:none">Delete forever</button>
        <input class="ws-file-input" id="wsFilePicker" type="file" multiple>
      </div>
      <div class="ws-tree">
        <div class="ws-section">
          <div class="ws-section-head" data-section="section:files" data-root-drop="true"><span class="ws-caret">${filesOpen ? '▾' : '▸'}</span><span class="ws-icon">📁</span><span class="ws-label">Files</span><span class="ws-meta">${files.length}</span></div>
          ${filesOpen ? filesBody : ''}
        </div>
        <div class="ws-section"><div class="ws-section-head" data-section="section:data"><span class="ws-caret">${dataOpen ? '▾' : '▸'}</span><span class="ws-icon">▦</span><span class="ws-label">Data</span><span class="ws-meta">${ws.project.clusters?.length || 0}</span></div>${dataOpen ? clusters : ''}</div>
        <div class="ws-section"><div class="ws-section-head" data-section="section:analyses"><span class="ws-caret">${analysisOpen ? '▾' : '▸'}</span><span class="ws-icon">📄</span><span class="ws-label">Analyses</span><span class="ws-meta">${ws.project.analyses?.length || 0}</span></div>${analysisOpen ? analyses : ''}</div>
        <div class="ws-section"><div class="ws-section-head" data-section="section:proposals"><span class="ws-caret">${proposalOpen ? '▾' : '▸'}</span><span class="ws-icon">➜</span><span class="ws-label">Proposals</span><span class="ws-meta">${ws.project.proposals?.length || 0}</span></div>${proposalOpen ? proposals : ''}</div>
        <div class="ws-section"><div class="ws-section-head" data-section="section:trash"><span class="ws-caret">${trashOpen ? '▾' : '▸'}</span><span class="ws-icon">🗑</span><span class="ws-label">Trash</span><span class="ws-meta">${ws.workspace.trash?.length || 0}</span></div>${trashOpen ? trash : ''}</div>
      </div>`;
    bindExplorer();
    updateToolbar();
  }

  function bindExplorer() {
    const tree = document.getElementById('projectTree');
    tree.querySelectorAll('[data-section]').forEach(row => row.addEventListener('click', () => toggleSection(row.dataset.section)));
    tree.querySelectorAll('.ws-folder-row').forEach(row => {
      row.addEventListener('click', event => {
        if (event.target.classList.contains('ws-caret')) {
          toggleSection(`folder:${row.dataset.folderId}`);
          return;
        }
        selectRow(row, event);
      });
      row.addEventListener('dblclick', event => { event.preventDefault(); renameSelectedRow(row); });
      setupDragSource(row);
      setupFolderDrop(row, row.dataset.folderId);
    });
    tree.querySelectorAll('.ws-item-row').forEach(row => {
      row.addEventListener('click', event => selectRow(row, event));
      row.addEventListener('dblclick', () => openRow(row));
      if (row.draggable) setupDragSource(row);
    });
    const root = tree.querySelector('[data-root-drop="true"]');
    if (root) setupFolderDrop(root, null);

    document.getElementById('wsAddFiles').onclick = () => document.getElementById('wsFilePicker').click();
    document.getElementById('wsFilePicker').onchange = async event => {
      await uploadFiles(event.target.files);
      event.target.value = '';
    };
    document.getElementById('wsNewFolder').onclick = createNewFolder;
    document.getElementById('wsDeleteSelected').onclick = deleteSelected;
    document.getElementById('wsRestoreSelected').onclick = restoreSelected;
    document.getElementById('wsPurgeSelected').onclick = purgeSelected;
  }

  function visibleRows() {
    return [...document.querySelectorAll('#projectTree [data-ws-key]')];
  }

  function selectRow(row, event) {
    const key = row.dataset.wsKey;
    const descriptor = {
      key,
      type: row.dataset.type,
      id: row.dataset.id,
      trashId: row.dataset.trashId || null,
    };
    const rows = visibleRows();
    if (event.shiftKey && ws.lastSelectedKey) {
      const start = rows.findIndex(r => r.dataset.wsKey === ws.lastSelectedKey);
      const end = rows.findIndex(r => r.dataset.wsKey === key);
      if (start >= 0 && end >= 0) {
        if (!event.ctrlKey && !event.metaKey) ws.selected.clear();
        for (const target of rows.slice(Math.min(start, end), Math.max(start, end) + 1)) {
          ws.selected.set(target.dataset.wsKey, {key:target.dataset.wsKey, type:target.dataset.type, id:target.dataset.id, trashId:target.dataset.trashId || null});
        }
      }
    } else if (event.ctrlKey || event.metaKey) {
      if (ws.selected.has(key)) ws.selected.delete(key); else ws.selected.set(key, descriptor);
      ws.lastSelectedKey = key;
    } else {
      ws.selected.clear();
      ws.selected.set(key, descriptor);
      ws.lastSelectedKey = key;
    }
    updateSelectionClasses();
    updateToolbar();
  }

  function updateSelectionClasses() {
    document.querySelectorAll('#projectTree [data-ws-key]').forEach(row => row.classList.toggle('selected', ws.selected.has(row.dataset.wsKey)));
  }

  function updateToolbar() {
    const values = [...ws.selected.values()];
    const allTrash = values.length > 0 && values.every(item => item.trashId);
    const deleteBtn = document.getElementById('wsDeleteSelected');
    const restoreBtn = document.getElementById('wsRestoreSelected');
    const purgeBtn = document.getElementById('wsPurgeSelected');
    if (!deleteBtn) return;
    deleteBtn.disabled = !values.length || allTrash;
    deleteBtn.style.display = allTrash ? 'none' : '';
    restoreBtn.style.display = allTrash ? '' : 'none';
    purgeBtn.style.display = allTrash ? '' : 'none';
  }

  async function createNewFolder() {
    if (!ws.activeProjectId) return;
    const name = window.prompt('New folder name');
    if (!name?.trim()) return;
    try {
      await api(`/api/projects/${ws.activeProjectId}/workspace/folders`, {
        method:'POST', body:JSON.stringify({name:name.trim(), parentFolderId:selectedFolderId()})
      });
      ws.selected.clear();
      await refreshProject();
    } catch (error) { alert(error.message); }
  }

  function isTextFile(file) {
    const name = file.name.toLowerCase();
    return file.type.startsWith('text/') || ['.py','.csv','.tsv','.txt','.json','.md','.yaml','.yml','.toml','.ini','.log'].some(ext => name.endsWith(ext));
  }

  function bytesToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) binary += String.fromCharCode(...bytes.subarray(i, Math.min(i + chunk, bytes.length)));
    return btoa(binary);
  }

  async function uploadFiles(fileList) {
    if (!ws.activeProjectId) return;
    const files = [...(fileList || [])];
    if (!files.length) return;
    const parentFolderId = selectedFolderId();
    for (const file of files) {
      try {
        const buffer = await file.arrayBuffer();
        let textContent = null;
        if (isTextFile(file) && file.size <= 5 * 1024 * 1024) textContent = await file.text();
        await api(`/api/projects/${ws.activeProjectId}/workspace/files`, {
          method:'POST',
          body:JSON.stringify({
            name:file.name,
            mimeType:file.type || 'application/octet-stream',
            size:file.size,
            contentBase64:bytesToBase64(buffer),
            textContent,
            parentFolderId,
          }),
        });
      } catch (error) {
        alert(`${file.name}: ${error.message}`);
        break;
      }
    }
    ws.selected.clear();
    await refreshProject();
  }

  async function deleteSelected() {
    const refs = [...ws.selected.values()].filter(item => !item.trashId).map(item => ({type:item.type,id:item.id}));
    if (!refs.length || !ws.activeProjectId) return;
    if (!confirm(`${refs.length} item(s)을 Trash로 이동할까요?`)) return;
    try {
      await api(`/api/projects/${ws.activeProjectId}/workspace/trash`, {method:'POST', body:JSON.stringify({items:refs})});
      ws.selected.clear();
      await refreshProject();
      if (typeof window.renderProjectHome === 'function') window.renderProjectHome();
    } catch (error) { alert(error.message); }
  }

  async function restoreSelected() {
    const ids = [...ws.selected.values()].map(item => item.trashId).filter(Boolean);
    if (!ids.length || !ws.activeProjectId) return;
    try {
      await api(`/api/projects/${ws.activeProjectId}/workspace/restore`, {method:'POST', body:JSON.stringify({trashIds:ids})});
      ws.selected.clear();
      await refreshProject();
    } catch (error) { alert(error.message); }
  }

  async function purgeSelected() {
    const ids = [...ws.selected.values()].map(item => item.trashId).filter(Boolean);
    if (!ids.length || !ws.activeProjectId) return;
    if (!confirm(`${ids.length} item(s)을 영구 삭제할까요? 이 작업은 되돌릴 수 없습니다.`)) return;
    try {
      await api(`/api/projects/${ws.activeProjectId}/workspace/purge`, {method:'POST', body:JSON.stringify({trashIds:ids})});
      ws.selected.clear();
      await refreshProject();
    } catch (error) { alert(error.message); }
  }

  async function renameSelectedRow(row) {
    if (!['file','folder'].includes(row.dataset.type) || row.dataset.trashId) return;
    const old = row.querySelector('.ws-label')?.textContent || '';
    const name = prompt('Rename', old);
    if (!name?.trim() || name.trim() === old) return;
    try {
      await api(`/api/projects/${ws.activeProjectId}/workspace/rename`, {method:'POST', body:JSON.stringify({type:row.dataset.type,id:row.dataset.id,name:name.trim()})});
      await refreshProject();
    } catch (error) { alert(error.message); }
  }

  function setupDragSource(row) {
    row.addEventListener('dragstart', event => {
      const current = {type:row.dataset.type,id:row.dataset.id};
      const selectedGeneric = [...ws.selected.values()].filter(item => ['file','folder'].includes(item.type) && !item.trashId).map(item => ({type:item.type,id:item.id}));
      ws.dragItems = selectedGeneric.some(item => item.type === current.type && item.id === current.id) ? selectedGeneric : [current];
      event.dataTransfer.effectAllowed = 'move';
      event.dataTransfer.setData('text/plain', JSON.stringify(ws.dragItems));
      row.classList.add('dragging');
    });
    row.addEventListener('dragend', () => row.classList.remove('dragging'));
  }

  function setupFolderDrop(row, folderId) {
    for (const name of ['dragenter','dragover']) row.addEventListener(name, event => {
      if (!ws.dragItems.length) return;
      event.preventDefault(); event.stopPropagation(); row.classList.add('drop-target');
    });
    row.addEventListener('dragleave', () => row.classList.remove('drop-target'));
    row.addEventListener('drop', async event => {
      if (!ws.dragItems.length) return;
      event.preventDefault(); event.stopPropagation(); row.classList.remove('drop-target');
      try {
        await api(`/api/projects/${ws.activeProjectId}/workspace/move`, {method:'POST',body:JSON.stringify({items:ws.dragItems,parentFolderId:folderId})});
        ws.dragItems = [];
        await refreshProject();
      } catch (error) { alert(error.message); }
    });
  }

  async function openRow(row) {
    if (row.dataset.trashId) return;
    const type = row.dataset.type;
    const id = row.dataset.id;
    if (type === 'file') return showFile(id);
    if (type === 'cluster' && typeof window.renderCluster === 'function') return window.renderCluster(id);
    if (type === 'analysis' && typeof window.showAnalysis === 'function') return window.showAnalysis(id);
    if (type === 'proposal' && typeof window.renderProposal === 'function') return window.renderProposal(id);
  }

  async function showFile(fileId) {
    try {
      const file = await api(`/api/projects/${ws.activeProjectId}/files/${fileId}`);
      const main = document.getElementById('mainView');
      if (!main) return;
      main.innerHTML = `<div class="ws-file-view"><div class="breadcrumb">Projects › ${esc(ws.project.title)} › Files › ${esc(file.name)}</div><div class="section-head"><div><h1 style="margin:0">${esc(file.name)}</h1><div class="muted">${esc(file.mimeType || 'unknown type')} · ${esc(formatSize(file.size))} · SHA-256 ${esc(String(file.contentHash || '').slice(0,12))}…</div></div><button class="ghost" id="wsDownloadFile">Download original</button></div>${file.textContent !== null && file.textContent !== undefined ? `<pre class="ws-code-preview">${esc(file.textContent)}</pre>` : '<div class="panel" style="margin-top:18px">Binary file preview is not available, but the original bytes are stored unchanged.</div>'}</div>`;
      document.getElementById('wsDownloadFile').onclick = () => downloadStoredFile(file);
      setFlow('data');
    } catch (error) { alert(error.message); }
  }

  function downloadStoredFile(file) {
    const binary = atob(file.contentBase64 || '');
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], {type:file.mimeType || 'application/octet-stream'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = file.name || 'file'; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function loadModules() {
    try {
      const payload = await api('/api/module-workshop/modules');
      ws.modules = payload.modules || [];
    } catch (error) { ws.modules = []; }
    renderModules();
  }

  function renderModules() {
    const panel = document.getElementById('leesinRightModules');
    if (!panel) return;
    const q = ws.moduleQuery.trim().toLowerCase();
    const filtered = ws.modules.filter(module => !q || [module.title,module.description,module.question,module.entryFunction].some(value => String(value || '').toLowerCase().includes(q)));
    const favorites = filtered.filter(module => ws.favoriteModules.has(module.id));
    const cards = list => list.map(module => moduleCard(module)).join('') || '<div class="ws-empty" style="padding-left:8px">None</div>';
    panel.innerHTML = `<div class="ws-right-head"><h2>Modules</h2><button class="ws-icon-btn" id="wsNewModule">+ New</button></div><input class="ws-module-search" id="wsModuleSearch" placeholder="Search modules…" value="${esc(ws.moduleQuery)}"><div class="muted">지금은 로컬 검색입니다. 자연어/GPT discovery는 Registry 다음 단계에서 붙입니다.</div><div class="ws-right-section">Favorites</div>${cards(favorites)}<div class="ws-right-section">My Modules</div>${cards(filtered)}`;
    document.getElementById('wsModuleSearch').addEventListener('input', event => { ws.moduleQuery = event.target.value; renderModules(); const input=document.getElementById('wsModuleSearch'); input?.focus(); input?.setSelectionRange(input.value.length,input.value.length); });
    document.getElementById('wsNewModule').onclick = () => window.openModuleWorkshop?.();
    panel.querySelectorAll('[data-use-module]').forEach(button => button.onclick = () => window.openModuleWorkshop?.(button.dataset.useModule));
    panel.querySelectorAll('[data-star-module]').forEach(button => button.onclick = () => toggleFavorite(button.dataset.starModule));
  }

  function moduleCard(module) {
    const favorite = ws.favoriteModules.has(module.id);
    return `<div class="ws-module-card"><div style="display:flex;align-items:flex-start;gap:7px"><button class="ws-star" data-star-module="${esc(module.id)}" title="Favorite">${favorite ? '★' : '☆'}</button><div style="min-width:0"><div class="ws-module-name">${esc(module.title || module.entryFunction)}</div><div class="ws-module-author">by you · ${esc(module.entryFunction)} · v${esc(module.version || '0.1.0')}</div></div></div>${module.description ? `<div class="ws-module-desc">${esc(module.description)}</div>` : ''}<div class="ws-module-actions"><button class="ghost" data-use-module="${esc(module.id)}">Use</button></div></div>`;
  }

  function toggleFavorite(moduleId) {
    if (ws.favoriteModules.has(moduleId)) ws.favoriteModules.delete(moduleId); else ws.favoriteModules.add(moduleId);
    localStorage.setItem('leesin.favoriteModules', JSON.stringify([...ws.favoriteModules]));
    renderModules();
  }

  function hookLegacyProjectOpening() {
    const originalOpen = window.openProject;
    if (typeof originalOpen === 'function' && !originalOpen.__wsWrapped) {
      const wrapped = async function(projectId) {
        ws.activeProjectId = projectId;
        ws.selected.clear();
        const result = await originalOpen(projectId);
        await refreshProject(projectId);
        setFlow('data');
        return result;
      };
      wrapped.__wsWrapped = true;
      window.openProject = wrapped;
    }
    const originalTree = window.renderTree;
    if (typeof originalTree === 'function' && !originalTree.__wsWrapped) {
      const wrappedTree = function() {
        if (ws.activeProjectId) refreshProject(ws.activeProjectId);
        else originalTree();
      };
      wrappedTree.__wsWrapped = true;
      window.renderTree = wrappedTree;
    }
  }

  async function initializeProject() {
    try {
      const bootstrap = await api('/api/bootstrap');
      if (!ws.activeProjectId && bootstrap.projects?.length) {
        ws.activeProjectId = bootstrap.projects[0].id;
        await refreshProject(ws.activeProjectId);
      }
    } catch (error) { console.warn(error); }
  }

  function installObservers() {
    const main = document.getElementById('mainView');
    if (!main) return;
    const observer = new MutationObserver(() => {
      detectFlowFromMain();
      if (document.getElementById('moduleWorkshopBtn')) document.getElementById('moduleWorkshopBtn').style.display = 'none';
    });
    observer.observe(main, {childList:true,subtree:true,attributes:true,attributeFilter:['style']});
  }

  async function init() {
    installStyles();
    installLayout();
    hookLegacyProjectOpening();
    installObservers();
    await Promise.all([initializeProject(), loadModules()]);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
