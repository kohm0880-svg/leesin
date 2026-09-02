(() => {
  const LEFT_OFF_KEY = 'leesin.leftOff';
  const RIGHT_OFF_KEY = 'leesin.rightOff';

  function installPolishStyles() {
    if (document.getElementById('leesinUxPolishStyles')) return;
    const style = document.createElement('style');
    style.id = 'leesinUxPolishStyles';
    style.textContent = `
      :root{
        --line:#d5e0e2!important;
        --muted:#66767b!important;
        --bg:#eef3f3!important;
        --panel:#fbfdfc!important;
        --ink:#203136!important;
        --ok:#3f6f5b!important;
        --warn:#8a6936!important;
        --bad:#8c4e4e!important;
        --leesin-accent:#567c7e;
        --leesin-accent-dark:#3e6266;
        --leesin-accent-soft:#e1ecec;
        --leesin-blue-soft:#e7edf4;
        --leesin-lilac-soft:#efedf4;
        --leesin-shadow:0 10px 30px rgba(46,67,71,.055);
        --leesin-handle-width:18px;
      }
      body{background:linear-gradient(135deg,#eef4f3 0%,#f2f3f6 100%)!important;color:var(--ink)!important}
      .topbar{background:linear-gradient(100deg,#223239 0%,#2b3942 100%)!important;box-shadow:0 1px 0 #ffffff12,0 8px 24px #1e29331a}
      .topbar .brand{letter-spacing:.015em}.topbar .tag{color:#c8d4d6!important}
      .shell>aside:first-child{background:#f5f8f7!important;border-right:0!important}
      .shell>main{background:linear-gradient(180deg,#f2f6f5 0%,#f4f5f7 100%)!important}
      .ws-right{background:linear-gradient(180deg,#f6f8fa 0%,#f4f6f8 100%)!important;border-left:0!important}
      .panel{background:rgba(251,253,252,.96)!important;border:1px solid #d7e1e2!important;border-radius:16px!important;box-shadow:var(--leesin-shadow)!important}
      .project-card,.tree-row,.analysis-card,.proposal-card{border-color:#d7e1e2!important;border-radius:12px!important;box-shadow:0 3px 14px rgba(46,67,71,.035)}
      .project-card:hover,.tree-row:hover{border-color:#afc3c5!important;background:#f9fbfa!important}
      input,select,textarea{background:#fbfcfc!important;border-color:#cad8da!important;color:var(--ink)!important;transition:border-color .14s ease,box-shadow .14s ease,background .14s ease}
      input:focus,select:focus,textarea:focus{outline:none!important;border-color:#82a5a7!important;box-shadow:0 0 0 3px rgba(86,124,126,.13)!important;background:#fff!important}
      .primary{background:var(--leesin-accent)!important;border:1px solid var(--leesin-accent)!important;color:#fff!important;border-radius:10px!important;box-shadow:0 5px 14px rgba(62,98,102,.13);font-weight:700}
      .primary:hover{background:var(--leesin-accent-dark)!important;border-color:var(--leesin-accent-dark)!important}
      .ghost,.ws-icon-btn{background:#f9fbfa!important;border-color:#cfdcde!important;color:#3b5157!important;border-radius:9px!important}
      .ghost:hover,.ws-icon-btn:hover:not(:disabled){background:#edf4f3!important;border-color:#adc3c5!important}
      .muted{color:#6b797e!important}
      .breadcrumb{color:#708185!important}

      /* True pane boundaries: resizable line + collapse control lives on the line. */
      .shell{grid-template-columns:var(--leesin-left-width) var(--leesin-handle-width) minmax(0,1fr) var(--leesin-handle-width) var(--leesin-right-width)!important}
      .ws-resize{width:var(--leesin-handle-width)!important;background:transparent!important;overflow:visible!important;cursor:col-resize!important;z-index:12!important}
      .ws-resize::before{content:'';position:absolute;top:0;bottom:0;left:50%;width:1px;background:#cbd8da;transform:translateX(-50%);transition:background .14s ease}
      .ws-resize:hover::before,.ws-resize.dragging::before{background:#88a8aa}
      .ws-boundary-toggle{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);z-index:14;width:25px;height:52px;padding:0;border:1px solid #c3d1d3;border-radius:999px;background:#f8fbfa;color:#50686d;font:700 10px/1 system-ui,sans-serif;box-shadow:0 5px 16px rgba(46,67,71,.1);cursor:pointer;opacity:.82;transition:opacity .14s ease,background .14s ease,border-color .14s ease,transform .14s ease}
      .ws-boundary-toggle:hover{opacity:1;background:#e7f0ef;border-color:#98b4b6;transform:translate(-50%,-50%) scale(1.04)}
      body.ws-left-off .shell{grid-template-columns:0 var(--leesin-handle-width) minmax(0,1fr) var(--leesin-handle-width) var(--leesin-right-width)!important}
      body.ws-left-off .shell>aside:first-child{display:none!important}
      body.ws-left-off .ws-resize.left{display:block!important;grid-column:2!important}
      body.ws-right-off .shell{grid-template-columns:var(--leesin-left-width) var(--leesin-handle-width) minmax(0,1fr) var(--leesin-handle-width) 0!important}
      body.ws-right-off .ws-right{display:none!important}
      body.ws-right-off .ws-resize.right{display:block!important;grid-column:4!important}
      body.ws-left-off.ws-right-off .shell{grid-template-columns:0 var(--leesin-handle-width) minmax(0,1fr) var(--leesin-handle-width) 0!important}
      #wsProjectsToggle,#wsModulesToggle{display:none!important}

      /* Flow is a quiet progress rail, not a row of office buttons. */
      .ws-flow{justify-content:center!important;gap:7px!important;padding:13px 18px!important;background:rgba(247,250,249,.9)!important;border-bottom:1px solid #d5e0e2!important;box-shadow:0 5px 18px rgba(46,67,71,.035)!important}
      .ws-flow-step{display:inline-flex!important;align-items:center!important;gap:7px!important;padding:7px 9px!important;border:1px solid transparent!important;background:transparent!important;color:#87979b!important;border-radius:999px!important;font-weight:700!important;transition:background .14s ease,color .14s ease,border-color .14s ease!important}
      .ws-flow-step::before{content:'';width:7px;height:7px;border-radius:999px;border:1.5px solid #9caaad;background:transparent;flex:0 0 7px}
      .ws-flow-step.done{color:#536d70!important}.ws-flow-step.done::before{background:#91afb0;border-color:#91afb0}
      .ws-flow-step.active{background:var(--leesin-accent-soft)!important;color:#355c60!important;border-color:#c1d5d6!important;box-shadow:inset 0 0 0 1px #ffffff80!important}
      .ws-flow-step.active::before{background:var(--leesin-accent)!important;border-color:var(--leesin-accent)!important;box-shadow:0 0 0 3px rgba(86,124,126,.12)}
      .ws-flow-step.clickable:hover{background:#ebf2f1!important;color:#3f6266!important}
      .ws-flow-arrow{font-size:0!important;color:#afbec1!important}.ws-flow-arrow::after{content:'›';font-size:16px}

      /* Explorer: softer selection, cleaner density. */
      .ws-project-title{font-size:14px!important;color:#30474d!important;margin-top:14px!important}
      .ws-toolbar{padding-bottom:8px;border-bottom:1px solid #e0e7e8}
      .ws-section-head,.ws-folder-row,.ws-item-row{border-radius:8px!important;transition:background .1s ease,box-shadow .1s ease,color .1s ease}
      .ws-section-head:hover,.ws-folder-row:hover,.ws-item-row:hover{background:#eaf1f0!important}
      .ws-section-head{color:#41595f!important}.ws-folder-row{color:#3d555a!important}
      .ws-item-row.selected,.ws-folder-row.selected{background:#dcebea!important;outline:1px solid #a9c5c6!important;box-shadow:inset 3px 0 0 #6e9597!important}
      .ws-folder-row.drop-target,.ws-section-head.drop-target{background:#e2eee8!important;outline:1px solid #a8c8b5!important}
      .ws-meta{color:#91a0a4!important}
      .ws-empty{color:#94a2a5!important}
      .ws-code-preview,.mw-result{background:#223036!important;color:#e4eded!important;border:1px solid #31434a!important;box-shadow:0 8px 24px rgba(34,48,54,.09)!important}

      /* Module shelf: quieter cards, clear author, one primary action. */
      .ws-right h2{font-size:19px!important;color:#2d444a!important}
      .ws-module-search{border-radius:12px!important;padding:10px 12px!important;background:#fbfcfd!important}
      .ws-right-section{color:#65767b!important;letter-spacing:.07em!important;margin-top:16px!important}
      .ws-module-card{border:1px solid #d5dfe2!important;background:linear-gradient(145deg,#fbfdfc 0%,#f7f8fb 100%)!important;border-radius:14px!important;padding:12px!important;margin:8px 0!important;box-shadow:0 5px 18px rgba(54,70,78,.04)!important;transition:border-color .14s ease,box-shadow .14s ease!important}
      .ws-module-card:hover{border-color:#b7c9cb!important;box-shadow:0 8px 24px rgba(54,70,78,.07)!important}
      .ws-module-name{font-size:14px!important;color:#273c42!important}.ws-module-author{margin-top:2px!important;color:#7b8a8f!important}.ws-module-desc{color:#5e7076!important}
      .ws-module-actions .ghost{border:0!important;background:#e4eeee!important;color:#355c60!important;font-weight:750!important;padding:7px 12px!important}
      .ws-star{color:#967642!important;line-height:1!important}

      /* Workshop lives in the Core pane. Saved modules belong only to the right shelf. */
      .mw-layout{display:block!important;max-width:1040px;margin:18px auto 0!important}
      .mw-layout>aside{display:none!important}
      .mw-step{color:#334c51!important}.mw-num{background:#557b7d!important;box-shadow:0 3px 10px rgba(62,98,102,.14)}
      .mw-code,.mw-data{background:#fbfdfd!important;border-color:#cbd9db!important;border-radius:12px!important}
      .mw-warning{background:#f7f2e6!important;border-color:#e6d8b7!important;color:#79633e!important;border-radius:12px!important}
      .mw-ok{background:#eaf3ee!important;border-color:#c9dfd2!important;color:#416c59!important;border-radius:11px!important}
      .mw-error{background:#f7ebeb!important;border-color:#e6caca!important;color:#875050!important;border-radius:11px!important}
      .mw-file-zone{background:linear-gradient(145deg,#f7fbfa 0%,#f6f7fb 100%)!important;border-color:#b9cacc!important;border-radius:15px!important}
      .mw-file-zone:hover,.mw-file-zone.is-dragging{background:#edf5f4!important;border-color:#789b9d!important}
      .mw-chip{background:#edf3f2!important;border-color:#d0dedd!important;color:#466064!important}
      .mw-details{border-top-color:#dce5e6!important}

      @media(max-width:980px){
        :root{--leesin-handle-width:14px}
        .ws-boundary-toggle{width:22px;height:46px;font-size:9px}
        .ws-flow{justify-content:flex-start!important}
      }
      @media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
    `;
    document.head.appendChild(style);
  }

  function stripTopToggles() {
    const topbar = document.querySelector('.topbar');
    for (const id of ['wsProjectsToggle', 'wsModulesToggle']) {
      document.getElementById(id)?.remove();
    }
    if (!topbar) return;
    [...topbar.children].forEach(child => {
      if (child.tagName === 'DIV' && child.style.flex === '1' && !child.textContent.trim()) child.remove();
    });
  }

  function setPaneCollapsed(side, collapsed) {
    const className = side === 'left' ? 'ws-left-off' : 'ws-right-off';
    const key = side === 'left' ? LEFT_OFF_KEY : RIGHT_OFF_KEY;
    document.body.classList.toggle(className, collapsed);
    localStorage.setItem(key, collapsed ? '1' : '0');
    syncBoundaryButtons();
  }

  function syncBoundaryButtons() {
    const left = document.querySelector('.ws-boundary-toggle[data-side="left"]');
    const right = document.querySelector('.ws-boundary-toggle[data-side="right"]');
    const leftOff = document.body.classList.contains('ws-left-off');
    const rightOff = document.body.classList.contains('ws-right-off');
    if (left) {
      left.textContent = leftOff ? '>>' : '<<';
      left.title = leftOff ? 'Projects 열기' : 'Projects 접기';
      left.setAttribute('aria-label', left.title);
    }
    if (right) {
      right.textContent = rightOff ? '<<' : '>>';
      right.title = rightOff ? 'Modules 열기' : 'Modules 접기';
      right.setAttribute('aria-label', right.title);
    }
  }

  function addBoundaryButton(handle, side) {
    if (!handle || handle.querySelector(`.ws-boundary-toggle[data-side="${side}"]`)) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ws-boundary-toggle';
    button.dataset.side = side;
    button.addEventListener('pointerdown', event => event.stopPropagation());
    button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      const isOff = document.body.classList.contains(side === 'left' ? 'ws-left-off' : 'ws-right-off');
      setPaneCollapsed(side, !isOff);
    });
    handle.appendChild(button);
  }

  function ensureBoundaryControls() {
    addBoundaryButton(document.querySelector('.ws-resize.left'), 'left');
    addBoundaryButton(document.querySelector('.ws-resize.right'), 'right');
    syncBoundaryButtons();
  }

  function polishWorkshop() {
    const layout = document.querySelector('#mainView .mw-layout');
    if (!layout) return;
    const savedAside = layout.querySelector(':scope > aside');
    const importButton = savedAside?.querySelector('#mwPasteModuleBtn');
    const head = document.querySelector('#mainView .section-head');
    const backButton = document.getElementById('mwBackBtn');
    if (importButton && head && !document.getElementById('mwImportMoved')) {
      importButton.id = 'mwImportMoved';
      importButton.textContent = 'Import Module';
      importButton.title = 'Paste Module JSON';
      if (backButton) head.insertBefore(importButton, backButton);
      else head.appendChild(importButton);
    }
  }

  function polishModuleShelf() {
    const right = document.getElementById('leesinRightModules');
    if (!right) return;
    const search = document.getElementById('wsModuleSearch');
    if (search) search.placeholder = 'What do you want to do?';
    const newButton = document.getElementById('wsNewModule');
    if (newButton) {
      newButton.textContent = '+';
      newButton.title = 'New Module';
      newButton.setAttribute('aria-label', 'New Module');
    }
    const explanatory = right.querySelector(':scope > .muted');
    if (explanatory && !explanatory.dataset.polished) {
      explanatory.dataset.polished = '1';
      explanatory.textContent = '이름 · 설명 · Question에서 빠르게 찾습니다. 자연어 discovery는 Registry 단계에서 연결합니다.';
    }
  }

  function polish() {
    stripTopToggles();
    ensureBoundaryControls();
    polishWorkshop();
    polishModuleShelf();
  }

  function init() {
    installPolishStyles();
    if (localStorage.getItem(LEFT_OFF_KEY) === '1') document.body.classList.add('ws-left-off');
    if (localStorage.getItem(RIGHT_OFF_KEY) === '1') document.body.classList.add('ws-right-off');
    polish();
    const observer = new MutationObserver(polish);
    observer.observe(document.documentElement, {childList:true, subtree:true});
    window.addEventListener('storage', syncBoundaryButtons);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();