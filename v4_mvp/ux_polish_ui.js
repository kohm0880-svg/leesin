(() => {
  'use strict';

  const VERSION = '0.6.0';
  const STEP_ORDER = ['data', 'module', 'mapping', 'analysis', 'result', 'next'];
  const STEP_LABELS = {
    data: 'Data',
    module: 'Module',
    mapping: 'Mapping',
    analysis: 'Analysis',
    result: 'Result',
    next: 'Next',
  };
  const STORAGE = {
    core: 'leesin.coreState.v3',
    users: 'leesin.localUsers.v1',
    currentUser: 'leesin.currentUser.v1',
    moduleMeta: 'leesin.moduleMeta.v2',
    projectMeta: 'leesin.projectMeta.v1',
    favorites: 'leesin.favorites.v2',
    usage: 'leesin.moduleUsage.v1',
    leftOff: 'leesin.leftOff',
    rightOff: 'leesin.rightOff',
  };

  const PUBLIC_MODULES = [
    {
      id: 'builtin:prime_crossover',
      kind: 'builtin',
      title: 'Single Boundary',
      entryFunction: 'SingleBoundaryModule',
      questionId: 'prime_crossover',
      author: '@leesin-labs',
      version: '0.1.1',
      visibility: 'public',
      description: '순서가 있는 입력에서 두 상태의 우위가 한 번 전환되는 경계 범위를 좁힙니다.',
      exampleQuestions: [
        '두 알고리즘의 성능 우위가 바뀌는 입력 크기는 어디인가?',
        '온도를 올릴 때 상태가 바뀌는 경계는 어디인가?',
        '어느 부하부터 failure가 발생하는가?',
      ],
      requiredColumns: ['N', 'algorithm', 'runtime_ms'],
      inputs: ['ordered input', 'two alternatives', 'score'],
      outputs: ['boundary bracket', 'next observation'],
      assumptions: ['탐색 구간에서 우위는 최대 한 번만 전환됩니다.'],
      limits: ['현재 구현의 기본 adapter는 prime benchmark에만 연결됩니다.'],
      tags: ['boundary', 'threshold', 'crossover', 'transition', '경계', '임계점'],
      accent: 'blue',
    },
    {
      id: 'registry:summary',
      kind: 'registry',
      title: 'Descriptive Summary',
      entryFunction: 'describe',
      author: '@open-stats',
      version: '1.0.0',
      visibility: 'public',
      description: '숫자 열 하나의 개수, 평균, 중앙값, 최솟값과 최댓값을 한 번에 계산합니다.',
      exampleQuestions: ['이 측정값들의 대표적인 크기와 범위는?', '데이터를 빠르게 요약하고 싶다.'],
      inputs: ['values'],
      outputs: ['count', 'mean', 'median', 'min', 'max'],
      assumptions: ['비어 있지 않은 수치형 열이 필요합니다.'],
      limits: ['분포 모형이나 인과관계를 추론하지 않습니다.'],
      tags: ['summary', 'mean', 'median', '통계', '요약'],
      accent: 'green',
      code: `import statistics\ndef describe(values):\n    clean = [float(v) for v in values if v is not None]\n    if not clean:\n        raise ValueError("No numeric values.")\n    return {\n        "count": len(clean),\n        "mean": statistics.mean(clean),\n        "median": statistics.median(clean),\n        "min": min(clean),\n        "max": max(clean),\n    }`,
      inputContract: [{name: 'values', kind: 'positional_or_keyword', required: true, default: null, annotation: null}],
    },
    {
      id: 'registry:pearson',
      kind: 'registry',
      title: 'Pearson Correlation',
      entryFunction: 'pearson',
      author: '@open-stats',
      version: '1.0.0',
      visibility: 'public',
      description: '길이가 같은 두 수치 열의 Pearson 상관계수를 계산합니다.',
      exampleQuestions: ['두 변수는 선형적으로 함께 변하는가?', 'x와 y의 상관 정도를 보고 싶다.'],
      inputs: ['x', 'y'],
      outputs: ['correlation coefficient'],
      assumptions: ['두 열의 길이가 같아야 합니다.', '수치형 값이 필요합니다.'],
      limits: ['상관관계는 인과관계를 의미하지 않습니다.'],
      tags: ['correlation', 'pearson', 'relationship', '상관', '관계'],
      accent: 'violet',
      code: `import math\ndef pearson(x, y):\n    a = [float(v) for v in x if v is not None]\n    b = [float(v) for v in y if v is not None]\n    if len(a) != len(b) or len(a) < 2:\n        raise ValueError("x and y need the same length of at least 2.")\n    ax = sum(a) / len(a)\n    by = sum(b) / len(b)\n    num = sum((u-ax)*(v-by) for u, v in zip(a, b))\n    den = math.sqrt(sum((u-ax)**2 for u in a) * sum((v-by)**2 for v in b))\n    if den == 0:\n        raise ValueError("Correlation is undefined for a constant column.")\n    return num / den`,
      inputContract: [
        {name: 'x', kind: 'positional_or_keyword', required: true, default: null, annotation: null},
        {name: 'y', kind: 'positional_or_keyword', required: true, default: null, annotation: null},
      ],
    },
    {
      id: 'registry:zscore',
      kind: 'registry',
      title: 'Z-score Flags',
      entryFunction: 'zscore_flags',
      author: '@data-notes',
      version: '0.8.0',
      visibility: 'public',
      description: '한 수치 열에서 지정한 절대 z-score 이상인 값의 위치와 값을 반환합니다.',
      exampleQuestions: ['평균에서 유난히 멀리 떨어진 값은 무엇인가?', '간단한 z-score 기준으로 후보를 표시하고 싶다.'],
      inputs: ['values', 'threshold'],
      outputs: ['flagged rows'],
      assumptions: ['평균과 표준편차를 사용하는 기준을 받아들여야 합니다.'],
      limits: ['정규성이나 이상치의 원인을 자동으로 판단하지 않습니다.'],
      tags: ['outlier', 'zscore', 'anomaly', '이상치', '후보'],
      accent: 'coral',
      code: `import statistics\ndef zscore_flags(values, threshold=3.0):\n    clean = [float(v) for v in values if v is not None]\n    if len(clean) < 2:\n        raise ValueError("At least two values are required.")\n    mean = statistics.mean(clean)\n    sd = statistics.stdev(clean)\n    if sd == 0:\n        return []\n    return [\n        {"index": i, "value": value, "z": (value-mean)/sd}\n        for i, value in enumerate(clean)\n        if abs((value-mean)/sd) >= float(threshold)\n    ]`,
      inputContract: [
        {name: 'values', kind: 'positional_or_keyword', required: true, default: null, annotation: null},
        {name: 'threshold', kind: 'positional_or_keyword', required: false, default: '3.0', annotation: null},
      ],
    },
  ];

  const app = {
    activeProjectId: null,
    project: null,
    workspace: null,
    savedModules: [],
    modules: [],
    moduleTab: 'browse',
    search: '',
    centerMode: 'core',
    renderToken: 0,
    observerScheduled: false,
    polishing: false,
    legacy: {},
  };

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
  }

  function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function loadJson(key, fallback) {
    try {
      const parsed = JSON.parse(localStorage.getItem(key) || 'null');
      return parsed === null ? fallback : parsed;
    } catch (_) {
      return fallback;
    }
  }

  function saveJson(key, value) {
    localStorage.setItem(key, JSON.stringify(value));
  }

  async function requestJson(path, options = {}) {
    const response = await fetch(path, {
      headers: {'Content-Type': 'application/json', ...(options.headers || {})},
      ...options,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    return data;
  }

  function nowIso() {
    return new Date().toISOString();
  }

  function randomId(prefix) {
    return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
  }

  function currentUser() {
    const id = localStorage.getItem(STORAGE.currentUser);
    if (!id) return null;
    const users = loadJson(STORAGE.users, {});
    return users[id] || null;
  }

  function favoriteKey() {
    return `${STORAGE.favorites}:${currentUser()?.id || 'guest'}`;
  }

  function favorites() {
    return new Set(loadJson(favoriteKey(), []));
  }

  function setFavorites(values) {
    saveJson(favoriteKey(), [...values]);
  }

  function moduleMetaStore() {
    return loadJson(STORAGE.moduleMeta, {});
  }

  function updateModuleMeta(id, patch) {
    const store = moduleMetaStore();
    store[id] = {...(store[id] || {}), ...patch, updatedAt: nowIso()};
    saveJson(STORAGE.moduleMeta, store);
  }

  function projectMetaStore() {
    return loadJson(STORAGE.projectMeta, {});
  }

  function ensureProjectOwner(projectId) {
    const user = currentUser();
    if (!projectId || !user) return;
    const store = projectMetaStore();
    if (!store[projectId]?.ownerId) {
      store[projectId] = {...(store[projectId] || {}), ownerId: user.id, owner: `@${user.username}`, visibility: 'private'};
      saveJson(STORAGE.projectMeta, store);
    }
  }

  function getProjectOwner(projectId) {
    const meta = projectMetaStore()[projectId] || {};
    return meta.owner || (currentUser() ? `@${currentUser().username}` : '@local');
  }

  function normalizeSavedModule(item) {
    const meta = moduleMetaStore()[item.id] || {};
    const user = currentUser();
    return {
      ...item,
      kind: 'saved',
      author: meta.author || (meta.ownerId === user?.id ? `@${user.username}` : meta.author) || '@local',
      ownerId: meta.ownerId || null,
      visibility: meta.visibility || 'private',
      tags: Array.isArray(meta.tags) ? meta.tags : [],
      exampleQuestions: Array.isArray(meta.exampleQuestions) ? meta.exampleQuestions : (item.question ? [item.question] : []),
      outputs: Array.isArray(meta.outputs) ? meta.outputs : ['function result'],
      inputs: (item.inputContract || []).map(input => input.name),
      accent: meta.accent || accentFromString(item.id),
    };
  }

  function accentFromString(value) {
    const accents = ['blue', 'violet', 'green', 'coral', 'amber', 'rose'];
    let hash = 0;
    for (const ch of String(value || '')) hash = ((hash << 5) - hash + ch.charCodeAt(0)) | 0;
    return accents[Math.abs(hash) % accents.length];
  }

  function refreshModuleList() {
    app.modules = [
      ...PUBLIC_MODULES,
      ...app.savedModules.map(normalizeSavedModule),
    ];
  }

  async function loadModules() {
    try {
      const payload = await requestJson('/api/module-workshop/modules');
      app.savedModules = payload.modules || [];
    } catch (_) {
      app.savedModules = [];
    }
    refreshModuleList();
    renderModuleShelf();
  }

  function coreStore() {
    return loadJson(STORAGE.core, {});
  }

  function newRun(projectId, carry = null, cycle = 1) {
    return {
      id: randomId('run'),
      projectId,
      cycle,
      stage: 'data',
      question: carry?.question || '',
      dataRefs: carry?.dataRefs ? deepClone(carry.dataRefs) : [],
      dataColumns: [],
      module: carry?.module ? deepClone(carry.module) : null,
      mapping: {},
      prepared: null,
      analysisId: null,
      analysis: null,
      result: null,
      proposal: null,
      createdAt: nowIso(),
      updatedAt: nowIso(),
    };
  }

  function getProjectCore(projectId) {
    const store = coreStore();
    if (!store[projectId]) {
      store[projectId] = {
        activeRun: newRun(projectId),
        completedRuns: [],
        discarded: [],
      };
      saveJson(STORAGE.core, store);
    }
    const state = store[projectId];
    if (!state.activeRun) state.activeRun = newRun(projectId);
    if (!Array.isArray(state.completedRuns)) state.completedRuns = [];
    if (!Array.isArray(state.discarded)) state.discarded = [];
    return state;
  }

  function saveProjectCore(projectId, state) {
    const store = coreStore();
    state.activeRun.updatedAt = nowIso();
    store[projectId] = state;
    saveJson(STORAGE.core, store);
  }

  function activeCore() {
    if (!app.activeProjectId) return null;
    return getProjectCore(app.activeProjectId);
  }

  function activeRun() {
    return activeCore()?.activeRun || null;
  }

  function saveRun(run) {
    const state = activeCore();
    if (!state) return;
    state.activeRun = run;
    saveProjectCore(app.activeProjectId, state);
  }

  function stageIndex(stage) {
    return Math.max(0, STEP_ORDER.indexOf(stage));
  }

  function clearAfterStage(run, target) {
    const next = deepClone(run);
    if (stageIndex(target) <= stageIndex('data')) {
      next.module = null;
      next.mapping = {};
      next.prepared = null;
      next.analysisId = null;
      next.analysis = null;
      next.result = null;
      next.proposal = null;
    } else if (stageIndex(target) <= stageIndex('module')) {
      next.mapping = {};
      next.prepared = null;
      next.analysisId = null;
      next.analysis = null;
      next.result = null;
      next.proposal = null;
    } else if (stageIndex(target) <= stageIndex('mapping')) {
      next.analysisId = null;
      next.analysis = null;
      next.result = null;
      next.proposal = null;
    } else if (stageIndex(target) <= stageIndex('analysis')) {
      next.analysisId = null;
      next.analysis = null;
      next.result = null;
      next.proposal = null;
    } else if (stageIndex(target) <= stageIndex('result')) {
      next.proposal = null;
    }
    next.stage = target;
    return next;
  }

  async function moveBackendDownstreamToTrash(run, target) {
    const refs = [];
    if (run.analysisId && stageIndex(target) <= stageIndex('analysis')) {
      refs.push({type: 'analysis', id: run.analysisId});
    } else if (run.proposal?.id && stageIndex(target) <= stageIndex('result')) {
      refs.push({type: 'proposal', id: run.proposal.id});
    }
    if (!refs.length) return [];
    try {
      const result = await requestJson(`/api/projects/${app.activeProjectId}/workspace/trash`, {
        method: 'POST',
        body: JSON.stringify({items: refs}),
      });
      return (result.trashed || []).map(item => item.trashId).filter(Boolean);
    } catch (error) {
      throw new Error(`이후 Analysis를 Trash로 옮기지 못했습니다: ${error.message}`);
    }
  }

  function downstreamLabels(run, target) {
    const labels = [];
    if (stageIndex(target) < stageIndex('module') && run.module) labels.push('Module binding');
    if (stageIndex(target) < stageIndex('mapping') && Object.keys(run.mapping || {}).length) labels.push('Mapping');
    if (stageIndex(target) < stageIndex('analysis') && (run.analysis || run.analysisId)) labels.push('Analysis');
    if (stageIndex(target) < stageIndex('result') && run.result) labels.push('Result');
    if (stageIndex(target) < stageIndex('next') && run.proposal) labels.push('Next proposal');
    return labels;
  }

  async function rollbackTo(target) {
    const state = activeCore();
    const run = state?.activeRun;
    if (!state || !run) return;
    if (stageIndex(target) >= stageIndex(run.stage)) return;
    const labels = downstreamLabels(run, target);
    const message = labels.length
      ? `${STEP_LABELS[target]} 단계로 돌아가면 이후 ${labels.join(', ')}이 현재 Run에서 제거됩니다. 계속할까요?`
      : `${STEP_LABELS[target]} 단계로 돌아갈까요?`;
    if (!window.confirm(message)) return;

    const snapshot = deepClone(run);
    let backendTrashIds = [];
    try {
      backendTrashIds = await moveBackendDownstreamToTrash(run, target);
    } catch (error) {
      alert(error.message);
      return;
    }

    state.discarded.unshift({
      id: randomId('discard'),
      at: nowIso(),
      target,
      snapshot,
      backendTrashIds,
    });
    state.discarded = state.discarded.slice(0, 20);
    state.activeRun = clearAfterStage(run, target);
    saveProjectCore(app.activeProjectId, state);
    app.centerMode = 'core';
    await refreshContext({render: false});
    renderCore();
    showUndoToast(`${STEP_LABELS[target]} 단계로 돌아왔습니다.`, state.discarded[0].id);
  }

  async function undoDiscard(discardId) {
    const state = activeCore();
    if (!state) return;
    const index = state.discarded.findIndex(item => item.id === discardId);
    if (index < 0) return;
    const item = state.discarded[index];
    if (item.backendTrashIds?.length) {
      try {
        await requestJson(`/api/projects/${app.activeProjectId}/workspace/restore`, {
          method: 'POST',
          body: JSON.stringify({trashIds: item.backendTrashIds}),
        });
      } catch (error) {
        alert(`복원하지 못했습니다: ${error.message}`);
        return;
      }
    }
    state.activeRun = item.snapshot;
    state.discarded.splice(index, 1);
    saveProjectCore(app.activeProjectId, state);
    await refreshContext({render: false});
    renderCore();
    showToast('이전 상태를 복원했습니다.');
  }

  function archiveAndStartCycle({carryModule = true, selectAllClusters = false} = {}) {
    const state = activeCore();
    const run = state?.activeRun;
    if (!state || !run) return;
    state.completedRuns.unshift({...deepClone(run), completedAt: nowIso()});
    state.completedRuns = state.completedRuns.slice(0, 50);
    const carry = {
      question: run.question,
      module: carryModule ? run.module : null,
      dataRefs: selectAllClusters
        ? (app.project?.clusters || []).map(item => ({kind: 'cluster', id: item.id, name: item.name || item.filename}))
        : [],
    };
    state.activeRun = newRun(app.activeProjectId, carry, Number(run.cycle || 1) + 1);
    saveProjectCore(app.activeProjectId, state);
    app.centerMode = 'core';
    void computeDataColumns(state.activeRun).then(() => renderCore());
  }

  function showToast(message, actionLabel = '', action = null) {
    let box = document.getElementById('productToast');
    if (!box) {
      box = document.createElement('div');
      box.id = 'productToast';
      document.body.appendChild(box);
    }
    box.innerHTML = `<span>${esc(message)}</span>${actionLabel ? `<button type="button">${esc(actionLabel)}</button>` : ''}`;
    box.classList.add('show');
    const button = box.querySelector('button');
    if (button && action) button.onclick = action;
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => box.classList.remove('show'), 6500);
  }

  function showUndoToast(message, discardId) {
    showToast(message, 'Undo', () => undoDiscard(discardId));
  }

  function installStyles() {
    if (document.getElementById('leesinProductStyles')) return;
    const style = document.createElement('style');
    style.id = 'leesinProductStyles';
    style.textContent = `
      :root{
        --line:#d7e0e7!important;
        --muted:#687681!important;
        --bg:#eef3f5!important;
        --panel:#fbfcfd!important;
        --ink:#24333a!important;
        --ok:#397663!important;
        --warn:#99713a!important;
        --bad:#9a555a!important;
        --p-blue:#5f83b8;
        --p-blue-soft:#e5edf8;
        --p-violet:#8172b6;
        --p-violet-soft:#eeeaf8;
        --p-green:#579078;
        --p-green-soft:#e4f1eb;
        --p-coral:#c8756d;
        --p-coral-soft:#f7e8e5;
        --p-amber:#bd8849;
        --p-amber-soft:#f6eddc;
        --p-rose:#af7192;
        --p-rose-soft:#f4e8ef;
        --p-teal:#527f83;
        --p-teal-soft:#e1eeee;
        --product-shadow:0 14px 36px rgba(44,62,72,.07);
        --leesin-handle-width:18px;
      }
      *{scrollbar-color:#b7c7cf transparent;scrollbar-width:thin}
      body{background:linear-gradient(135deg,#edf5f3 0%,#f1f2f7 48%,#f6f0f2 100%)!important;color:var(--ink)!important}
      .topbar{background:linear-gradient(102deg,#24363d 0%,#31434e 55%,#3d3b52 100%)!important;box-shadow:0 8px 28px rgba(30,44,52,.18)!important}
      .topbar .brand{letter-spacing:.02em}.topbar .tag{color:#ced9de!important}
      .shell>aside:first-child{background:linear-gradient(180deg,#f7faf9 0%,#f4f6f8 100%)!important;border-right:0!important}
      .shell>main{background:linear-gradient(180deg,rgba(243,248,247,.9) 0%,rgba(246,245,249,.96) 100%)!important}
      .ws-right{background:linear-gradient(180deg,#f8f8fb 0%,#f4f7f8 100%)!important;border-left:0!important}
      .panel{background:rgba(252,253,253,.96)!important;border:1px solid #d9e1e6!important;border-radius:17px!important;box-shadow:var(--product-shadow)!important}
      input,select,textarea{background:#fcfdfd!important;border-color:#ccd8df!important;color:var(--ink)!important;border-radius:11px!important;transition:.14s ease!important}
      input:focus,select:focus,textarea:focus{outline:none!important;border-color:#779aa6!important;box-shadow:0 0 0 3px rgba(95,131,184,.12)!important;background:#fff!important}
      .primary{background:linear-gradient(135deg,#557e83,#5f78a0)!important;border:0!important;color:#fff!important;border-radius:11px!important;font-weight:750!important;box-shadow:0 7px 17px rgba(63,94,111,.16)!important}
      .primary:hover{filter:brightness(.96)}
      .ghost,.ws-icon-btn{background:#f9fbfb!important;border-color:#cfdae1!important;color:#40545e!important;border-radius:10px!important}
      .ghost:hover,.ws-icon-btn:hover:not(:disabled){background:#eaf2f3!important;border-color:#abc0c8!important}
      .muted{color:#6c7a84!important}.breadcrumb{color:#71828d!important}

      .shell{grid-template-columns:var(--leesin-left-width) var(--leesin-handle-width) minmax(0,1fr) var(--leesin-handle-width) var(--leesin-right-width)!important}
      .ws-resize{width:var(--leesin-handle-width)!important;background:transparent!important;overflow:visible!important;cursor:col-resize!important;z-index:12!important;position:relative!important}
      .ws-resize::before{content:'';position:absolute;inset:0 auto 0 50%;width:1px;background:#c8d5dc;transform:translateX(-50%)}
      .ws-resize:hover::before,.ws-resize.dragging::before{background:#779aa6}
      .ws-boundary-toggle{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);z-index:14;width:25px;height:54px;padding:0;border:1px solid #c1d0d6;border-radius:999px;background:rgba(250,252,252,.96);color:#4d6872;font:750 10px/1 system-ui,sans-serif;box-shadow:0 6px 18px rgba(45,67,78,.12);cursor:pointer;opacity:.88}
      .ws-boundary-toggle:hover{opacity:1;background:#e7f0f1;border-color:#91adb7}
      body.ws-left-off .shell{grid-template-columns:0 var(--leesin-handle-width) minmax(0,1fr) var(--leesin-handle-width) var(--leesin-right-width)!important}
      body.ws-left-off .shell>aside:first-child{display:none!important}
      body.ws-left-off .ws-resize.left{display:block!important;grid-column:2!important}
      body.ws-right-off .shell{grid-template-columns:var(--leesin-left-width) var(--leesin-handle-width) minmax(0,1fr) var(--leesin-handle-width) 0!important}
      body.ws-right-off .ws-right{display:none!important}
      body.ws-right-off .ws-resize.right{display:block!important;grid-column:4!important}
      body.ws-left-off.ws-right-off .shell{grid-template-columns:0 var(--leesin-handle-width) minmax(0,1fr) var(--leesin-handle-width) 0!important}
      #wsProjectsToggle,#wsModulesToggle,#moduleWorkshopBtn{display:none!important}

      .ws-flow{justify-content:center!important;gap:6px!important;padding:12px 18px!important;background:rgba(249,251,251,.88)!important;backdrop-filter:blur(12px)!important;border-bottom:1px solid #d7e0e6!important;box-shadow:0 6px 20px rgba(48,66,76,.04)!important}
      .product-flow-step{display:inline-flex;align-items:center;gap:7px;border:1px solid transparent;background:transparent;color:#8a979f;font-size:12px;font-weight:750;padding:7px 10px;border-radius:999px;white-space:nowrap}
      .product-flow-step .dot{width:8px;height:8px;border-radius:50%;border:1.5px solid currentColor;background:transparent}
      .product-flow-step.reached{cursor:pointer;color:#596d76}.product-flow-step.reached:hover{background:#edf2f4}
      .product-flow-step.active[data-step="data"]{background:var(--p-blue-soft);color:#466d9e;border-color:#c7d7eb}
      .product-flow-step.active[data-step="module"]{background:var(--p-violet-soft);color:#68599b;border-color:#d9d1ed}
      .product-flow-step.active[data-step="mapping"]{background:var(--p-amber-soft);color:#90652f;border-color:#ead6b4}
      .product-flow-step.active[data-step="analysis"]{background:var(--p-coral-soft);color:#a25b55;border-color:#ebc9c4}
      .product-flow-step.active[data-step="result"]{background:var(--p-green-soft);color:#3f725e;border-color:#c6dfd3}
      .product-flow-step.active[data-step="next"]{background:var(--p-rose-soft);color:#8f5575;border-color:#e4cad9}
      .product-flow-step.active .dot,.product-flow-step.done .dot{background:currentColor}
      .product-flow-arrow{color:#b7c1c7;font-size:16px}

      .ws-project-title{font-size:14px!important;color:#30474f!important}
      .ws-toolbar{padding-bottom:8px;border-bottom:1px solid #e1e7eb}
      .ws-section-head,.ws-folder-row,.ws-item-row{border-radius:8px!important;transition:.1s ease!important}
      .ws-section-head:hover,.ws-folder-row:hover,.ws-item-row:hover{background:#eaf1f2!important}
      .ws-section:nth-of-type(2)>.ws-section-head{color:#4f75a2!important}
      .ws-section:nth-of-type(3)>.ws-section-head{color:#755f9f!important}
      .ws-section:nth-of-type(4)>.ws-section-head{color:#a36b3f!important}
      .ws-section:nth-of-type(5)>.ws-section-head{color:#9a607f!important}
      .ws-item-row.selected,.ws-folder-row.selected{background:#dfecef!important;outline:1px solid #acc6ce!important;box-shadow:inset 3px 0 0 #668e99!important}
      .ws-folder-row.drop-target,.ws-section-head.drop-target{background:#e4f0e9!important;outline:1px solid #a9cbb7!important}
      .ws-code-preview,.mw-result{background:#25343d!important;color:#e5eef0!important;border:1px solid #344852!important;box-shadow:0 10px 26px rgba(36,51,60,.11)!important}

      .core-shell{max-width:1120px;margin:0 auto;padding:24px 26px 48px}
      .core-header{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;margin-bottom:18px}
      .core-eyebrow{font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#768791}
      .core-title{font-size:26px;line-height:1.18;margin:5px 0 4px;color:#263941}
      .core-meta{font-size:13px;color:#71808a}
      .core-header-actions{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
      .core-stage-card{background:rgba(252,253,253,.97);border:1px solid #d8e1e6;border-radius:19px;box-shadow:var(--product-shadow);overflow:hidden}
      .core-stage-accent{height:5px;background:linear-gradient(90deg,var(--p-blue),var(--p-violet),var(--p-coral),var(--p-green))}
      .core-stage-body{padding:22px}
      .core-stage-head{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;margin-bottom:16px}
      .core-stage-head h2{margin:0;font-size:21px}.core-stage-head p{margin:5px 0 0;color:#6b7a84;font-size:13px}
      .core-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(280px,.42fr);gap:16px}
      .core-subcard{border:1px solid #dce4e8;border-radius:15px;background:#fafcfc;padding:15px}
      .core-subcard h3{margin:0 0 9px;font-size:15px}
      .core-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:14px}
      .core-next-row{display:flex;justify-content:flex-end;gap:8px;margin-top:18px}
      .core-data-list{display:grid;gap:8px;max-height:440px;overflow:auto;padding-right:3px}
      .core-data-item{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:10px;align-items:center;padding:11px 12px;border:1px solid #d9e2e7;border-radius:13px;background:#fff;transition:.12s ease}
      .core-data-item:hover{border-color:#b7c9d3;box-shadow:0 5px 16px rgba(58,76,87,.05)}
      .core-data-item.selected{background:linear-gradient(135deg,#edf4fb,#f5f0fa);border-color:#b9cbe2}
      .core-data-name{font-weight:750;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.core-data-meta{font-size:12px;color:#7a8891;margin-top:2px}
      .core-mini{border:1px solid #d2dde3;background:#f9fbfc;border-radius:9px;padding:6px 9px;font-size:12px;color:#48606b}
      .core-mini:hover{background:#edf3f5}
      .core-selection-summary{display:flex;gap:6px;flex-wrap:wrap}.core-chip{display:inline-flex;align-items:center;gap:5px;background:#eef3f6;border:1px solid #d7e1e7;color:#536771;border-radius:999px;padding:5px 8px;font-size:12px}
      .module-slot{min-height:150px;border:1.5px dashed #b8c8d1;border-radius:17px;background:linear-gradient(140deg,#eef4fa 0%,#f3eff8 52%,#fbf4ef 100%);display:flex;align-items:center;justify-content:center;padding:20px;text-align:center;transition:.14s ease}
      .module-slot.dragover{border-color:#6e7fb2;box-shadow:0 0 0 4px rgba(129,114,182,.12);transform:translateY(-1px)}
      .module-slot.attached{border-style:solid;justify-content:flex-start;text-align:left;background:linear-gradient(140deg,#f5f8fc,#f8f5fb)}
      .module-slot-title{font-size:18px;font-weight:850}.module-slot-author{font-size:12px;color:#7a8790;margin-top:3px}.module-slot-desc{font-size:13px;color:#5d6d76;margin-top:8px;line-height:1.45}
      .module-badge{display:inline-flex;align-items:center;border-radius:999px;padding:4px 7px;font-size:11px;font-weight:800;margin-right:5px}
      .badge-compatible{background:#e2f0e8;color:#3e755e}.badge-mapping{background:#f5ecd9;color:#8f662f}.badge-public{background:#e8edf8;color:#536e9d}.badge-private{background:#efedf3;color:#6c607d}
      .mapping-row{display:grid;grid-template-columns:minmax(130px,.45fr) 28px minmax(190px,1fr);gap:9px;align-items:center;padding:9px 0;border-bottom:1px solid #e7ecef}.mapping-row:last-child{border-bottom:0}
      .mapping-arrow{color:#a1afb7;text-align:center}
      .analysis-running{padding:46px 20px;text-align:center}.analysis-spinner{width:38px;height:38px;margin:0 auto 15px;border-radius:50%;border:4px solid #dde6ea;border-top-color:var(--p-coral);animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
      .result-status{display:inline-flex;border-radius:999px;padding:5px 9px;font-size:11px;font-weight:850;letter-spacing:.04em}.result-ok{background:var(--p-green-soft);color:#3e735e}.result-stop{background:var(--p-coral-soft);color:#9b5752}.result-info{background:var(--p-amber-soft);color:#8a6333}
      .result-summary{font-size:19px;line-height:1.45;font-weight:760;margin:13px 0}.result-json{white-space:pre-wrap;word-break:break-word;background:#26353d;color:#e7eff0;border-radius:13px;padding:15px;max-height:430px;overflow:auto;font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
      .result-columns{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}.result-list{margin:0;padding-left:18px;color:#556872;font-size:13px}.result-list li{margin:5px 0}
      .core-question{width:100%;margin-top:6px}.core-note{font-size:12px;color:#74838c;margin-top:5px}
      .run-history{display:grid;gap:8px;max-height:420px;overflow:auto}.run-history-item{border:1px solid #dce4e8;border-radius:12px;padding:11px;background:#fafcfc}.run-history-item strong{display:block;margin-bottom:3px}

      .ws-right{padding:14px!important}.module-shelf-head{display:flex;align-items:center;justify-content:space-between;gap:8px}.module-shelf-head h2{margin:0;color:#30444d}.module-shelf-actions{display:flex;gap:6px}
      .module-search-wrap{position:relative;margin-top:12px}.module-search-wrap input{width:100%;padding:11px 38px 11px 12px!important;background:#fbfcfd!important}.module-search-icon{position:absolute;right:12px;top:50%;transform:translateY(-50%);color:#87949c}
      .module-tabs{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin:10px 0 12px;padding:4px;background:#e9eef2;border-radius:12px}.module-tab{border:0;background:transparent;border-radius:9px;padding:7px 5px;font-size:11px;font-weight:800;color:#6c7b84}.module-tab.active{background:#fff;color:#3f5661;box-shadow:0 3px 10px rgba(48,68,78,.08)}
      .module-search-note{font-size:11px;color:#7b8992;margin:0 2px 10px}.module-results{display:grid;gap:9px}
      .product-module-card{position:relative;border:1px solid #d8e1e7;border-radius:15px;background:linear-gradient(145deg,#fcfdfd,#f7f9fb);padding:12px;box-shadow:0 5px 18px rgba(52,70,80,.045);cursor:grab;overflow:hidden}
      .product-module-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--card-accent,#6e8ca2)}
      .product-module-card:hover{border-color:#b9c9d2;box-shadow:0 9px 23px rgba(52,70,80,.075)}.product-module-card:active{cursor:grabbing}
      .product-module-top{display:flex;align-items:flex-start;justify-content:space-between;gap:8px}.product-module-title{font-size:14px;font-weight:850;color:#2f424b}.product-module-author{font-size:11px;color:#7a8992;margin-top:2px}.product-module-description{font-size:12px;color:#5d6f79;line-height:1.42;margin:8px 0}.product-module-tags{display:flex;gap:4px;flex-wrap:wrap}.product-module-tag{font-size:10px;color:#667680;background:#edf2f5;border-radius:999px;padding:3px 6px}.product-module-actions{display:flex;gap:6px;align-items:center;margin-top:10px}.product-module-actions .use{flex:1;border:0;border-radius:9px;padding:7px;background:#e6eeee;color:#3d6468;font-weight:800}.product-module-actions .more,.product-module-star{border:0;background:transparent;padding:4px 6px;color:#6f7f88}.product-module-star{font-size:17px;color:#a17a43}
      .compat-text{font-size:10px;font-weight:800;margin-top:7px;color:#5c806e}.compat-text.mapping{color:#956d36}
      .empty-shelf{padding:18px 10px;border:1px dashed #cdd8de;border-radius:13px;text-align:center;color:#829099;font-size:12px}

      .data-lens{border:1px solid #d8e1e6;border-radius:16px;background:#fbfdfd;overflow:hidden;box-shadow:0 9px 26px rgba(48,66,76,.055)}
      .data-lens-head{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:12px 13px;border-bottom:1px solid #e0e7eb;background:linear-gradient(90deg,#f2f6fb,#f6f1f8,#fbf5ee)}
      .data-lens-title{font-weight:850;color:#334851}.data-lens-meta{font-size:11px;color:#7c8a92;margin-top:2px}.lens-toggle{display:flex;gap:4px;padding:3px;background:#e7edf1;border-radius:10px}.lens-toggle button{border:0;background:transparent;border-radius:8px;padding:6px 8px;font-size:11px;font-weight:800;color:#667780}.lens-toggle button.active{background:#fff;color:#3e5661;box-shadow:0 3px 9px rgba(48,66,76,.08)}
      .data-lens-body{overflow:auto;max-height:66vh}.data-grid{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;font-size:12px}.data-grid th{position:sticky;top:0;z-index:2;background:#f3f6f8;color:#52656f;font-weight:800;border-bottom:1px solid #dce4e8;border-right:1px solid #e4eaed;padding:8px 9px}.data-grid td{position:relative;border-bottom:1px solid #e8edef;border-right:1px solid #edf1f3;padding:7px 9px;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;background:#fff}.data-grid tr:hover td{outline:1px solid rgba(94,130,161,.14)}
      .heat-cell{background:var(--heat-bg,#fff)!important}.heat-cell::after{content:'';position:absolute;left:0;bottom:0;height:3px;width:var(--heat-width,0%);background:var(--heat-bar,#829ab3);opacity:.74}.heat-cell span{position:relative;z-index:1}.data-raw{margin:0;padding:15px;background:#25343d;color:#e6eff0;white-space:pre;overflow:auto;min-height:240px;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}.lens-note{padding:8px 12px;border-top:1px solid #e4eaed;font-size:11px;color:#77858e;background:#f8fafb}
      .lens-dialog{border:0;border-radius:18px;padding:0;max-width:min(1180px,94vw);width:94vw;box-shadow:0 28px 90px rgba(28,42,50,.28)}.lens-dialog::backdrop{background:rgba(25,36,43,.55);backdrop-filter:blur(4px)}.lens-dialog-wrap{padding:14px}.lens-dialog-top{display:flex;justify-content:flex-end;margin-bottom:8px}

      .product-dialog{border:0;border-radius:18px;padding:0;max-width:620px;width:min(92vw,620px);box-shadow:0 28px 90px rgba(28,42,50,.28)}.product-dialog::backdrop{background:rgba(25,36,43,.5);backdrop-filter:blur(4px)}.product-dialog-body{padding:20px}.product-dialog-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:15px}.product-dialog h2{margin:0;color:#2f434c}.product-dialog-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.account-button{margin-left:auto;border:1px solid #5e707c;background:rgba(255,255,255,.08);color:#eff5f6;border-radius:999px;padding:7px 11px;font-size:12px;font-weight:750}.account-button:hover{background:rgba(255,255,255,.14)}
      #productToast{position:fixed;left:50%;bottom:24px;transform:translate(-50%,18px);z-index:1000;display:flex;align-items:center;gap:12px;background:#263941;color:#eff5f5;border:1px solid #41565f;border-radius:999px;padding:10px 13px 10px 16px;box-shadow:0 14px 36px rgba(30,45,54,.25);opacity:0;pointer-events:none;transition:.18s ease;font-size:13px}#productToast.show{opacity:1;transform:translate(-50%,0);pointer-events:auto}#productToast button{border:0;border-radius:999px;background:#dfeeed;color:#315c60;padding:6px 10px;font-weight:850}

      .mw-layout{display:block!important;max-width:1040px;margin:18px auto 0!important}.mw-layout>aside{display:none!important}.mw-step{color:#354d57!important}.mw-num{background:linear-gradient(135deg,var(--p-violet),var(--p-blue))!important}.mw-code,.mw-data{background:#fbfdfd!important;border-color:#cbd9df!important;border-radius:13px!important}.mw-warning{background:#f7f0df!important;border-color:#e8d6ae!important;color:#79613b!important;border-radius:12px!important}.mw-ok{background:#e5f1eb!important;border-color:#c4ddcf!important;color:#3f6f5b!important;border-radius:11px!important}.mw-error{background:#f7e7e7!important;border-color:#e7c6c6!important;color:#8a5054!important;border-radius:11px!important}.mw-file-zone{background:linear-gradient(145deg,#f1f7fa,#f6f0f8,#fbf6ef)!important;border-color:#b9cbd4!important;border-radius:16px!important}.mw-file-zone:hover,.mw-file-zone.is-dragging{background:#edf4f5!important;border-color:#789eaa!important}.mw-chip{background:#edf1f7!important;border-color:#d5dce8!important;color:#58677b!important}

      @media(max-width:1000px){.core-grid,.result-columns,.product-dialog-grid{grid-template-columns:1fr}.core-shell{padding:20px 18px 42px}.product-flow-step{padding:6px 7px}.product-flow-step .label{display:none}.ws-flow{justify-content:flex-start!important}.ws-boundary-toggle{width:22px;height:48px}}
      @media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
    `;
    document.head.appendChild(style);
  }

  function stripTopToggles() {
    for (const id of ['wsProjectsToggle', 'wsModulesToggle', 'moduleWorkshopBtn']) document.getElementById(id)?.remove();
    const topbar = document.querySelector('.topbar');
    if (!topbar) return;
    [...topbar.children].forEach(child => {
      if (child.tagName === 'DIV' && child.style.flex === '1' && !child.textContent.trim()) child.remove();
    });
  }

  function setPaneCollapsed(side, collapsed) {
    const className = side === 'left' ? 'ws-left-off' : 'ws-right-off';
    const key = side === 'left' ? STORAGE.leftOff : STORAGE.rightOff;
    document.body.classList.toggle(className, collapsed);
    localStorage.setItem(key, collapsed ? '1' : '0');
    syncBoundaryButtons();
  }

  function syncBoundaryButtons() {
    const leftOff = document.body.classList.contains('ws-left-off');
    const rightOff = document.body.classList.contains('ws-right-off');
    const left = document.querySelector('.ws-boundary-toggle[data-side="left"]');
    const right = document.querySelector('.ws-boundary-toggle[data-side="right"]');
    if (left) {
      const text = leftOff ? '>>' : '<<';
      if (left.textContent !== text) left.textContent = text;
      left.title = leftOff ? 'Projects 열기' : 'Projects 접기';
    }
    if (right) {
      const text = rightOff ? '<<' : '>>';
      if (right.textContent !== text) right.textContent = text;
      right.title = rightOff ? 'Modules 열기' : 'Modules 접기';
    }
  }

  function addBoundaryButton(handle, side) {
    if (!handle || handle.querySelector(`[data-side="${side}"]`)) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ws-boundary-toggle';
    button.dataset.side = side;
    button.addEventListener('pointerdown', event => event.stopPropagation());
    button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      const className = side === 'left' ? 'ws-left-off' : 'ws-right-off';
      setPaneCollapsed(side, !document.body.classList.contains(className));
    });
    handle.appendChild(button);
  }

  function ensureBoundaryControls() {
    addBoundaryButton(document.querySelector('.ws-resize.left'), 'left');
    addBoundaryButton(document.querySelector('.ws-resize.right'), 'right');
    syncBoundaryButtons();
  }

  function installAccountButton() {
    const topbar = document.querySelector('.topbar');
    if (!topbar) return;
    let button = document.getElementById('productAccountButton');
    if (!button) {
      button = document.createElement('button');
      button.id = 'productAccountButton';
      button.className = 'account-button';
      button.onclick = openAccountDialog;
      topbar.appendChild(button);
    }
    const user = currentUser();
    button.textContent = user ? `@${user.username}` : 'Sign in';
  }

  function ensureDialog(id, html) {
    let dialog = document.getElementById(id);
    if (!dialog) {
      dialog = document.createElement('dialog');
      dialog.id = id;
      dialog.className = 'product-dialog';
      dialog.innerHTML = html;
      document.body.appendChild(dialog);
    }
    return dialog;
  }

  function openAccountDialog() {
    const user = currentUser();
    const dialog = ensureDialog('productAccountDialog', '<div></div>');
    dialog.innerHTML = `
      <div class="product-dialog-body">
        <div class="product-dialog-head"><div><div class="core-eyebrow">Local MVP identity</div><h2>${user ? 'Account' : 'Sign in'}</h2></div><button class="ghost" data-close>Close</button></div>
        <p class="muted">OAuth callback이 없는 로컬 실행이므로 실제 Google/GitHub 로그인을 흉내 내지 않습니다. 지금은 제작자·즐겨찾기·소유권 UX를 시험하기 위한 로컬 프로필입니다.</p>
        <div class="product-dialog-grid">
          <div class="field"><label>Display name</label><input id="accountName" value="${esc(user?.name || '고현민')}"></div>
          <div class="field"><label>Username</label><input id="accountUsername" value="${esc(user?.username || 'hyunmin')}"></div>
        </div>
        <div class="core-toolbar" style="justify-content:flex-end">
          ${user ? '<button class="ghost" id="accountSignOut">Sign out</button>' : ''}
          <button class="primary" id="accountSave">${user ? 'Save profile' : 'Continue locally'}</button>
        </div>
      </div>`;
    dialog.querySelector('[data-close]').onclick = () => dialog.close();
    dialog.querySelector('#accountSave').onclick = () => {
      const name = dialog.querySelector('#accountName').value.trim();
      const username = dialog.querySelector('#accountUsername').value.trim().replace(/^@/, '').replace(/[^a-zA-Z0-9_-]/g, '');
      if (!name || !username) {
        alert('이름과 username을 입력하세요.');
        return;
      }
      const users = loadJson(STORAGE.users, {});
      const id = user?.id || randomId('user');
      users[id] = {id, name, username, provider: 'local', createdAt: user?.createdAt || nowIso()};
      saveJson(STORAGE.users, users);
      localStorage.setItem(STORAGE.currentUser, id);
      if (app.activeProjectId) ensureProjectOwner(app.activeProjectId);
      dialog.close();
      installAccountButton();
      refreshModuleList();
      renderModuleShelf();
      if (app.centerMode === 'core') renderCore();
    };
    const signOut = dialog.querySelector('#accountSignOut');
    if (signOut) signOut.onclick = () => {
      localStorage.removeItem(STORAGE.currentUser);
      dialog.close();
      installAccountButton();
      refreshModuleList();
      renderModuleShelf();
      if (app.centerMode === 'core') renderCore();
    };
    dialog.showModal();
  }

  function renderFlowRail() {
    const box = document.getElementById('leesinFlowBar');
    if (!box) return;
    let stage = 'data';
    if (app.centerMode === 'core') stage = activeRun()?.stage || 'data';
    else if (app.centerMode === 'workshop') {
      const result = document.getElementById('mwResultPanel');
      const mapping = document.getElementById('mwPreparedPanel');
      if (result && getComputedStyle(result).display !== 'none' && document.getElementById('mwResult')?.textContent.trim()) stage = 'result';
      else if (mapping && getComputedStyle(mapping).display !== 'none') stage = 'mapping';
      else stage = 'module';
    } else {
      const breadcrumb = document.querySelector('#mainView .breadcrumb')?.textContent || '';
      if (breadcrumb.includes('Analyses')) stage = 'result';
      else if (breadcrumb.includes('Proposals')) stage = 'next';
      else stage = 'data';
    }
    const activeIndex = stageIndex(stage);
    const signature = `${app.centerMode}:${stage}:${activeRun()?.id || ''}`;
    if (box.dataset.productSignature === signature && box.querySelector('.product-flow-step')) return;
    box.dataset.productSignature = signature;
    box.innerHTML = STEP_ORDER.map((step, index) => {
      const active = index === activeIndex;
      const reached = index <= activeIndex && app.centerMode === 'core';
      return `${index ? '<span class="product-flow-arrow">›</span>' : ''}<button type="button" class="product-flow-step ${active ? 'active' : ''} ${index < activeIndex ? 'done' : ''} ${reached ? 'reached' : ''}" data-step="${step}" ${reached ? '' : 'disabled'}><span class="dot"></span><span class="label">${STEP_LABELS[step]}</span></button>`;
    }).join('');
    box.querySelectorAll('.product-flow-step.reached').forEach(button => {
      button.onclick = () => {
        const target = button.dataset.step;
        if (target === activeRun()?.stage) return;
        rollbackTo(target);
      };
    });
  }

  function selectedDataKeys(run) {
    return new Set((run.dataRefs || []).map(ref => `${ref.kind}:${ref.id}`));
  }

  function tableFiles() {
    return (app.workspace?.files || []).filter(file => /\.(csv|tsv|txt)$/i.test(file.name || ''));
  }

  function dataRefName(ref) {
    if (ref.kind === 'cluster') {
      const item = app.project?.clusters?.find(cluster => cluster.id === ref.id);
      return item?.name || item?.filename || ref.name || ref.id;
    }
    const item = app.workspace?.files?.find(file => file.id === ref.id);
    return item?.name || ref.name || ref.id;
  }

  function dataItemHtml(item, kind, selected) {
    const name = item.name || item.filename || item.id;
    const meta = kind === 'cluster'
      ? `${item.filename || 'CSV cluster'} · ${item.protocol || 'protocol unspecified'}`
      : `${formatSize(item.size)} · original file`;
    return `<label class="core-data-item ${selected ? 'selected' : ''}" data-data-kind="${kind}" data-data-id="${esc(item.id)}">
      <input type="checkbox" ${selected ? 'checked' : ''}>
      <span><span class="core-data-name">${kind === 'cluster' ? '▦' : '📄'} ${esc(name)}</span><span class="core-data-meta">${esc(meta)}</span></span>
      <button type="button" class="core-mini" data-preview-kind="${kind}" data-preview-id="${esc(item.id)}">Preview</button>
    </label>`;
  }

  function renderCore() {
    if (!app.activeProjectId || !app.project) return;
    app.centerMode = 'core';
    const run = activeRun();
    const state = activeCore();
    const main = document.getElementById('mainView');
    if (!run || !main) return;
    const owner = getProjectOwner(app.activeProjectId);
    main.classList.remove('ws-main-pad');
    main.innerHTML = `<div class="core-shell">
      <div class="core-header">
        <div><div class="core-eyebrow">Core workspace · Cycle ${esc(run.cycle || 1)}</div><h1 class="core-title">${esc(app.project.title)}</h1><div class="core-meta">${esc(owner)} · ${esc(app.project.description || 'Project-based experimental workflow')}</div></div>
        <div class="core-header-actions"><button class="ghost" id="coreHistoryBtn">Run history ${state.completedRuns.length ? `(${state.completedRuns.length})` : ''}</button><button class="ghost" id="coreAddDataBtn">+ Data</button></div>
      </div>
      <div class="core-stage-card"><div class="core-stage-accent"></div><div class="core-stage-body">${stageHtml(run)}</div></div>
    </div>`;
    bindCore(run);
    renderFlowRail();
  }

  function stageHtml(run) {
    if (run.stage === 'data') return dataStageHtml(run);
    if (run.stage === 'module') return moduleStageHtml(run);
    if (run.stage === 'mapping') return mappingStageHtml(run);
    if (run.stage === 'analysis') return analysisStageHtml(run);
    if (run.stage === 'result') return resultStageHtml(run);
    return nextStageHtml(run);
  }

  function dataStageHtml(run) {
    const selected = selectedDataKeys(run);
    const clusters = app.project?.clusters || [];
    const files = tableFiles();
    const items = [
      ...clusters.map(item => dataItemHtml(item, 'cluster', selected.has(`cluster:${item.id}`))),
      ...files.map(item => dataItemHtml(item, 'file', selected.has(`file:${item.id}`))),
    ].join('') || '<div class="empty-shelf">CSV/TSV Cluster나 Project file을 추가하세요.</div>';
    return `<div class="core-stage-head"><div><h2>Choose data</h2><p>분석에 사용할 자료를 고릅니다. 원본 파일과 기존 Data Cluster를 함께 관리할 수 있습니다.</p></div><span class="module-badge badge-compatible">${run.dataRefs.length} selected</span></div>
      <div class="core-grid"><div class="core-subcard"><div class="core-toolbar" style="margin-top:0"><button class="ghost" id="selectAllData">Select all</button><button class="ghost" id="clearData">Clear</button>${/prime/i.test(app.project.title) ? '<button class="ghost" id="generatePrimeData">Generate prime sample</button>' : ''}</div><div class="core-data-list">${items}</div></div>
      <div class="core-subcard"><h3>Current selection</h3><div class="core-selection-summary">${run.dataRefs.length ? run.dataRefs.map(ref => `<span class="core-chip">${esc(dataRefName(ref))}</span>`).join('') : '<span class="muted">아직 선택한 Data가 없습니다.</span>'}</div><p class="core-note">선택한 Data의 열은 다음 단계에서 Module 호환성과 Mapping에 사용됩니다.</p></div></div>
      <div class="core-next-row"><button class="primary" id="continueModule" ${run.dataRefs.length ? '' : 'disabled'}>Continue to Module →</button></div>`;
  }

  function moduleStageHtml(run) {
    const module = moduleById(run.module?.id);
    const attached = module ? `<div style="width:100%"><div class="module-slot-title">${esc(module.title)}</div><div class="module-slot-author">${esc(module.author)} · v${esc(module.version || '0.1.0')}</div><div class="module-slot-desc">${esc(module.description || '')}</div><div style="margin-top:10px"><span class="module-badge badge-${module.visibility === 'public' ? 'public' : 'private'}">${esc(module.visibility || 'private')}</span>${compatibilityBadge(module)}</div></div>` : `<div><div class="module-slot-title">Drop a Module here</div><div class="module-slot-desc">오른쪽 선반에서 <strong>Use</strong>를 누르거나 Module 카드를 이곳으로 드래그하세요.</div></div>`;
    return `<div class="core-stage-head"><div><h2>Attach a Module</h2><p>Question은 계산기가 아니라 의미 기록입니다. 실제 분석 규칙은 Module이 담당합니다.</p></div></div>
      <div class="module-slot ${module ? 'attached' : ''}" id="coreModuleSlot">${attached}</div>
      <div class="field" style="margin-top:15px"><label>Question <span class="muted">(human-readable, optional)</span></label><input class="core-question" id="coreQuestionInput" value="${esc(run.question || '')}" placeholder="이 Module을 현재 Data에 왜 적용하는가?"></div>
      <div class="core-next-row"><button class="ghost" id="backData">← Data</button><button class="primary" id="continueMapping" ${module ? '' : 'disabled'}>Continue to Mapping →</button></div>`;
  }

  function mappingStageHtml(run) {
    const module = moduleById(run.module?.id);
    if (!module) return '<div class="empty-shelf">Module을 먼저 선택하세요.</div>';
    if (module.kind === 'builtin') {
      const rows = (module.requiredColumns || []).map(name => `<div class="mapping-row"><strong>${esc(name)}</strong><div class="mapping-arrow">←</div><div><span class="core-chip">Column: ${esc(name)}</span></div></div>`).join('');
      const invalidFiles = run.dataRefs.some(ref => ref.kind !== 'cluster');
      return `<div class="core-stage-head"><div><h2>Confirm mapping</h2><p>${esc(module.title)}의 첫 MVP binding은 명시적으로 고정되어 있습니다.</p></div>${compatibilityBadge(module)}</div><div class="core-subcard">${rows}</div>${invalidFiles ? '<div class="mw-error" style="margin-top:12px">현재 built-in adapter는 Data Cluster만 실행할 수 있습니다. Project file은 제거하거나 Cluster로 추가하세요.</div>' : ''}<div class="core-next-row"><button class="ghost" id="backModule">← Module</button><button class="primary" id="runAnalysis" ${invalidFiles ? 'disabled' : ''}>Run analysis</button></div>`;
    }
    if (!run.prepared) {
      return `<div class="core-stage-head"><div><h2>Prepare mapping</h2><p>함수 signature와 Data header를 읽어 입력 연결을 제안합니다.</p></div></div><div class="analysis-running"><div class="analysis-spinner"></div><strong>Reading function and data…</strong><div class="muted" style="margin-top:6px">모호한 부분만 사람이 확인합니다.</div></div>`;
    }
    const params = run.prepared.selectedFunction?.parameters || [];
    const columns = run.prepared.data?.columns || [];
    const rows = params.filter(param => !['var_positional', 'var_keyword'].includes(param.kind)).map(param => {
      const current = run.mapping?.[param.name] ?? run.prepared.suggestedMapping?.[param.name] ?? '';
      const options = [
        `<option value="">-- choose --</option>`,
        `<option value="__rows__" ${current === '__rows__' ? 'selected' : ''}>Whole table (rows)</option>`,
        ...columns.map(column => `<option value="${esc(column)}" ${current === column ? 'selected' : ''}>Column: ${esc(column)}</option>`),
        ...(!param.required ? [`<option value="__default__" ${current === '__default__' || !current ? 'selected' : ''}>Use default (${esc(param.default)})</option>`] : []),
      ].join('');
      return `<div class="mapping-row"><div><strong>${esc(param.name)}</strong>${param.annotation ? `<div class="core-note">${esc(param.annotation)}</div>` : ''}</div><div class="mapping-arrow">←</div><select data-map-param="${esc(param.name)}">${options}</select></div>`;
    }).join('');
    return `<div class="core-stage-head"><div><h2>Confirm mapping</h2><p>${esc(module.title)} · ${esc(run.prepared.data?.rowCount || 0)} rows</p></div>${compatibilityBadge(module)}</div><div class="core-subcard">${rows || '<span class="muted">이 함수에는 연결할 입력이 없습니다.</span>'}</div><div class="core-next-row"><button class="ghost" id="backModule">← Module</button><button class="primary" id="runAnalysis">Run analysis</button></div>`;
  }

  function analysisStageHtml(run) {
    return `<div class="core-stage-head"><div><h2>Analysis</h2><p>${esc(moduleById(run.module?.id)?.title || '')}</p></div></div><div class="analysis-running"><div class="analysis-spinner"></div><strong>Running the declared function…</strong><div class="muted" style="margin-top:6px">Leesin은 Module 밖의 추론을 임의로 채우지 않습니다.</div></div>`;
  }

  function resultStageHtml(run) {
    const result = run.result || {};
    const status = result.status || 'ok';
    const cls = status === 'ok' ? 'result-ok' : ['assumption_failed', 'protocol_mismatch', 'invalid_data', 'error'].includes(status) ? 'result-stop' : 'result-info';
    const label = result.statusLabel || (status === 'ok' ? 'RESULT' : status.toUpperCase());
    const summary = result.summary || result.title || 'Analysis finished.';
    const body = result.kind === 'custom'
      ? `<div class="result-json">${esc(JSON.stringify(result.value, null, 2))}</div>`
      : `${result.preview?.length ? previewObjectTable(result.preview) : ''}`;
    const assumptions = result.assumptions || [];
    const limits = result.limits || [];
    return `<div class="core-stage-head"><div><span class="result-status ${cls}">${esc(label)}</span><div class="result-summary">${esc(summary)}</div><div class="core-meta">${esc(moduleById(run.module?.id)?.title || '')}${run.question ? ` · ${esc(run.question)}` : ''}</div></div></div>${body}<div class="result-columns"><div class="core-subcard"><h3>Assumptions</h3>${listHtml(assumptions)}</div><div class="core-subcard"><h3>Limits</h3>${listHtml(limits)}</div></div>${result.diagnostics?.length ? `<div class="core-subcard" style="margin-top:14px"><h3>Diagnostics</h3>${listHtml(result.diagnostics)}</div>` : ''}<div class="core-next-row"><button class="ghost" id="backMapping">← Mapping</button>${run.proposal ? '<button class="primary" id="reviewNext">Review Next →</button>' : '<button class="primary" id="newCycle">Start another cycle</button>'}</div>`;
  }

  function nextStageHtml(run) {
    const proposal = run.proposal || {};
    const payload = proposal.payload || proposal;
    const input = payload.input || {};
    return `<div class="core-stage-head"><div><h2>Next experiment</h2><p>Module이 반환한 다음 관측 제안입니다. 실행 여부는 사람이 결정합니다.</p></div><span class="module-badge badge-public">Proposal</span></div><div class="core-subcard"><div class="result-summary">${Object.keys(input).length ? Object.entries(input).map(([key, value]) => `${esc(key)} = ${esc(value)}`).join(', ') : 'New observation'}</div><p>${esc(payload.reason || '현재 Result에 이어 새로운 Data를 얻습니다.')}</p>${payload.repeatInstruction ? `<p class="muted">${esc(payload.repeatInstruction)}</p>` : ''}</div><div class="core-next-row"><button class="ghost" id="backResult">← Result</button><button class="primary" id="executeNext">${proposal.id && input.N != null ? 'Run next experiment' : 'Start next cycle'}</button></div>`;
  }

  function listHtml(items) {
    return items?.length ? `<ul class="result-list">${items.map(item => `<li>${esc(item)}</li>`).join('')}</ul>` : '<div class="muted">None declared.</div>';
  }

  function previewObjectTable(rows) {
    if (!rows?.length) return '';
    const columns = Object.keys(rows[0]);
    const shown = rows.slice(0, 30);
    return `<div class="data-lens"><div class="data-lens-body"><table class="data-grid"><thead><tr>${columns.map(column => `<th>${esc(column)}</th>`).join('')}</tr></thead><tbody>${shown.map(row => `<tr>${columns.map(column => `<td>${esc(row[column])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>${rows.length > shown.length ? `<div class="lens-note">Showing ${shown.length} of ${rows.length} rows.</div>` : ''}</div>`;
  }

  function bindCore(run) {
    document.getElementById('coreHistoryBtn')?.addEventListener('click', openRunHistory);
    document.getElementById('coreAddDataBtn')?.addEventListener('click', () => window.openClusterDialog?.());

    document.querySelectorAll('[data-preview-kind]').forEach(button => {
      button.onclick = event => {
        event.preventDefault();
        event.stopPropagation();
        openDataPreview({kind: button.dataset.previewKind, id: button.dataset.previewId});
      };
    });
    document.querySelectorAll('.core-data-item').forEach(label => {
      const checkbox = label.querySelector('input');
      checkbox.onchange = () => toggleDataRef(label.dataset.dataKind, label.dataset.dataId, checkbox.checked);
    });
    document.getElementById('selectAllData')?.addEventListener('click', () => {
      run.dataRefs = [
        ...(app.project?.clusters || []).map(item => ({kind: 'cluster', id: item.id, name: item.name || item.filename})),
        ...tableFiles().map(item => ({kind: 'file', id: item.id, name: item.name})),
      ];
      run.mapping = {}; run.prepared = null; run.result = null; run.proposal = null;
      saveRun(run); void computeDataColumns(run).then(renderCore);
    });
    document.getElementById('clearData')?.addEventListener('click', () => {
      run.dataRefs = []; run.dataColumns = []; run.mapping = {}; run.prepared = null; run.result = null; run.proposal = null;
      saveRun(run); renderCore(); renderModuleShelf();
    });
    document.getElementById('generatePrimeData')?.addEventListener('click', generatePrimeSample);
    document.getElementById('continueModule')?.addEventListener('click', () => setStage('module'));
    document.getElementById('backData')?.addEventListener('click', () => rollbackTo('data'));
    document.getElementById('backModule')?.addEventListener('click', () => rollbackTo('module'));
    document.getElementById('backMapping')?.addEventListener('click', () => rollbackTo('mapping'));
    document.getElementById('backResult')?.addEventListener('click', () => rollbackTo('result'));
    document.getElementById('continueMapping')?.addEventListener('click', () => setStage('mapping'));
    document.getElementById('runAnalysis')?.addEventListener('click', runCurrentAnalysis);
    document.getElementById('reviewNext')?.addEventListener('click', () => setStage('next'));
    document.getElementById('newCycle')?.addEventListener('click', () => archiveAndStartCycle({carryModule: true}));
    document.getElementById('executeNext')?.addEventListener('click', executeNextExperiment);

    const question = document.getElementById('coreQuestionInput');
    if (question) question.oninput = () => { run.question = question.value; saveRun(run); };

    const slot = document.getElementById('coreModuleSlot');
    if (slot) {
      for (const eventName of ['dragenter', 'dragover']) slot.addEventListener(eventName, event => {
        event.preventDefault();
        slot.classList.add('dragover');
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
      });
      for (const eventName of ['dragleave', 'dragend']) slot.addEventListener(eventName, () => slot.classList.remove('dragover'));
      slot.addEventListener('drop', event => {
        event.preventDefault();
        slot.classList.remove('dragover');
        const id = event.dataTransfer?.getData('application/x-leesin-module') || event.dataTransfer?.getData('text/plain');
        if (id && moduleById(id)) attachModule(id);
      });
    }

    document.querySelectorAll('[data-map-param]').forEach(select => {
      select.onchange = () => {
        run.mapping[select.dataset.mapParam] = select.value;
        saveRun(run);
      };
    });

    if (run.stage === 'mapping' && run.module && moduleById(run.module.id)?.kind !== 'builtin' && !run.prepared) {
      void prepareCurrentMapping();
    }
  }

  async function toggleDataRef(kind, id, checked) {
    const run = activeRun();
    if (!run) return;
    const key = `${kind}:${id}`;
    const current = new Map(run.dataRefs.map(ref => [`${ref.kind}:${ref.id}`, ref]));
    if (checked) current.set(key, {kind, id, name: dataRefName({kind, id})}); else current.delete(key);
    run.dataRefs = [...current.values()];
    run.mapping = {}; run.prepared = null; run.analysis = null; run.analysisId = null; run.result = null; run.proposal = null;
    saveRun(run);
    await computeDataColumns(run);
    renderCore();
    renderModuleShelf();
  }

  async function computeDataColumns(run) {
    try {
      const data = await combinedData(run.dataRefs);
      run.dataColumns = data.columns;
    } catch (_) {
      run.dataColumns = [];
    }
    saveRun(run);
  }

  function setStage(stage) {
    const run = activeRun();
    if (!run) return;
    if (stageIndex(stage) < stageIndex(run.stage)) {
      void rollbackTo(stage);
      return;
    }
    run.stage = stage;
    saveRun(run);
    renderCore();
    renderModuleShelf();
  }

  function moduleById(id) {
    return app.modules.find(module => module.id === id) || null;
  }

  function moduleSnapshot(module) {
    return module ? {
      id: module.id,
      title: module.title,
      kind: module.kind,
      version: module.version,
      author: module.author,
      entryFunction: module.entryFunction,
      questionId: module.questionId || null,
    } : null;
  }

  async function attachModule(id) {
    const module = moduleById(id);
    const run = activeRun();
    if (!module || !run) return;
    if (stageIndex(run.stage) > stageIndex('module') && run.module?.id !== id) {
      await rollbackTo('module');
      if (stageIndex(activeRun()?.stage || 'data') > stageIndex('module')) return;
    }
    const nextRun = activeRun();
    nextRun.module = moduleSnapshot(module);
    nextRun.mapping = {};
    nextRun.prepared = null;
    nextRun.analysis = null;
    nextRun.analysisId = null;
    nextRun.result = null;
    nextRun.proposal = null;
    nextRun.stage = 'module';
    saveRun(nextRun);
    incrementModuleUsage(id);
    app.centerMode = 'core';
    renderCore();
    renderModuleShelf();
    showToast(`${module.title}을 현재 Run에 장착했습니다.`);
  }

  function incrementModuleUsage(id) {
    const store = loadJson(STORAGE.usage, {});
    store[id] = Number(store[id] || 0) + 1;
    saveJson(STORAGE.usage, store);
  }

  function compatibility(module) {
    const columns = activeRun()?.dataColumns || [];
    if (!columns.length) return {state: 'unknown', text: 'Choose data first'};
    const normalized = new Set(columns.map(value => String(value).toLowerCase().replace(/[^a-z0-9가-힣]/g, '')));
    const required = module.requiredColumns || (module.inputContract || []).filter(input => input.required !== false).map(input => input.name) || [];
    if (!required.length) return {state: 'compatible', text: 'Ready'};
    const exact = required.filter(name => normalized.has(String(name).toLowerCase().replace(/[^a-z0-9가-힣]/g, ''))).length;
    if (exact === required.length) return {state: 'compatible', text: 'Compatible'};
    if (columns.length >= required.length) return {state: 'mapping', text: 'Mapping needed'};
    return {state: 'mapping', text: `${required.length - exact} input(s) missing`};
  }

  function compatibilityBadge(module) {
    const item = compatibility(module);
    return `<span class="module-badge ${item.state === 'compatible' ? 'badge-compatible' : 'badge-mapping'}">${esc(item.text)}</span>`;
  }

  async function prepareCurrentMapping() {
    const run = activeRun();
    const module = moduleById(run?.module?.id);
    if (!run || !module || !module.code) return;
    try {
      const data = await combinedData(run.dataRefs);
      const prepared = await requestJson('/api/module-workshop/prepare', {
        method: 'POST',
        body: JSON.stringify({code: module.code, dataText: data.text, functionName: module.entryFunction}),
      });
      run.prepared = prepared;
      run.mapping = {...(prepared.suggestedMapping || {}), ...(run.mapping || {})};
      run.dataColumns = prepared.data?.columns || data.columns;
      saveRun(run);
      renderCore();
      renderModuleShelf();
    } catch (error) {
      run.stage = 'result';
      run.result = {kind: 'custom', status: 'invalid_data', statusLabel: 'ANALYSIS STOPPED', summary: error.message, value: null, assumptions: module.assumptions || [], limits: module.limits || []};
      saveRun(run);
      renderCore();
    }
  }

  async function runCurrentAnalysis() {
    const run = activeRun();
    const module = moduleById(run?.module?.id);
    if (!run || !module) return;
    run.stage = 'analysis';
    saveRun(run);
    renderCore();
    try {
      if (module.kind === 'builtin') {
        const clusterIds = run.dataRefs.filter(ref => ref.kind === 'cluster').map(ref => ref.id);
        const analysis = await requestJson(`/api/projects/${app.activeProjectId}/analyze`, {
          method: 'POST',
          body: JSON.stringify({questionId: module.questionId, clusterIds}),
        });
        const outcome = analysis.outcome || {};
        run.analysisId = analysis.id;
        run.analysis = analysis;
        run.result = {
          kind: 'builtin',
          status: outcome.status || 'ok',
          statusLabel: outcome.status_label || outcome.status || 'RESULT',
          title: outcome.title,
          summary: outcome.summary,
          preview: outcome.preview || [],
          assumptions: outcome.assumptions || [],
          limits: outcome.limits || [],
          diagnostics: outcome.diagnostics || [],
        };
        run.proposal = analysis.proposal || null;
      } else {
        const data = await combinedData(run.dataRefs);
        const result = await requestJson('/api/module-workshop/run', {
          method: 'POST',
          body: JSON.stringify({
            code: module.code,
            functionName: module.entryFunction,
            dataText: data.text,
            mapping: run.mapping || {},
          }),
        });
        run.analysis = {executionMs: result.executionMs, rowCount: result.rowCount};
        run.result = {
          kind: 'custom',
          status: 'ok',
          statusLabel: 'RESULT',
          title: module.title,
          summary: `${module.title} returned ${result.resultType}.`,
          value: result.result,
          assumptions: module.assumptions || [],
          limits: module.limits || [],
          diagnostics: [`${result.rowCount} rows`, `${Number(result.executionMs || 0).toFixed(3)} ms`],
        };
        const possibleProposal = result.result && typeof result.result === 'object' ? (result.result.proposal || result.result.next) : null;
        run.proposal = possibleProposal ? {payload: possibleProposal} : null;
      }
      run.stage = 'result';
      saveRun(run);
      await refreshContext({render: false});
      renderCore();
    } catch (error) {
      run.stage = 'result';
      run.result = {
        kind: 'custom', status: 'error', statusLabel: 'ANALYSIS STOPPED', summary: error.message,
        value: null, assumptions: module.assumptions || [], limits: module.limits || [], diagnostics: [error.name || 'Error'],
      };
      run.proposal = null;
      saveRun(run);
      renderCore();
    }
  }

  async function executeNextExperiment() {
    const run = activeRun();
    if (!run?.proposal) {
      archiveAndStartCycle({carryModule: true});
      return;
    }
    const input = run.proposal.payload?.input || run.proposal.input || {};
    if (run.proposal.id && input.N != null) {
      const button = document.getElementById('executeNext');
      if (button) { button.disabled = true; button.textContent = 'Running…'; }
      try {
        await requestJson(`/api/projects/${app.activeProjectId}/proposals/${run.proposal.id}/start`, {method: 'POST', body: '{}'});
        await requestJson(`/api/projects/${app.activeProjectId}/mvp/prime-benchmark`, {
          method: 'POST',
          body: JSON.stringify({nValues: [Number(input.N)], originProposalId: run.proposal.id}),
        });
        await refreshContext({render: false});
        archiveAndStartCycle({carryModule: true, selectAllClusters: true});
        showToast(`N=${input.N} 실험을 실행하고 Data 단계로 돌아왔습니다.`);
        window.renderTree?.();
      } catch (error) {
        alert(error.message);
        if (button) { button.disabled = false; button.textContent = 'Run next experiment'; }
      }
      return;
    }
    archiveAndStartCycle({carryModule: true});
    showToast('새 실험 Cycle을 시작했습니다.');
  }

  async function generatePrimeSample() {
    const raw = prompt('실험할 N 값을 쉼표로 입력하세요.', '5,100');
    if (raw == null) return;
    const values = raw.split(',').map(value => Number(value.trim())).filter(Number.isFinite);
    if (!values.length) return;
    try {
      const response = await requestJson(`/api/projects/${app.activeProjectId}/mvp/prime-benchmark`, {
        method: 'POST', body: JSON.stringify({nValues: values, originProposalId: null}),
      });
      await refreshContext({render: false});
      const run = activeRun();
      run.dataRefs.push({kind: 'cluster', id: response.cluster.id, name: response.cluster.name});
      saveRun(run);
      await computeDataColumns(run);
      renderCore();
      window.renderTree?.();
      showToast(`${response.cluster.name}을 Data에 추가했습니다.`);
    } catch (error) {
      alert(error.message);
    }
  }

  function formatSize(value) {
    const n = Number(value || 0);
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }

  async function dataTextForRef(ref) {
    if (ref.kind === 'cluster') {
      const cluster = app.project?.clusters?.find(item => item.id === ref.id);
      if (!cluster) throw new Error(`Unknown cluster: ${ref.id}`);
      return {name: cluster.name || cluster.filename, text: cluster.csvText || ''};
    }
    const file = await requestJson(`/api/projects/${app.activeProjectId}/files/${ref.id}`);
    if (file.textContent == null) throw new Error(`${file.name}은 text table로 읽을 수 없습니다.`);
    return {name: file.name, text: file.textContent};
  }

  async function combinedData(refs) {
    if (!refs?.length) throw new Error('Data를 하나 이상 선택하세요.');
    const sources = [];
    for (const ref of refs) sources.push(await dataTextForRef(ref));
    const parsed = sources.map(source => ({...source, table: parseDelimited(source.text)}));
    const header = parsed[0].table.columns;
    const normalized = header.map(value => value.trim().toLowerCase());
    const rows = [];
    for (const source of parsed) {
      const current = source.table.columns.map(value => value.trim().toLowerCase());
      if (current.length !== normalized.length || current.some((value, index) => value !== normalized[index])) {
        throw new Error(`${source.name}의 header가 다른 Data와 일치하지 않습니다. 이번 MVP에서는 같은 열 구조만 합칠 수 있습니다.`);
      }
      rows.push(...source.table.rows);
    }
    return {columns: header, rows, text: toCsv(header, rows), sources};
  }

  function detectDelimiter(text) {
    const line = String(text || '').split(/\r?\n/).find(value => value.trim()) || '';
    const candidates = ['\t', ',', ';', '|'];
    let best = ',';
    let bestCount = -1;
    for (const delimiter of candidates) {
      let count = 0;
      let quoted = false;
      for (let i = 0; i < line.length; i += 1) {
        if (line[i] === '"') quoted = !quoted;
        else if (!quoted && line[i] === delimiter) count += 1;
      }
      if (count > bestCount) { best = delimiter; bestCount = count; }
    }
    return best;
  }

  function parseDelimited(text) {
    const input = String(text || '').replace(/^\ufeff/, '');
    if (!input.trim()) throw new Error('Data가 비어 있습니다.');
    const delimiter = detectDelimiter(input);
    const matrix = [];
    let row = [];
    let field = '';
    let quoted = false;
    for (let i = 0; i < input.length; i += 1) {
      const ch = input[i];
      if (quoted) {
        if (ch === '"' && input[i + 1] === '"') { field += '"'; i += 1; }
        else if (ch === '"') quoted = false;
        else field += ch;
      } else if (ch === '"') quoted = true;
      else if (ch === delimiter) { row.push(field); field = ''; }
      else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && input[i + 1] === '\n') i += 1;
        row.push(field); field = '';
        if (row.some(value => String(value).trim() !== '')) matrix.push(row);
        row = [];
      } else field += ch;
    }
    row.push(field);
    if (row.some(value => String(value).trim() !== '')) matrix.push(row);
    if (!matrix.length) throw new Error('Header를 읽을 수 없습니다.');
    const columns = matrix.shift().map(value => String(value).trim());
    if (!columns.length || columns.some(value => !value)) throw new Error('비어 있는 header가 있습니다.');
    const rows = matrix.map(values => {
      const item = {};
      columns.forEach((column, index) => { item[column] = values[index] ?? ''; });
      return item;
    });
    return {columns, rows, delimiter};
  }

  function csvEscape(value) {
    const text = String(value ?? '');
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function toCsv(columns, rows) {
    return [columns.map(csvEscape).join(','), ...rows.map(row => columns.map(column => csvEscape(row[column])).join(','))].join('\n');
  }

  function numericStats(table) {
    const stats = {};
    for (const column of table.columns) {
      const values = table.rows.map(row => Number(row[column])).filter(Number.isFinite);
      if (values.length >= Math.max(1, Math.ceil(table.rows.length * 0.5))) {
        stats[column] = {min: Math.min(...values), max: Math.max(...values)};
      }
    }
    return stats;
  }

  function heatStyle(value, stat) {
    const number = Number(value);
    if (!stat || !Number.isFinite(number)) return '';
    const span = stat.max - stat.min;
    const t = span === 0 ? 0.5 : Math.max(0, Math.min(1, (number - stat.min) / span));
    const hue = t < 0.5 ? 215 + t * 100 : 265 - (t - 0.5) * 470;
    const safeHue = ((hue % 360) + 360) % 360;
    const bg = `hsla(${safeHue.toFixed(0)},58%,78%,.43)`;
    const bar = `hsl(${safeHue.toFixed(0)},48%,58%)`;
    return `--heat-bg:${bg};--heat-bar:${bar};--heat-width:${(8 + t * 92).toFixed(1)}%`;
  }

  function dataLensHtml(text, title = 'Data') {
    let table;
    try { table = parseDelimited(text); }
    catch (error) { return `<div class="mw-error">${esc(error.message)}</div><pre class="data-raw">${esc(text)}</pre>`; }
    const id = randomId('lens');
    const maxRows = 120;
    const maxColumns = 30;
    const shownColumns = table.columns.slice(0, maxColumns);
    const shownRows = table.rows.slice(0, maxRows);
    const stats = numericStats(table);
    const tableRows = shownRows.map(row => `<tr>${shownColumns.map(column => `<td><span>${esc(row[column])}</span></td>`).join('')}</tr>`).join('');
    const heatRows = shownRows.map(row => `<tr>${shownColumns.map(column => `<td class="${stats[column] ? 'heat-cell' : ''}" style="${heatStyle(row[column], stats[column])}"><span>${esc(row[column])}</span></td>`).join('')}</tr>`).join('');
    const head = `<thead><tr>${shownColumns.map(column => `<th>${esc(column)}</th>`).join('')}</tr></thead>`;
    const note = `${table.rows.length} rows · ${table.columns.length} columns${table.rows.length > maxRows || table.columns.length > maxColumns ? ` · previewing first ${Math.min(maxRows, table.rows.length)} × ${Math.min(maxColumns, table.columns.length)}` : ''}`;
    return `<div class="data-lens" id="${id}"><div class="data-lens-head"><div><div class="data-lens-title">${esc(title)}</div><div class="data-lens-meta">${esc(note)}</div></div><div class="lens-toggle"><button class="active" data-lens-mode="heat">Heat</button><button data-lens-mode="table">Table</button><button data-lens-mode="raw">Raw</button></div></div><div class="data-lens-body"><div data-lens-view="heat"><table class="data-grid">${head}<tbody>${heatRows}</tbody></table></div><div data-lens-view="table" hidden><table class="data-grid">${head}<tbody>${tableRows}</tbody></table></div><pre class="data-raw" data-lens-view="raw" hidden>${esc(text)}</pre></div><div class="lens-note">Heat lens는 각 숫자 열 안에서 최소–최대 크기를 색으로 표시합니다. 원본 값은 바꾸지 않습니다.</div></div>`;
  }

  function bindDataLens(root) {
    root.querySelectorAll('.data-lens').forEach(lens => {
      if (lens.dataset.bound) return;
      lens.dataset.bound = '1';
      lens.querySelectorAll('[data-lens-mode]').forEach(button => {
        button.onclick = () => {
          const mode = button.dataset.lensMode;
          lens.querySelectorAll('[data-lens-mode]').forEach(item => item.classList.toggle('active', item === button));
          lens.querySelectorAll('[data-lens-view]').forEach(view => { view.hidden = view.dataset.lensView !== mode; });
        };
      });
    });
  }

  async function openDataPreview(ref) {
    try {
      const source = await dataTextForRef(ref);
      const dialog = ensureDialog('productDataLensDialog', '<div></div>');
      dialog.className = 'lens-dialog';
      dialog.innerHTML = `<div class="lens-dialog-wrap"><div class="lens-dialog-top"><button class="ghost" data-close>Close</button></div>${dataLensHtml(source.text, source.name)}</div>`;
      dialog.querySelector('[data-close]').onclick = () => dialog.close();
      bindDataLens(dialog);
      dialog.showModal();
    } catch (error) {
      alert(error.message);
    }
  }

  function enhanceLegacyDataView() {
    const main = document.getElementById('mainView');
    if (!main) return;
    const fileView = main.querySelector('.ws-file-view');
    const filePre = fileView?.querySelector('.ws-code-preview');
    if (fileView && filePre && !fileView.querySelector('.data-lens')) {
      const title = fileView.querySelector('h1')?.textContent || 'Data';
      if (/\.(csv|tsv|txt)$/i.test(title)) {
        const wrapper = document.createElement('div');
        wrapper.style.marginTop = '18px';
        wrapper.innerHTML = dataLensHtml(filePre.textContent, title);
        filePre.before(wrapper);
        filePre.hidden = true;
        bindDataLens(wrapper);
      }
    }
    const summary = [...main.querySelectorAll('summary')].find(item => item.textContent.trim() === 'Raw CSV');
    const details = summary?.closest('details');
    const raw = details?.querySelector('pre');
    if (raw && !details.parentElement.querySelector(':scope > .data-lens')) {
      const title = main.querySelector('section.panel h2')?.textContent || 'Data Cluster';
      const wrapper = document.createElement('div');
      wrapper.style.marginTop = '16px';
      wrapper.innerHTML = dataLensHtml(raw.textContent, title);
      details.before(wrapper);
      details.hidden = true;
      bindDataLens(wrapper);
    }
  }

  function moduleSearchScore(module, query) {
    if (!query.trim()) return Number(loadJson(STORAGE.usage, {})[module.id] || 0);
    const phrase = query.toLowerCase().trim();
    const tokens = phrase.split(/[^a-z0-9가-힣]+/).filter(Boolean);
    const fields = {
      title: String(module.title || '').toLowerCase(),
      description: String(module.description || '').toLowerCase(),
      author: String(module.author || '').toLowerCase(),
      examples: (module.exampleQuestions || []).join(' ').toLowerCase(),
      tags: (module.tags || []).join(' ').toLowerCase(),
      inputs: (module.inputs || []).join(' ').toLowerCase(),
      outputs: (module.outputs || []).join(' ').toLowerCase(),
      assumptions: (module.assumptions || []).join(' ').toLowerCase(),
    };
    let score = 0;
    if (fields.title.includes(phrase)) score += 60;
    if (fields.examples.includes(phrase)) score += 45;
    if (fields.description.includes(phrase)) score += 30;
    for (const token of tokens) {
      if (fields.title.includes(token)) score += 14;
      if (fields.examples.includes(token)) score += 11;
      if (fields.tags.includes(token)) score += 9;
      if (fields.description.includes(token)) score += 6;
      if (fields.inputs.includes(token) || fields.outputs.includes(token)) score += 5;
      if (fields.assumptions.includes(token)) score += 3;
      if (fields.author.includes(token)) score += 2;
    }
    if (compatibility(module).state === 'compatible') score += 4;
    return score;
  }

  function modulesForTab() {
    const user = currentUser();
    const fav = favorites();
    let modules = app.modules;
    if (app.moduleTab === 'my') {
      modules = modules.filter(module => module.kind === 'saved' && (!module.ownerId || module.ownerId === user?.id || module.author === `@${user?.username}`));
    } else if (app.moduleTab === 'favorites') {
      modules = modules.filter(module => fav.has(module.id));
    } else {
      modules = modules.filter(module => module.visibility === 'public' || module.kind === 'builtin' || module.kind === 'registry' || module.kind === 'saved');
    }
    return modules
      .map(module => ({module, score: moduleSearchScore(module, app.search)}))
      .filter(item => !app.search.trim() || item.score > 0)
      .sort((a, b) => b.score - a.score || a.module.title.localeCompare(b.module.title))
      .map(item => item.module);
  }

  function accentColor(name) {
    return ({blue:'#6688b5',violet:'#8473b4',green:'#5e927a',coral:'#c7776f',amber:'#bf8b4e',rose:'#af7393'})[name] || '#6688b5';
  }

  function moduleCardHtml(module) {
    const fav = favorites().has(module.id);
    const compat = compatibility(module);
    const tags = (module.tags || []).slice(0, 3).map(tag => `<span class="product-module-tag">${esc(tag)}</span>`).join('');
    return `<article class="product-module-card" draggable="true" data-product-module="${esc(module.id)}" style="--card-accent:${accentColor(module.accent)}"><div class="product-module-top"><div><div class="product-module-title">${esc(module.title)}</div><div class="product-module-author">${esc(module.author || '@local')} · v${esc(module.version || '0.1.0')}</div></div><button class="product-module-star" type="button" data-star="${esc(module.id)}" title="Favorite">${fav ? '★' : '☆'}</button></div><div class="product-module-description">${esc(module.description || 'No description')}</div><div class="product-module-tags">${tags}</div><div class="compat-text ${compat.state === 'compatible' ? '' : 'mapping'}">${esc(compat.text)}</div><div class="product-module-actions"><button type="button" class="use" data-use="${esc(module.id)}">Use</button><button type="button" class="more" data-more="${esc(module.id)}">•••</button></div></article>`;
  }

  function renderModuleShelf() {
    const panel = document.getElementById('leesinRightModules');
    if (!panel) return;
    const modules = modulesForTab();
    const signature = JSON.stringify({tab: app.moduleTab, search: app.search, ids: modules.map(item => item.id), fav: [...favorites()], cols: activeRun()?.dataColumns || [], user: currentUser()?.id || null});
    if (panel.dataset.productSignature === signature && panel.querySelector('.module-shelf-head')) return;
    panel.dataset.productSignature = signature;
    panel.innerHTML = `<div class="module-shelf-head"><h2>Modules</h2><div class="module-shelf-actions"><button class="ws-icon-btn" id="productImportModule" title="Import Module JSON">⇩</button><button class="ws-icon-btn" id="productNewModule" title="New Module">＋</button></div></div><div class="module-search-wrap"><input id="productModuleSearch" value="${esc(app.search)}" placeholder="What do you want to do?"><span class="module-search-icon">⌕</span></div><div class="module-tabs"><button class="module-tab ${app.moduleTab === 'browse' ? 'active' : ''}" data-module-tab="browse">Browse</button><button class="module-tab ${app.moduleTab === 'my' ? 'active' : ''}" data-module-tab="my">My</button><button class="module-tab ${app.moduleTab === 'favorites' ? 'active' : ''}" data-module-tab="favorites">Favorites</button></div><div class="module-search-note">이름·설명·예시 Question·입력·출력·태그를 함께 검색합니다. GPT discovery는 공개 Registry 이후 연결합니다.</div><div class="module-results">${modules.length ? modules.map(moduleCardHtml).join('') : '<div class="empty-shelf">조건에 맞는 Module이 없습니다.</div>'}</div>`;
    panel.querySelector('#productModuleSearch').oninput = event => {
      app.search = event.target.value;
      renderModuleShelf();
      const input = document.getElementById('productModuleSearch');
      input?.focus();
      input?.setSelectionRange(input.value.length, input.value.length);
    };
    panel.querySelectorAll('[data-module-tab]').forEach(button => button.onclick = () => { app.moduleTab = button.dataset.moduleTab; renderModuleShelf(); });
    panel.querySelectorAll('[data-use]').forEach(button => button.onclick = () => attachModule(button.dataset.use));
    panel.querySelectorAll('[data-star]').forEach(button => button.onclick = () => toggleFavorite(button.dataset.star));
    panel.querySelectorAll('[data-more]').forEach(button => button.onclick = () => openModuleDetails(button.dataset.more));
    panel.querySelectorAll('[data-product-module]').forEach(card => {
      card.ondragstart = event => {
        event.dataTransfer.effectAllowed = 'copy';
        event.dataTransfer.setData('application/x-leesin-module', card.dataset.productModule);
        event.dataTransfer.setData('text/plain', card.dataset.productModule);
      };
    });
    panel.querySelector('#productNewModule').onclick = () => {
      app.centerMode = 'workshop';
      window.openModuleWorkshop?.();
      setTimeout(renderFlowRail, 30);
    };
    panel.querySelector('#productImportModule').onclick = () => {
      app.centerMode = 'workshop';
      window.openModuleWorkshop?.();
      setTimeout(() => document.getElementById('mwImportMoved')?.click(), 50);
    };
  }

  function toggleFavorite(id) {
    const values = favorites();
    if (values.has(id)) values.delete(id); else values.add(id);
    setFavorites(values);
    renderModuleShelf();
  }

  function openModuleDetails(id) {
    const module = moduleById(id);
    if (!module) return;
    const meta = moduleMetaStore()[id] || {};
    const mine = module.kind === 'saved';
    const dialog = ensureDialog('productModuleDialog', '<div></div>');
    dialog.innerHTML = `<div class="product-dialog-body"><div class="product-dialog-head"><div><div class="core-eyebrow">${esc(module.kind)} Module</div><h2>${esc(module.title)}</h2><div class="core-meta">${esc(module.author)} · v${esc(module.version || '0.1.0')}</div></div><button class="ghost" data-close>Close</button></div><p>${esc(module.description || '')}</p><div class="product-dialog-grid"><div class="field"><label>Visibility</label><select id="moduleVisibility" ${mine ? '' : 'disabled'}><option value="private" ${(meta.visibility || module.visibility) === 'private' ? 'selected' : ''}>Private</option><option value="public" ${(meta.visibility || module.visibility) === 'public' ? 'selected' : ''}>Public</option></select></div><div class="field"><label>Author</label><input value="${esc(module.author || '@local')}" disabled></div></div><div class="field"><label>Tags</label><input id="moduleTags" value="${esc((meta.tags || module.tags || []).join(', '))}" ${mine ? '' : 'disabled'}></div><div class="field"><label>Example questions</label><textarea id="moduleExamples" ${mine ? '' : 'disabled'}>${esc((meta.exampleQuestions || module.exampleQuestions || []).join('\n'))}</textarea></div><div class="field"><label>Outputs</label><input id="moduleOutputs" value="${esc((meta.outputs || module.outputs || []).join(', '))}" ${mine ? '' : 'disabled'}></div><div class="core-toolbar" style="justify-content:flex-end"><button class="ghost" id="moduleUseNow">Use</button>${module.code ? `<button class="ghost" id="moduleFork">${mine ? 'Edit code' : 'Fork'}</button>` : ''}${mine ? '<button class="primary" id="moduleMetaSave">Save metadata</button>' : ''}</div></div>`;
    dialog.querySelector('[data-close]').onclick = () => dialog.close();
    dialog.querySelector('#moduleUseNow').onclick = () => { dialog.close(); attachModule(id); };
    const fork = dialog.querySelector('#moduleFork');
    if (fork) fork.onclick = () => {
      if (mine) {
        dialog.close(); app.centerMode = 'workshop'; window.openModuleWorkshop?.(id);
      } else {
        void forkModule(module).then(() => dialog.close());
      }
    };
    const save = dialog.querySelector('#moduleMetaSave');
    if (save) save.onclick = () => {
      const user = currentUser();
      updateModuleMeta(id, {
        ownerId: user?.id || null,
        author: user ? `@${user.username}` : module.author,
        visibility: dialog.querySelector('#moduleVisibility').value,
        tags: dialog.querySelector('#moduleTags').value.split(',').map(value => value.trim()).filter(Boolean),
        exampleQuestions: dialog.querySelector('#moduleExamples').value.split('\n').map(value => value.trim()).filter(Boolean),
        outputs: dialog.querySelector('#moduleOutputs').value.split(',').map(value => value.trim()).filter(Boolean),
      });
      refreshModuleList(); renderModuleShelf(); dialog.close();
    };
    dialog.showModal();
  }

  async function forkModule(module) {
    try {
      const saved = await requestJson('/api/module-workshop/modules', {
        method: 'POST',
        body: JSON.stringify({
          code: module.code,
          functionName: module.entryFunction,
          title: `${module.title} (fork)`,
          description: module.description || '',
          question: (module.exampleQuestions || [])[0] || '',
          assumptions: (module.assumptions || []).join('\n'),
          limits: (module.limits || []).join('\n'),
        }),
      });
      const user = currentUser();
      updateModuleMeta(saved.id, {
        ownerId: user?.id || null,
        author: user ? `@${user.username}` : '@local',
        visibility: 'private',
        tags: module.tags || [],
        exampleQuestions: module.exampleQuestions || [],
        outputs: module.outputs || [],
        forkedFrom: module.id,
      });
      await loadModules();
      await attachModule(saved.id);
      showToast(`${module.title}을 My Modules로 Fork했습니다.`);
    } catch (error) {
      alert(error.message);
    }
  }

  function openRunHistory() {
    const state = activeCore();
    const dialog = ensureDialog('productRunHistoryDialog', '<div></div>');
    const rows = state.completedRuns.map(run => `<div class="run-history-item"><strong>Cycle ${esc(run.cycle)} · ${esc(moduleById(run.module?.id)?.title || run.module?.title || 'No module')}</strong><div class="muted">${esc(run.completedAt || run.updatedAt)} · ${esc(run.result?.statusLabel || run.result?.status || 'No result')}</div>${run.result?.summary ? `<div style="margin-top:5px">${esc(run.result.summary)}</div>` : ''}</div>`).join('') || '<div class="empty-shelf">완료된 Cycle이 아직 없습니다.</div>';
    dialog.innerHTML = `<div class="product-dialog-body"><div class="product-dialog-head"><div><div class="core-eyebrow">Throughout history</div><h2>Run history</h2></div><button class="ghost" data-close>Close</button></div><div class="run-history">${rows}</div></div>`;
    dialog.querySelector('[data-close]').onclick = () => dialog.close();
    dialog.showModal();
  }

  async function refreshContext({render = true} = {}) {
    if (!app.activeProjectId) return;
    const token = ++app.renderToken;
    try {
      const [project, workspace] = await Promise.all([
        requestJson(`/api/projects/${app.activeProjectId}`),
        requestJson(`/api/projects/${app.activeProjectId}/workspace`),
      ]);
      if (token !== app.renderToken) return;
      app.project = project;
      app.workspace = workspace;
      ensureProjectOwner(app.activeProjectId);
      const run = activeRun();
      if (run?.dataRefs?.length && !run.dataColumns?.length) await computeDataColumns(run);
      if (render && app.centerMode === 'core') renderCore();
      renderModuleShelf();
    } catch (error) {
      const main = document.getElementById('mainView');
      if (main && render) main.innerHTML = `<div class="core-shell"><div class="mw-error">${esc(error.message)}</div></div>`;
    }
  }

  function wrapLegacyFunctions() {
    if (window.openProject && !window.openProject.__productWrapped) {
      const original = window.openProject;
      const wrapped = async function(projectId) {
        app.activeProjectId = projectId;
        app.centerMode = 'core';
        const result = await original(projectId);
        await refreshContext({render: true});
        return result;
      };
      wrapped.__productWrapped = true;
      window.openProject = wrapped;
    }

    if (!app.legacy.renderProjectHome && window.renderProjectHome) app.legacy.renderProjectHome = window.renderProjectHome;
    window.renderProjectHome = function() {
      app.centerMode = 'core';
      void refreshContext({render: true});
    };

    for (const [name, mode] of [['renderCluster','legacy'], ['showAnalysis','legacy'], ['renderProposal','legacy']]) {
      const original = window[name];
      if (typeof original === 'function' && !original.__productWrapped) {
        const wrapped = function(...args) {
          app.centerMode = mode;
          const result = original(...args);
          setTimeout(() => { enhanceLegacyDataView(); renderFlowRail(); }, 0);
          return result;
        };
        wrapped.__productWrapped = true;
        window[name] = wrapped;
      }
    }

    if (window.openModuleWorkshop && !window.openModuleWorkshop.__productWrapped) {
      const original = window.openModuleWorkshop;
      const wrapped = async function(...args) {
        app.centerMode = 'workshop';
        const result = await original(...args);
        setTimeout(() => { polishWorkshop(); renderFlowRail(); }, 0);
        return result;
      };
      wrapped.__productWrapped = true;
      window.openModuleWorkshop = wrapped;
    }
  }

  function polishWorkshop() {
    const layout = document.querySelector('#mainView .mw-layout');
    if (!layout) return;
    const aside = layout.querySelector(':scope > aside');
    const importButton = aside?.querySelector('#mwPasteModuleBtn');
    const head = document.querySelector('#mainView .section-head');
    const back = document.getElementById('mwBackBtn');
    if (importButton && head && !document.getElementById('mwImportMoved')) {
      importButton.id = 'mwImportMoved';
      importButton.textContent = 'Import Module';
      if (back) head.insertBefore(importButton, back); else head.appendChild(importButton);
    }
    if (back && !back.dataset.productBound) {
      back.dataset.productBound = '1';
      back.onclick = () => { app.centerMode = 'core'; renderCore(); };
    }
  }

  function detectCurrentView() {
    const main = document.getElementById('mainView');
    if (!main) return;
    const breadcrumb = main.querySelector('.breadcrumb')?.textContent || '';
    if (breadcrumb.includes('Module Workshop')) app.centerMode = 'workshop';
    else if (breadcrumb.includes('Analyses') || breadcrumb.includes('Proposals') || breadcrumb.includes('› Data ›') || breadcrumb.includes('› Files ›')) app.centerMode = 'legacy';
    enhanceLegacyDataView();
    polishWorkshop();
    renderFlowRail();
  }

  function installGlobalClickHooks() {
    document.addEventListener('click', event => {
      if (event.target?.id === 'mwSaveBtn') {
        setTimeout(async () => {
          const before = new Set(app.savedModules.map(module => module.id));
          await loadModules();
          const created = app.savedModules.find(module => !before.has(module.id));
          if (created) {
            const user = currentUser();
            updateModuleMeta(created.id, {ownerId: user?.id || null, author: user ? `@${user.username}` : '@local', visibility: 'private'});
            refreshModuleList();
            renderModuleShelf();
          }
        }, 650);
      }
    }, true);
  }

  function schedulePolish() {
    if (app.observerScheduled) return;
    app.observerScheduled = true;
    requestAnimationFrame(() => {
      app.observerScheduled = false;
      if (app.polishing) return;
      app.polishing = true;
      try {
        stripTopToggles();
        ensureBoundaryControls();
        installAccountButton();
        detectCurrentView();
      } finally {
        app.polishing = false;
      }
    });
  }

  async function initialize() {
    installStyles();
    if (localStorage.getItem(STORAGE.leftOff) === '1') document.body.classList.add('ws-left-off');
    if (localStorage.getItem(STORAGE.rightOff) === '1') document.body.classList.add('ws-right-off');
    stripTopToggles();
    ensureBoundaryControls();
    installAccountButton();
    wrapLegacyFunctions();
    installGlobalClickHooks();
    await loadModules();
    try {
      const bootstrap = await requestJson('/api/bootstrap');
      app.activeProjectId = app.activeProjectId || bootstrap.projects?.[0]?.id || null;
      if (app.activeProjectId) {
        ensureProjectOwner(app.activeProjectId);
        await refreshContext({render: true});
      }
    } catch (error) {
      console.warn('Leesin product shell init failed:', error);
    }
    const observer = new MutationObserver(schedulePolish);
    observer.observe(document.documentElement, {childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class']});
    schedulePolish();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize);
  else void initialize();
})();
