(() => {
  'use strict';

  let projectRefreshTimer = null;

  function normalizeFlowRail() {
    const analysisStep = document.querySelector('.product-flow-step[data-step="analysis"]');
    if (!analysisStep || analysisStep.classList.contains('active')) return;
    analysisStep.classList.remove('reached');
    analysisStep.disabled = true;
    analysisStep.title = 'Analysis는 실행 상태입니다. 다시 실행하려면 Mapping 단계로 돌아가세요.';
  }

  function refreshCoreAfterExplorerChange() {
    clearTimeout(projectRefreshTimer);
    projectRefreshTimer = setTimeout(() => {
      if (!document.querySelector('#mainView .core-shell')) return;
      if (typeof window.renderProjectHome === 'function') window.renderProjectHome();
    }, 140);
  }

  function watchProjectTree() {
    const tree = document.getElementById('projectTree');
    if (!tree || tree.dataset.productRefreshObserved) return;
    tree.dataset.productRefreshObserved = '1';
    const observer = new MutationObserver(refreshCoreAfterExplorerChange);
    observer.observe(tree, {childList: true, subtree: true});
  }

  function patch() {
    normalizeFlowRail();
    watchProjectTree();
  }

  let scheduled = false;
  const observer = new MutationObserver(() => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      patch();
    });
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      patch();
      observer.observe(document.documentElement, {childList: true, subtree: true});
    });
  } else {
    patch();
    observer.observe(document.documentElement, {childList: true, subtree: true});
  }
})();
