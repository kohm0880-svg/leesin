(() => {
  // Temporary MVP-only UI. Delete v4_mvp/mvp_adapters and the tiny app.py
  // injection/endpoint when built-in experiment execution is no longer needed.
  async function runPrimeMvp(nValues, originProposalId = null) {
    if (!state?.project?.id) {
      alert('프로젝트를 먼저 선택하세요.');
      return;
    }
    const normalized = nValues.map(Number).filter(Number.isFinite);
    if (!normalized.length) {
      alert('N 값을 하나 이상 입력하세요.');
      return;
    }
    try {
      const result = await api(`/api/projects/${state.project.id}/mvp/prime-benchmark`, {
        method: 'POST',
        body: JSON.stringify({nValues: normalized, originProposalId})
      });
      state.project = await api(`/api/projects/${state.project.id}`);
      renderTree();
      renderProjectHome();
      alert(`실험 완료: ${result.cluster.name}\n${result.rowCount} rows saved.`);
    } catch (err) {
      alert(err.message);
    }
  }

  window.runPrimeMvpInitial = async function () {
    const raw = prompt('실험할 N 값을 쉼표로 입력하세요.', '1000,100000');
    if (raw == null) return;
    const values = raw.split(',').map(v => Number(v.trim())).filter(Number.isFinite);
    await runPrimeMvp(values);
  };

  const originalRenderProjectHome = renderProjectHome;
  renderProjectHome = function () {
    originalRenderProjectHome();
    const header = document.querySelector('#mainView .section-head');
    if (!header || document.getElementById('runPrimeMvpBtn')) return;
    const button = document.createElement('button');
    button.className = 'ghost';
    button.id = 'runPrimeMvpBtn';
    button.textContent = '⚗ Run MVP experiment';
    button.title = 'MVP-only: prime benchmark를 실행하고 새 Data Cluster를 자동 생성합니다.';
    button.onclick = window.runPrimeMvpInitial;
    header.appendChild(button);
  };

  // For this MVP only, "Start next experiment" actually executes the benchmark
  // and creates the next cluster instead of opening the manual CSV upload dialog.
  startProposal = async function (proposalId) {
    try {
      const proposal = await api(`/api/projects/${state.project.id}/proposals/${proposalId}/start`, {
        method: 'POST',
        body: '{}'
      });
      const n = Number(proposal.payload?.input?.N);
      if (!Number.isFinite(n)) {
        alert('이 Proposal에는 MVP adapter가 실행할 N 값이 없습니다.');
        return;
      }
      await runPrimeMvp([n], proposalId);
    } catch (err) {
      alert(err.message);
    }
  };

  // Initial project rendering may finish before/after this adapter loads.
  setTimeout(() => {
    if (state?.project) renderProjectHome();
  }, 50);
})();
