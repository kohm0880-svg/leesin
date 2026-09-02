(() => {
  'use strict';

  async function requestJson(path, options = {}) {
    const response = await fetch(path, {
      headers: {'Content-Type': 'application/json', ...(options.headers || {})},
      ...options,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    return data;
  }

  function activeProjectId() {
    const title = document.querySelector('#mainView .core-title');
    if (!title) return window.state?.activeProjectId || window.state?.project?.id || null;
    const cards = [...document.querySelectorAll('#projectList .project-card')];
    const card = cards.find(item => item.querySelector('strong')?.textContent === title.textContent);
    if (card?.dataset.projectId) return card.dataset.projectId;
    if (card) {
      const handler = card.getAttribute('onclick') || '';
      const match = handler.match(/openProject\(['\"]([^'\"]+)/);
      if (match) return match[1];
    }
    return window.state?.activeProjectId || window.state?.project?.id || null;
  }

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
  }

  function installStyles() {
    if (document.getElementById('projectControlsStyles')) return;
    const style = document.createElement('style');
    style.id = 'projectControlsStyles';
    style.textContent = `
      #projectList .project-card{position:relative;padding-right:42px}
      .project-card-settings{position:absolute;right:8px;top:8px;width:27px;height:27px;border:1px solid #d2dce2;border-radius:9px;background:rgba(250,252,252,.92);color:#667680;font-weight:900;line-height:1;padding:0;display:flex;align-items:center;justify-content:center}
      .project-card-settings:hover{background:#eaf1f3;border-color:#aec1c9;color:#415a65}
      .project-card.current-project{border-color:#aebfd2!important;box-shadow:inset 3px 0 0 #718fb8,0 5px 16px rgba(58,76,87,.05)!important}
    `;
    document.head.appendChild(style);
  }

  function ensureDialog() {
    let dialog = document.getElementById('projectSettingsDialog');
    if (!dialog) {
      dialog = document.createElement('dialog');
      dialog.id = 'projectSettingsDialog';
      dialog.className = 'product-dialog';
      document.body.appendChild(dialog);
    }
    return dialog;
  }

  async function openProjectSettings(projectId = activeProjectId()) {
    if (!projectId) return;
    let project;
    try {
      project = await requestJson(`/api/projects/${projectId}`);
    } catch (error) {
      alert(error.message);
      return;
    }

    const dialog = ensureDialog();
    dialog.innerHTML = `
      <div class="product-dialog-body">
        <div class="product-dialog-head">
          <div><div class="core-eyebrow">Project settings</div><h2>${esc(project.title)}</h2></div>
          <button class="ghost" data-close>Close</button>
        </div>
        <div class="field"><label>Project name</label><input id="projectSettingsTitle" value="${esc(project.title)}"></div>
        <div class="field"><label>Description</label><textarea id="projectSettingsDescription">${esc(project.description || '')}</textarea></div>
        <div class="core-toolbar" style="justify-content:space-between;margin-top:18px">
          <button class="ghost" id="projectDeleteBtn" style="color:#9a4f55;border-color:#dfbfc2">Delete project</button>
          <button class="primary" id="projectSaveBtn">Save</button>
        </div>
        <div class="core-note">Project 삭제는 이 Project의 Data, Analysis, Proposal, Files와 Trash를 함께 영구 삭제합니다.</div>
      </div>`;

    dialog.querySelector('[data-close]').onclick = () => dialog.close();
    dialog.querySelector('#projectSaveBtn').onclick = async () => {
      const title = dialog.querySelector('#projectSettingsTitle').value.trim();
      const description = dialog.querySelector('#projectSettingsDescription').value;
      if (!title) {
        alert('Project name을 입력하세요.');
        return;
      }
      try {
        await requestJson(`/api/projects/${projectId}/settings`, {
          method: 'POST',
          body: JSON.stringify({title, description}),
        });
        dialog.close();
        location.reload();
      } catch (error) {
        alert(error.message);
      }
    };

    dialog.querySelector('#projectDeleteBtn').onclick = async () => {
      const typed = prompt(`영구 삭제하려면 Project 이름을 그대로 입력하세요.\n\n${project.title}`);
      if (typed !== project.title) {
        if (typed !== null) alert('Project 이름이 일치하지 않아 삭제하지 않았습니다.');
        return;
      }
      try {
        await requestJson(`/api/projects/${projectId}/delete`, {method: 'POST', body: '{}'});
        for (const key of ['leesin.coreState.v3', 'leesin.projectMeta.v1']) {
          try {
            const data = JSON.parse(localStorage.getItem(key) || '{}');
            delete data[projectId];
            localStorage.setItem(key, JSON.stringify(data));
          } catch (_) {}
        }
        dialog.close();
        location.reload();
      } catch (error) {
        alert(error.message);
      }
    };

    dialog.showModal();
  }

  function projectIdFromCard(card) {
    if (card.dataset.projectId) return card.dataset.projectId;
    const handler = card.getAttribute('onclick') || '';
    const match = handler.match(/openProject\(['\"]([^'\"]+)/);
    if (!match) return null;
    card.dataset.projectId = match[1];
    return match[1];
  }

  function installProjectSidebarControls() {
    document.getElementById('coreProjectSettingsBtn')?.remove();
    const activeId = activeProjectId();
    document.querySelectorAll('#projectList .project-card').forEach(card => {
      const projectId = projectIdFromCard(card);
      if (!projectId) return;
      card.classList.toggle('current-project', projectId === activeId);
      let button = card.querySelector('.project-card-settings');
      if (!button) {
        button = document.createElement('button');
        button.type = 'button';
        button.className = 'project-card-settings';
        button.textContent = '⋯';
        button.title = 'Project settings';
        button.setAttribute('aria-label', 'Project settings');
        button.addEventListener('click', event => {
          event.preventDefault();
          event.stopPropagation();
          void openProjectSettings(projectId);
        });
        card.appendChild(button);
      }
    });
  }

  function selectedRenameRow() {
    const rows = [...document.querySelectorAll('#projectTree [data-ws-key].selected')]
      .filter(row => !row.dataset.trashId && ['file', 'folder'].includes(row.dataset.type));
    return rows.length === 1 ? rows[0] : null;
  }

  async function renameSelectedExplorerItem() {
    const row = selectedRenameRow();
    const projectId = activeProjectId();
    if (!row || !projectId) return;
    const oldName = row.querySelector('.ws-label')?.textContent || '';
    const name = prompt('Rename', oldName);
    if (!name?.trim() || name.trim() === oldName) return;
    try {
      await requestJson(`/api/projects/${projectId}/workspace/rename`, {
        method: 'POST',
        body: JSON.stringify({type: row.dataset.type, id: row.dataset.id, name: name.trim()}),
      });
      window.renderTree?.();
    } catch (error) {
      alert(error.message);
    }
  }

  function installRenameButton() {
    const toolbar = document.querySelector('#projectTree .ws-toolbar');
    if (!toolbar) return;
    let button = document.getElementById('wsRenameVisible');
    if (!button) {
      button = document.createElement('button');
      button.type = 'button';
      button.className = 'ws-icon-btn';
      button.id = 'wsRenameVisible';
      button.textContent = 'Rename';
      button.title = 'Select exactly one file or folder. F2 also works.';
      button.onclick = renameSelectedExplorerItem;
      const deleteButton = document.getElementById('wsDeleteSelected');
      toolbar.insertBefore(button, deleteButton || null);
    }
    button.disabled = !selectedRenameRow();
  }

  function installF2() {
    if (document.documentElement.dataset.projectControlsF2) return;
    document.documentElement.dataset.projectControlsF2 = '1';
    document.addEventListener('keydown', event => {
      if (event.key !== 'F2') return;
      if (!selectedRenameRow()) return;
      event.preventDefault();
      void renameSelectedExplorerItem();
    });
  }

  let scheduled = false;
  function patch() {
    installStyles();
    installProjectSidebarControls();
    installRenameButton();
    installF2();
  }

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
      observer.observe(document.documentElement, {childList: true, subtree: true, attributes: true, attributeFilter: ['class']});
    });
  } else {
    patch();
    observer.observe(document.documentElement, {childList: true, subtree: true, attributes: true, attributeFilter: ['class']});
  }
})();