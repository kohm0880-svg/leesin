from __future__ import annotations

import base64
import hashlib
from typing import Any

from v4_mvp import store as core_store


WORKSPACE_KEYS = ("workspace_files", "workspace_folders", "workspace_trash")
LEGACY_STORES = {
    "cluster": "clusters",
    "analysis": "analyses",
    "proposal": "proposals",
}


def _ensure_workspace(state: dict[str, Any]) -> None:
    for key in WORKSPACE_KEYS:
        if not isinstance(state.get(key), dict):
            state[key] = {}


def _require_project(state: dict[str, Any], project_id: str) -> dict[str, Any]:
    project = state.get("projects", {}).get(project_id)
    if not project:
        raise KeyError(f"Unknown project: {project_id}")
    return project


def _public_file(item: dict[str, Any], *, include_content: bool = False) -> dict[str, Any]:
    result = {
        key: value
        for key, value in item.items()
        if key not in {"contentBase64", "textContent"}
    }
    if include_content:
        result["contentBase64"] = item.get("contentBase64", "")
        result["textContent"] = item.get("textContent")
    return result


def _trash_entry(entry: dict[str, Any]) -> dict[str, Any]:
    payload = entry.get("payload") or {}
    return {
        "trashId": entry["id"],
        "type": entry.get("type"),
        "id": payload.get("id"),
        "name": (
            payload.get("name")
            or payload.get("title")
            or payload.get("filename")
            or payload.get("id")
            or entry.get("type")
        ),
        "deletedAt": entry.get("deletedAt"),
        "parentFolderId": payload.get("parentFolderId"),
    }


def workspace_detail(project_id: str) -> dict[str, Any]:
    state = core_store._read_state()
    _ensure_workspace(state)
    _require_project(state, project_id)
    files = [
        _public_file(item)
        for item in state["workspace_files"].values()
        if item.get("projectId") == project_id
    ]
    folders = [
        dict(item)
        for item in state["workspace_folders"].values()
        if item.get("projectId") == project_id
    ]
    trash = [
        _trash_entry(entry)
        for entry in state["workspace_trash"].values()
        if entry.get("projectId") == project_id
    ]
    files.sort(key=lambda item: (str(item.get("name") or "").lower(), item.get("createdAt", "")))
    folders.sort(key=lambda item: (str(item.get("name") or "").lower(), item.get("createdAt", "")))
    trash.sort(key=lambda item: str(item.get("deletedAt") or ""), reverse=True)
    return {"files": files, "folders": folders, "trash": trash}


def create_folder(
    project_id: str,
    *,
    name: str,
    parent_folder_id: str | None = None,
) -> dict[str, Any]:
    clean = str(name or "").strip()
    if not clean:
        raise ValueError("Folder name is required.")
    with core_store._LOCK:
        state = core_store._read_state()
        _ensure_workspace(state)
        _require_project(state, project_id)
        if parent_folder_id:
            parent = state["workspace_folders"].get(parent_folder_id)
            if not parent or parent.get("projectId") != project_id:
                raise ValueError("Parent folder is not available in this project.")
        folder_id = core_store._new_id("folder")
        folder = {
            "id": folder_id,
            "projectId": project_id,
            "name": clean,
            "parentFolderId": parent_folder_id,
            "createdAt": core_store.utc_now(),
        }
        state["workspace_folders"][folder_id] = folder
        core_store._write_state(state)
        return dict(folder)


def add_project_file(
    project_id: str,
    *,
    name: str,
    content_base64: str,
    mime_type: str = "",
    size: int | None = None,
    text_content: str | None = None,
    parent_folder_id: str | None = None,
) -> dict[str, Any]:
    clean = str(name or "").strip()
    if not clean:
        raise ValueError("File name is required.")
    try:
        raw = base64.b64decode(str(content_base64 or ""), validate=True)
    except Exception as exc:
        raise ValueError("File content is not valid base64.") from exc
    if size is not None and int(size) != len(raw):
        raise ValueError("Uploaded file size does not match its content.")
    with core_store._LOCK:
        state = core_store._read_state()
        _ensure_workspace(state)
        _require_project(state, project_id)
        if parent_folder_id:
            parent = state["workspace_folders"].get(parent_folder_id)
            if not parent or parent.get("projectId") != project_id:
                raise ValueError("Parent folder is not available in this project.")
        file_id = core_store._new_id("file")
        item = {
            "id": file_id,
            "projectId": project_id,
            "name": clean,
            "mimeType": str(mime_type or ""),
            "size": len(raw),
            "contentHash": hashlib.sha256(raw).hexdigest(),
            "contentBase64": str(content_base64 or ""),
            "textContent": text_content,
            "parentFolderId": parent_folder_id,
            "createdAt": core_store.utc_now(),
        }
        state["workspace_files"][file_id] = item
        core_store._write_state(state)
        return _public_file(item)


def get_project_file(project_id: str, file_id: str) -> dict[str, Any]:
    state = core_store._read_state()
    _ensure_workspace(state)
    _require_project(state, project_id)
    item = state["workspace_files"].get(file_id)
    if not item or item.get("projectId") != project_id:
        raise KeyError(f"Unknown project file: {file_id}")
    return _public_file(item, include_content=True)


def _descendant_folder_ids(
    state: dict[str, Any], project_id: str, folder_id: str
) -> set[str]:
    result = {folder_id}
    changed = True
    while changed:
        changed = False
        for folder in state["workspace_folders"].values():
            if (
                folder.get("projectId") == project_id
                and folder.get("parentFolderId") in result
                and folder["id"] not in result
            ):
                result.add(folder["id"])
                changed = True
    return result


def _put_trash(
    state: dict[str, Any], project_id: str, item_type: str, payload: dict[str, Any]
) -> str:
    trash_id = core_store._new_id("trash")
    state["workspace_trash"][trash_id] = {
        "id": trash_id,
        "projectId": project_id,
        "type": item_type,
        "payload": payload,
        "deletedAt": core_store.utc_now(),
    }
    return trash_id


def trash_items(project_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    moved: list[dict[str, Any]] = []
    with core_store._LOCK:
        state = core_store._read_state()
        _ensure_workspace(state)
        _require_project(state, project_id)
        seen: set[tuple[str, str]] = set()
        for ref in items:
            item_type = str(ref.get("type") or "")
            item_id = str(ref.get("id") or "")
            if not item_id or (item_type, item_id) in seen:
                continue
            seen.add((item_type, item_id))

            if item_type == "folder":
                folder = state["workspace_folders"].get(item_id)
                if not folder or folder.get("projectId") != project_id:
                    continue
                folder_ids = _descendant_folder_ids(state, project_id, item_id)
                for file_id, file_item in list(state["workspace_files"].items()):
                    if (
                        file_item.get("projectId") == project_id
                        and file_item.get("parentFolderId") in folder_ids
                    ):
                        payload = state["workspace_files"].pop(file_id)
                        trash_id = _put_trash(state, project_id, "file", payload)
                        moved.append({"type": "file", "id": file_id, "trashId": trash_id})
                for folder_id in sorted(folder_ids, reverse=True):
                    payload = state["workspace_folders"].pop(folder_id, None)
                    if payload:
                        trash_id = _put_trash(state, project_id, "folder", payload)
                        moved.append({"type": "folder", "id": folder_id, "trashId": trash_id})
                continue

            if item_type == "file":
                payload = state["workspace_files"].pop(item_id, None)
                if payload and payload.get("projectId") == project_id:
                    trash_id = _put_trash(state, project_id, "file", payload)
                    moved.append({"type": "file", "id": item_id, "trashId": trash_id})
                continue

            store_name = LEGACY_STORES.get(item_type)
            if not store_name:
                continue
            payload = state[store_name].pop(item_id, None)
            if not payload or payload.get("projectId") != project_id:
                if payload:
                    state[store_name][item_id] = payload
                continue
            trash_id = _put_trash(state, project_id, item_type, payload)
            moved.append({"type": item_type, "id": item_id, "trashId": trash_id})
            if item_type == "analysis":
                for proposal_id, proposal in list(state["proposals"].items()):
                    if (
                        proposal.get("projectId") == project_id
                        and proposal.get("parentAnalysisId") == item_id
                    ):
                        proposal_payload = state["proposals"].pop(proposal_id)
                        proposal_trash = _put_trash(
                            state, project_id, "proposal", proposal_payload
                        )
                        moved.append(
                            {"type": "proposal", "id": proposal_id, "trashId": proposal_trash}
                        )
        core_store._write_state(state)
    return {"trashed": moved}


def restore_items(project_id: str, trash_ids: list[str]) -> dict[str, Any]:
    restored: list[dict[str, Any]] = []
    with core_store._LOCK:
        state = core_store._read_state()
        _ensure_workspace(state)
        _require_project(state, project_id)
        for trash_id in trash_ids:
            entry = state["workspace_trash"].get(str(trash_id))
            if not entry or entry.get("projectId") != project_id:
                continue
            item_type = str(entry.get("type") or "")
            payload = dict(entry.get("payload") or {})
            item_id = str(payload.get("id") or "")
            if not item_id:
                continue
            if item_type == "file":
                state["workspace_files"][item_id] = payload
            elif item_type == "folder":
                parent_id = payload.get("parentFolderId")
                if parent_id and parent_id not in state["workspace_folders"]:
                    payload["parentFolderId"] = None
                state["workspace_folders"][item_id] = payload
            else:
                store_name = LEGACY_STORES.get(item_type)
                if not store_name:
                    continue
                state[store_name][item_id] = payload
            state["workspace_trash"].pop(str(trash_id), None)
            restored.append({"type": item_type, "id": item_id})
        core_store._write_state(state)
    return {"restored": restored}


def purge_trash(project_id: str, trash_ids: list[str]) -> dict[str, Any]:
    purged: list[str] = []
    with core_store._LOCK:
        state = core_store._read_state()
        _ensure_workspace(state)
        _require_project(state, project_id)
        for trash_id in trash_ids:
            entry = state["workspace_trash"].get(str(trash_id))
            if entry and entry.get("projectId") == project_id:
                state["workspace_trash"].pop(str(trash_id), None)
                purged.append(str(trash_id))
        core_store._write_state(state)
    return {"purged": purged}


def rename_item(
    project_id: str, *, item_type: str, item_id: str, name: str
) -> dict[str, Any]:
    clean = str(name or "").strip()
    if not clean:
        raise ValueError("Name is required.")
    store_name = "workspace_files" if item_type == "file" else "workspace_folders" if item_type == "folder" else None
    if not store_name:
        raise ValueError("Only project files and folders can be renamed in this MVP.")
    with core_store._LOCK:
        state = core_store._read_state()
        _ensure_workspace(state)
        _require_project(state, project_id)
        item = state[store_name].get(item_id)
        if not item or item.get("projectId") != project_id:
            raise KeyError(f"Unknown {item_type}: {item_id}")
        item["name"] = clean
        item["updatedAt"] = core_store.utc_now()
        core_store._write_state(state)
        return _public_file(item) if item_type == "file" else dict(item)


def move_items(
    project_id: str,
    *,
    items: list[dict[str, Any]],
    parent_folder_id: str | None,
) -> dict[str, Any]:
    moved: list[dict[str, Any]] = []
    with core_store._LOCK:
        state = core_store._read_state()
        _ensure_workspace(state)
        _require_project(state, project_id)
        if parent_folder_id:
            parent = state["workspace_folders"].get(parent_folder_id)
            if not parent or parent.get("projectId") != project_id:
                raise ValueError("Target folder is not available.")
        for ref in items:
            item_type = str(ref.get("type") or "")
            item_id = str(ref.get("id") or "")
            store_name = "workspace_files" if item_type == "file" else "workspace_folders" if item_type == "folder" else None
            if not store_name:
                continue
            item = state[store_name].get(item_id)
            if not item or item.get("projectId") != project_id:
                continue
            if item_type == "folder":
                descendants = _descendant_folder_ids(state, project_id, item_id)
                if parent_folder_id in descendants:
                    raise ValueError("A folder cannot be moved inside itself or its descendant.")
            item["parentFolderId"] = parent_folder_id
            item["updatedAt"] = core_store.utc_now()
            moved.append({"type": item_type, "id": item_id})
        core_store._write_state(state)
    return {"moved": moved}
