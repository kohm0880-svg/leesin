from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STORE_PATH = Path(
    os.environ.get(
        "LEESIN_V4_STORE",
        str(Path(__file__).resolve().parent / "runtime" / "store.json"),
    )
)
_LOCK = threading.RLock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _empty_state() -> dict[str, Any]:
    return {
        "projects": {},
        "clusters": {},
        "analyses": {},
        "proposals": {},
    }


def _ensure_seed_project(state: dict[str, Any]) -> None:
    if state["projects"]:
        return
    project_id = "project_prime_mvp"
    state["projects"][project_id] = {
        "id": project_id,
        "title": "Prime Algorithm Benchmark",
        "description": (
            "Leesin_V4 MVP: selected data + explicit question + assumptions/limits + next observation."
        ),
        "createdAt": utc_now(),
    }


def _read_state() -> dict[str, Any]:
    with _LOCK:
        if not STORE_PATH.exists():
            state = _empty_state()
            _ensure_seed_project(state)
            _write_state(state)
            return state
        try:
            state = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = _empty_state()
        for key in ("projects", "clusters", "analyses", "proposals"):
            if not isinstance(state.get(key), dict):
                state[key] = {}
        # Seed only when the store is created for the first time. If a user
        # intentionally deletes every project, keep the store empty.
        return state


def _write_state(state: dict[str, Any]) -> None:
    with _LOCK:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, ensure_ascii=False, indent=2)
        fd, temp_name = tempfile.mkstemp(
            prefix="store-",
            suffix=".json",
            dir=str(STORE_PATH.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(temp_name, STORE_PATH)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def list_projects() -> list[dict[str, Any]]:
    state = _read_state()
    result: list[dict[str, Any]] = []
    for project in state["projects"].values():
        project_id = project["id"]
        result.append(
            {
                **project,
                "clusterCount": sum(
                    1 for item in state["clusters"].values() if item["projectId"] == project_id
                ),
                "analysisCount": sum(
                    1 for item in state["analyses"].values() if item["projectId"] == project_id
                ),
                "proposalCount": sum(
                    1 for item in state["proposals"].values() if item["projectId"] == project_id
                ),
            }
        )
    return sorted(result, key=lambda item: item["createdAt"])


def create_project(title: str, description: str = "") -> dict[str, Any]:
    clean_title = str(title or "").strip()
    if not clean_title:
        raise ValueError("Project title is required.")
    with _LOCK:
        state = _read_state()
        project_id = _new_id("project")
        project = {
            "id": project_id,
            "title": clean_title,
            "description": str(description or "").strip(),
            "createdAt": utc_now(),
        }
        state["projects"][project_id] = project
        _write_state(state)
        return dict(project)


def update_project(
    project_id: str,
    *,
    title: str,
    description: str | None = None,
) -> dict[str, Any]:
    clean_title = str(title or "").strip()
    if not clean_title:
        raise ValueError("Project title is required.")
    with _LOCK:
        state = _read_state()
        project = _require_project(state, project_id)
        project["title"] = clean_title
        if description is not None:
            project["description"] = str(description or "").strip()
        project["updatedAt"] = utc_now()
        _write_state(state)
        return dict(project)


def delete_project(project_id: str) -> dict[str, Any]:
    """Permanently remove one project and every object owned by it."""
    with _LOCK:
        state = _read_state()
        project = dict(_require_project(state, project_id))

        removed = {
            "clusters": 0,
            "analyses": 0,
            "proposals": 0,
            "files": 0,
            "folders": 0,
            "trash": 0,
        }

        for store_name, counter in (
            ("clusters", "clusters"),
            ("analyses", "analyses"),
            ("proposals", "proposals"),
            ("workspace_files", "files"),
            ("workspace_folders", "folders"),
            ("workspace_trash", "trash"),
        ):
            items = state.get(store_name)
            if not isinstance(items, dict):
                continue
            for item_id, item in list(items.items()):
                if isinstance(item, dict) and item.get("projectId") == project_id:
                    items.pop(item_id, None)
                    removed[counter] += 1

        state["projects"].pop(project_id, None)
        _write_state(state)
        return {"deletedProject": project, "removed": removed}


def _require_project(state: dict[str, Any], project_id: str) -> dict[str, Any]:
    project = state["projects"].get(project_id)
    if not project:
        raise KeyError(f"Unknown project: {project_id}")
    return project


def project_detail(project_id: str) -> dict[str, Any]:
    state = _read_state()
    project = _require_project(state, project_id)
    clusters = [
        dict(item)
        for item in state["clusters"].values()
        if item["projectId"] == project_id
    ]
    analyses = [
        dict(item)
        for item in state["analyses"].values()
        if item["projectId"] == project_id
    ]
    proposals = [
        dict(item)
        for item in state["proposals"].values()
        if item["projectId"] == project_id
    ]
    clusters.sort(key=lambda item: item["createdAt"])
    analyses.sort(key=lambda item: item["createdAt"], reverse=True)
    proposals.sort(key=lambda item: item["createdAt"], reverse=True)
    return {
        **project,
        "clusters": clusters,
        "analyses": analyses,
        "proposals": proposals,
    }


def add_cluster(
    project_id: str,
    *,
    name: str,
    filename: str,
    csv_text: str,
    protocol: str = "",
    context: str = "",
    origin_proposal_id: str | None = None,
) -> dict[str, Any]:
    if not str(csv_text or "").strip():
        raise ValueError("CSV content is required.")
    with _LOCK:
        state = _read_state()
        _require_project(state, project_id)
        cluster_id = _new_id("cluster")
        content_hash = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
        cluster = {
            "id": cluster_id,
            "projectId": project_id,
            "name": str(name or "").strip() or str(filename or "").strip() or cluster_id,
            "filename": str(filename or "").strip() or "data.csv",
            "csvText": csv_text,
            "contentHash": content_hash,
            "protocol": str(protocol or "").strip(),
            "context": str(context or "").strip(),
            "originProposalId": origin_proposal_id,
            "createdAt": utc_now(),
        }
        state["clusters"][cluster_id] = cluster
        if origin_proposal_id and origin_proposal_id in state["proposals"]:
            state["proposals"][origin_proposal_id]["status"] = "EXECUTED"
            state["proposals"][origin_proposal_id]["executedClusterId"] = cluster_id
        _write_state(state)
        return dict(cluster)


def selected_clusters(project_id: str, cluster_ids: list[str]) -> list[dict[str, Any]]:
    state = _read_state()
    _require_project(state, project_id)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cluster_id in cluster_ids:
        if cluster_id in seen:
            continue
        seen.add(cluster_id)
        cluster = state["clusters"].get(cluster_id)
        if not cluster or cluster["projectId"] != project_id:
            raise ValueError(f"Cluster is not in this project: {cluster_id}")
        result.append(dict(cluster))
    return result


def save_analysis(
    project_id: str,
    *,
    question_id: str,
    cluster_ids: list[str],
    module_version: str,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    with _LOCK:
        state = _read_state()
        _require_project(state, project_id)
        analysis_id = _new_id("analysis")
        snapshots: list[dict[str, Any]] = []
        for cluster_id in cluster_ids:
            cluster = state["clusters"].get(cluster_id)
            if not cluster or cluster["projectId"] != project_id:
                raise ValueError(f"Cluster is not in this project: {cluster_id}")
            snapshots.append(
                {
                    "clusterId": cluster_id,
                    "name": cluster["name"],
                    "filename": cluster["filename"],
                    "contentHash": cluster["contentHash"],
                    "protocol": cluster.get("protocol", ""),
                    "context": cluster.get("context", ""),
                }
            )

        analysis = {
            "id": analysis_id,
            "projectId": project_id,
            "questionId": question_id,
            "clusterIds": list(cluster_ids),
            "clusterSnapshots": snapshots,
            "moduleVersion": module_version,
            "status": outcome.get("status"),
            "outcome": outcome,
            "createdAt": utc_now(),
        }
        state["analyses"][analysis_id] = analysis

        proposal_payload = outcome.get("proposal")
        proposal: dict[str, Any] | None = None
        if isinstance(proposal_payload, dict):
            proposal_id = _new_id("proposal")
            proposal = {
                "id": proposal_id,
                "projectId": project_id,
                "parentAnalysisId": analysis_id,
                "status": "PROPOSED",
                "payload": proposal_payload,
                "createdAt": utc_now(),
            }
            state["proposals"][proposal_id] = proposal
            analysis["proposalId"] = proposal_id
        _write_state(state)
        if proposal:
            analysis = {**analysis, "proposal": proposal}
        return analysis


def start_proposal(project_id: str, proposal_id: str) -> dict[str, Any]:
    with _LOCK:
        state = _read_state()
        _require_project(state, project_id)
        proposal = state["proposals"].get(proposal_id)
        if not proposal or proposal["projectId"] != project_id:
            raise KeyError(f"Unknown proposal: {proposal_id}")
        if proposal.get("status") == "PROPOSED":
            proposal["status"] = "STARTED"
            proposal["startedAt"] = utc_now()
        _write_state(state)
        return dict(proposal)
