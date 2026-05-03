from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from models import K_M, DiagnosisResult, ExperimentConfig
from stats_engine import DataQualityAnalyzer
from storage import (
    CLUSTER_STORE_PATH,
    GOAL_STORE_PATH,
    STORE_DIR,
    axis_subset_key,
    delete_peer_cluster,
    get_peer_group,
    init_database,
    list_cluster_summaries,
    load_cluster_store,
    load_goal_store,
    load_peer_clusters,
    normalize_goals_for_display,
    peer_group_key,
    peer_group_subset_counts,
    save_data_cluster,
    save_goal_store,
    should_save_data_clusters,
    storage_label,
    use_demo_peer_group,
    validate_goal,
)


TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


def goal_subset(goal: dict[str, Any], selected_axis_names: list[str] | None = None) -> dict[str, Any]:
    if not selected_axis_names:
        axes = [dict(axis) for axis in goal["axes"]]
    else:
        requested = {str(name).strip() for name in selected_axis_names if str(name).strip()}
        axes = [dict(axis) for axis in goal["axes"] if axis["name"] in requested]
    if not axes:
        raise ValueError("분석에 포함할 Axis를 하나 이상 선택하세요.")
    return {
        "id": goal["id"],
        "name": goal["name"],
        "K_m": float(goal.get("K_m", K_M)),
        "axes": axes,
    }


def build_cluster_vector(
    rows: list[dict[str, Any]],
    axis_mapping: dict[str, str],
    selected_goal: dict[str, Any],
    method: str = "mean",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Summarize one CSV file into one cluster vector.

    CSV file = one data cluster. Rows inside the CSV are repeated observations.
    Current method is streaming axis-wise mean; the method argument keeps room for
    future median/std/IQR summaries without changing the caller contract.
    """
    if method != "mean":
        raise ValueError("Only mean cluster summarization is currently supported.")
    if not rows:
        raise ValueError("업로드된 데이터가 비어 있습니다.")

    axes = selected_goal["axes"]
    means = np.zeros(len(axes), dtype=float)
    counts = np.zeros(len(axes), dtype=int)
    columns = list(rows[0].keys()) if rows else []

    for csv_row_number, row in enumerate(rows, start=2):
        for axis_index, axis in enumerate(axes):
            axis_name = axis["name"]
            column = axis_mapping.get(axis_name)
            if not column:
                raise ValueError(f"Axis '{axis_name}'에 매핑된 CSV column이 없습니다.")
            raw_value = row.get(column, "")
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Row {csv_row_number}, axis '{axis_name}', column '{column}' has non-numeric value: {raw_value!r}"
                ) from exc
            counts[axis_index] += 1
            means[axis_index] += (numeric - means[axis_index]) / counts[axis_index]

    for axis_index, axis in enumerate(axes):
        if counts[axis_index] == 0:
            raise ValueError(f"Axis '{axis['name']}'에 사용할 수 있는 numeric row가 없습니다.")

    return means, {
        "row_count": len(rows),
        "columns": columns,
        "summary_method": method,
        "cluster_definition": "CSV file = one cluster; CSV rows = repeated observations summarized into one cluster vector.",
    }


build_target_vector = build_cluster_vector


def build_axis_distribution(values: np.ndarray, domain_min: float, domain_max: float, resolution: float) -> dict[str, Any]:
    total_bins = max(1, int(np.ceil((domain_max - domain_min) / resolution)))
    counts = [0 for _ in range(total_bins)]
    for value in values:
        if not np.isfinite(value):
            continue
        clipped = min(max(float(value), domain_min), domain_max - np.finfo(float).eps)
        index = int(np.floor((clipped - domain_min) / resolution))
        counts[max(0, min(total_bins - 1, index))] += 1
    occupied = sum(1 for count in counts if count > 0)
    return {
        "totalBins": total_bins,
        "occupiedBins": occupied,
        "coverage": occupied / total_bins if total_bins else 0.0,
        "bins": counts,
    }


def axis_display_label(axis: dict[str, Any]) -> str:
    return f"{axis['name']} ({axis.get('unit')})" if axis.get("unit") else str(axis["name"])


def build_report_visualizations(
    goal: dict[str, Any],
    peer_group: np.ndarray,
    target_vector: np.ndarray,
    result: DiagnosisResult,
) -> dict[str, Any]:
    axes = goal["axes"]
    sample_size_items = []
    coverage_axes = []
    equitability_axes = []
    goal_k_m = float(goal.get("K_m", K_M))

    for index, axis in enumerate(axes):
        axis_values = peer_group[:, index]
        distribution = build_axis_distribution(
            axis_values,
            float(axis["domainMin"]),
            float(axis["domainMax"]),
            float(axis["resolution"]),
        )
        non_empty_count = int(np.count_nonzero(~np.isnan(axis_values)))
        sample_size_items.append(
            {
                "axis": axis["name"],
                "label": axis_display_label(axis),
                "unit": axis.get("unit", ""),
                "count": non_empty_count,
                "z": round(float(non_empty_count / (non_empty_count + goal_k_m)), 6),
            }
        )
        coverage_axes.append(
            {
                "axis": axis["name"],
                "label": axis_display_label(axis),
                "unit": axis.get("unit", ""),
                "domainMin": float(axis["domainMin"]),
                "domainMax": float(axis["domainMax"]),
                "resolution": float(axis["resolution"]),
                "targetValue": round(float(target_vector[index]), 6),
                "peerValues": [round(float(value), 6) for value in axis_values],
                **distribution,
            }
        )
        equitability_axes.append(
            {
                "axis": axis["name"],
                "label": axis_display_label(axis),
                "unit": axis.get("unit", ""),
                "status": "balanced" if result.equitability_E >= 0.5 else "imbalanced",
                "bins": distribution["bins"],
            }
        )

    return {
        "sampleSize": {"peerGroupCount": int(len(peer_group)), "z": round(float(result.sample_size_Z), 6), "items": sample_size_items},
        "coverage": {"score": round(float(result.coverage_C), 6), "axes": coverage_axes},
        "equitability": {
            "score": round(float(result.equitability_E), 6),
            "status": "balanced" if result.equitability_E >= 0.5 else "imbalanced",
            "axes": equitability_axes,
        },
        "peerRows": [[round(float(value), 6) for value in row] for row in peer_group.tolist()],
        "targetVector": [round(float(value), 6) for value in target_vector.tolist()],
        "axisNames": [axis["name"] for axis in axes],
    }


def confidence_reasons(result: DiagnosisResult) -> list[dict[str, Any]]:
    reasons = [
        {
            "label": "Sample Size",
            "score": round(float(result.sample_size_Z), 4),
            "impact": "down" if result.sample_size_Z < 0.6 else "stable",
            "message": "K_m은 Sample Size confidence가 0.5에 도달하는 Peer Group 수를 의미하는 half-saturation constant입니다.",
        },
        {
            "label": "Coverage",
            "score": round(float(result.coverage_C), 4),
            "impact": "down" if result.coverage_C < 0.3 else "stable",
            "message": "Coverage는 사용자가 설정한 Domain Range와 Resolution 기준에서 Peer Group이 실험 공간을 얼마나 점유했는지를 나타내는 상대적 대표성 지표입니다.",
        },
        {
            "label": "Equitability",
            "score": round(float(result.equitability_E), 4),
            "impact": "down" if result.equitability_E < 0.5 else "stable",
            "message": "점유된 bin 안에서 Peer Group cluster vector들이 얼마나 균형 있게 분포하는지 반영합니다.",
        },
    ]
    if result.w_eff < 0.7:
        reasons.append(
            {
                "label": "Engine Robustness",
                "score": round(float(result.w_eff), 4),
                "impact": "down",
                "message": "Mardia 첨도 기반 효율 가중치가 낮아 최종 confidence가 보수적으로 조정되었습니다.",
            }
        )
    return reasons


def build_summary(result: DiagnosisResult) -> list[str]:
    messages: list[str] = []
    if result.heterogeneity > 0.95 and result.confidence > 0.7:
        messages.append("이질성과 confidence가 모두 높습니다. 새로운 물리적 발견 가능성을 우선 검토할 수 있습니다.")
    elif result.heterogeneity > 0.95 and result.confidence <= 0.4:
        messages.append("타겟 군집은 Peer Group에서 벗어나지만 confidence가 낮습니다. 설계/coverage/sample 부족 가능성을 함께 봐야 합니다.")
    elif result.heterogeneity <= 0.5:
        messages.append("타겟 군집은 현재 Peer Group과 통계적으로 크게 다르지 않습니다.")
    else:
        messages.append("이질성이 일부 확인됩니다. 추가 군집 확보와 분포 검증을 권장합니다.")

    if result.sample_size_Z < 0.5:
        messages.append("Sample Size가 부족합니다. 같은 Experiment Goal과 Axis 구성의 누적 군집을 더 확보하세요.")
    if result.coverage_C < 0.3:
        messages.append(
            "Coverage가 낮습니다. 현재 전체 bin 수가 Peer Group 수에 비해 큽니다. "
            "Domain Range가 너무 넓거나 Resolution이 현재 데이터 규모에 비해 너무 세밀할 수 있습니다."
        )
    if result.equitability_E < 0.5:
        messages.append("Equitability가 낮습니다. 일부 bin에 Peer Group cluster vector가 몰려 있습니다.")
    if result.w_eff < 0.7:
        messages.append("Engine Robustness가 낮아 최종 confidence가 보수적으로 반영되었습니다.")
    return messages


def make_cluster_record(
    goal: dict[str, Any],
    selected_goal: dict[str, Any],
    cluster_vector: np.ndarray,
    dataset_meta: dict[str, Any],
) -> dict[str, Any]:
    axis_names = [axis["name"] for axis in selected_goal["axes"]]
    values = [round(float(value), 12) for value in cluster_vector]
    key = peer_group_key(str(goal["id"]), axis_names)
    fingerprint_payload = {
        "goalId": goal["id"],
        "peerGroupKey": key,
        "axisNames": axis_names,
        "values": values,
        "rowCount": dataset_meta["row_count"],
        "summaryMethod": dataset_meta.get("summary_method", "mean"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "id": f"cluster_{uuid.uuid4().hex}",
        "goalId": goal["id"],
        "goalName": goal["name"],
        "axisNames": axis_names,
        "axisSignature": axis_subset_key(axis_names),
        "peerGroupKey": key,
        "values": values,
        "rowCount": int(dataset_meta["row_count"]),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "storagePolicy": "sanitized_numeric_axis_vector",
    }


def analyze_request(payload: dict[str, Any]) -> dict[str, Any]:
    goals = normalize_goals_for_display(load_goal_store())
    goal_id = str(payload.get("goalId", ""))
    goal = next((item for item in goals if item["id"] == goal_id), None)
    if goal is None:
        raise ValueError("선택한 Experiment Goal이 존재하지 않습니다.")

    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("업로드된 CSV row가 비어 있습니다.")

    axis_mapping = payload.get("axisMapping", {})
    if not isinstance(axis_mapping, dict):
        raise ValueError("Axis column mapping 정보가 없습니다.")

    selected_axis_names = payload.get("selectedAxes")
    if selected_axis_names is None:
        selected_axis_names = [axis["name"] for axis in goal["axes"]]
    if not isinstance(selected_axis_names, list):
        raise ValueError("selectedAxes must be a list.")
    selected_goal = goal_subset(goal, [str(name) for name in selected_axis_names])
    axis_names = [axis["name"] for axis in selected_goal["axes"]]
    key = peer_group_key(str(goal["id"]), axis_names)

    config = ExperimentConfig(
        axis_names=axis_names,
        domain_range=[(axis["domainMin"], axis["domainMax"]) for axis in selected_goal["axes"]],
        resolution=[axis["resolution"] for axis in selected_goal["axes"]],
        K_m=float(selected_goal.get("K_m", K_M)),
    )

    cluster_vector, dataset_meta = build_cluster_vector(rows, axis_mapping, selected_goal)
    pending_cluster = make_cluster_record(goal, selected_goal, cluster_vector, dataset_meta)
    saved_cluster = None
    saved_cluster_is_new = False

    try:
        peer_group = get_peer_group(goal, axis_names)
        analyzer = DataQualityAnalyzer(config)
        analyzer.add_peers(peer_group)
        result = analyzer.diagnose(cluster_vector)
    except ValueError as exc:
        if should_save_data_clusters():
            saved_cluster, saved_cluster_is_new = save_data_cluster(pending_cluster)
        stored_count = len(load_peer_clusters(str(goal["id"]), axis_names))
        saved_text = "저장했습니다" if saved_cluster_is_new else "이미 저장되어 있습니다"
        raise ValueError(
            "이 Experiment Goal과 Axis 구성이 정확히 같은 누적 군집만 Peer Group으로 사용됩니다. "
            f"현재 저장된 Peer Group N={stored_count}, 필요한 최소 N={len(axis_names) + 1}. "
            f"이번 CSV 군집은 원본 없이 axis 숫자 벡터로 {saved_text}. "
            f"새 Goal은 충분한 과거 군집이 쌓이기 전까지 분석 신뢰도가 낮거나 분석이 제한될 수 있습니다. 내부 사유: {exc}"
        ) from exc

    if should_save_data_clusters():
        saved_cluster, saved_cluster_is_new = save_data_cluster(pending_cluster)

    summary = build_summary(result)
    summary.append(f"Peer Group Key는 '{key}'입니다. CSV 파일 하나는 하나의 데이터 군집으로 저장됩니다.")
    if saved_cluster:
        status = "새 군집으로 저장했습니다" if saved_cluster_is_new else "이미 저장된 군집이라 중복 저장하지 않았습니다"
        summary.append(f"업로드 원본은 저장하지 않고 axis 숫자 벡터만 {status}.")

    return {
        "meta": {
            "experiment_goal": goal["name"],
            "goal_id": goal["id"],
            "peer_group_key": key,
            "target_rows": dataset_meta["row_count"],
            "uploaded_columns": dataset_meta["columns"],
            "summary_method": dataset_meta["summary_method"],
            "peer_group_size": int(len(peer_group)),
            "axis_names": config.axis_names,
            "axes": selected_goal["axes"],
            "available_axes": goal["axes"],
            "config": asdict(config),
            "cluster_definition": dataset_meta["cluster_definition"],
            "storage_policy": "Raw upload rows, filenames, and unmapped columns are not stored.",
        },
        "result": result.to_payload(config.axis_names),
        "summary": summary,
        "confidenceReasons": confidence_reasons(result),
        "visualizations": build_report_visualizations(selected_goal, peer_group, cluster_vector, result),
        "clusters": list_cluster_summaries(),
        "savedDataCluster": None
        if saved_cluster is None
        else {
            "id": saved_cluster["id"],
            "isNew": saved_cluster_is_new,
            "axisNames": saved_cluster["axisNames"],
            "rowCount": saved_cluster["rowCount"],
            "storeFile": storage_label(CLUSTER_STORE_PATH),
            "storagePolicy": saved_cluster["storagePolicy"],
        },
    }


def build_bootstrap_payload(admin_allowed: bool) -> dict[str, Any]:
    goals = normalize_goals_for_display(load_goal_store())
    peer_counts = {}
    peer_subset_counts = {}
    for goal in goals:
        full_axis_names = [axis["name"] for axis in goal["axes"]]
        try:
            peer_counts[goal["id"]] = int(len(get_peer_group(goal, full_axis_names)))
        except ValueError:
            peer_counts[goal["id"]] = 0
        peer_subset_counts[goal["id"]] = peer_group_subset_counts(goal)
    admin_auth_required = bool(os.environ.get("ADMIN_TOKEN")) and not admin_allowed
    return {
        "adminAllowed": admin_allowed,
        "adminAuthRequired": admin_auth_required,
        "goals": goals,
        "clusters": list_cluster_summaries(),
        "peerCounts": peer_counts,
        "peerSubsetCounts": peer_subset_counts,
        "acceptedUploadTypes": [".csv", ".tsv", ".txt"],
        "storage": {
            "storeDir": storage_label(STORE_DIR),
            "goalStoreFile": storage_label(GOAL_STORE_PATH),
            "clusterStoreFile": storage_label(CLUSTER_STORE_PATH),
            "clusterCount": len(load_cluster_store()),
            "demoPeerGroupEnabled": use_demo_peer_group(),
            "savedItems": ["Experiment Goal 설정", "비식별 numeric cluster vector"],
            "unsavedItems": ["원본 CSV 파일", "파일명", "비매핑 column", "개인정보 column"],
        },
        "domainDefinitions": {
            "cluster": "CSV 파일 하나는 하나의 데이터 군집으로 취급됩니다. CSV 내부 row들은 해당 군집의 반복 관측값이며, 시스템은 row들을 axis별 대표값으로 집계하여 하나의 cluster vector를 만듭니다.",
            "coverage": "Coverage는 절대적인 데이터 품질 점수가 아니라, 사용자가 설정한 Domain Range와 Resolution 기준에서 Peer Group이 실험 공간을 얼마나 점유했는지를 나타내는 상대적 대표성 지표입니다.",
            "K_m": "K_m은 Sample Size confidence가 0.5에 도달하는 Peer Group 수를 의미하는 half-saturation constant입니다. Z=N/(N+K_m), N=K_m일 때 Z=0.5입니다.",
        },
    }


def render_page(admin_allowed: bool) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    bootstrap = json.dumps(build_bootstrap_payload(admin_allowed), ensure_ascii=False)
    return template.replace("__BOOTSTRAP__", bootstrap)


class AppHandler(BaseHTTPRequestHandler):
    def _admin_allowed(self) -> bool:
        if os.environ.get("ALLOW_REMOTE_ADMIN", "").lower() in {"1", "true", "yes", "on"}:
            return True
        admin_token = os.environ.get("ADMIN_TOKEN", "")
        if admin_token and self.headers.get("X-Admin-Token", "") == admin_token:
            return True
        try:
            return ipaddress.ip_address(self.client_address[0]).is_loopback
        except ValueError:
            return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(render_page(admin_allowed=self._admin_allowed()))
            return
        if parsed.path == "/api/bootstrap":
            self._send_json(build_bootstrap_payload(admin_allowed=self._admin_allowed()))
            return
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        payload = json.loads(body) if body else {}
        try:
            if parsed.path == "/api/analyze":
                self._send_json(analyze_request(payload))
                return
            if parsed.path == "/api/admin/goals":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin Token이 필요합니다."}, status=HTTPStatus.FORBIDDEN)
                    return
                saved_goal = validate_goal(payload)
                goals = load_goal_store()
                existing_index = next((index for index, item in enumerate(goals) if item["id"] == saved_goal["id"]), None)
                if existing_index is None:
                    goals.append(saved_goal)
                else:
                    goals[existing_index] = saved_goal
                save_goal_store(goals)
                self._send_json({"savedGoal": saved_goal, "goals": normalize_goals_for_display(goals), "clusters": list_cluster_summaries()})
                return
            if parsed.path == "/api/admin/goals/delete":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin Token이 필요합니다."}, status=HTTPStatus.FORBIDDEN)
                    return
                goal_id = payload.get("id")
                goals = [goal for goal in load_goal_store() if goal["id"] != goal_id]
                if not goals:
                    goals = load_goal_store()
                save_goal_store(goals)
                self._send_json({"goals": normalize_goals_for_display(goals), "clusters": list_cluster_summaries()})
                return
            if parsed.path == "/api/admin/clusters/delete":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin Token이 필요합니다."}, status=HTTPStatus.FORBIDDEN)
                    return
                deleted = delete_peer_cluster(str(payload.get("id", "")))
                if not deleted:
                    raise ValueError("삭제할 군집을 찾을 수 없습니다.")
                self._send_json({"deleted": True, "clusters": list_cluster_summaries()})
                return
        except Exception as exc:
            self._send_json({"error": str(exc), "clusters": list_cluster_summaries()}, status=HTTPStatus.BAD_REQUEST)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    init_database()
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Serving Leesin data quality certification at http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
