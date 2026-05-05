from __future__ import annotations

import argparse
import csv
import html
import io
import ipaddress
import json
import os
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from models import K_M, DiagnosisResult, ExperimentConfig
from stats_engine import BinGridTracker, DataQualityAnalyzer
from storage import (
    CLUSTER_STORE_PATH,
    GOAL_STORE_PATH,
    STORE_DIR,
    axis_subset_key,
    cluster_fingerprint,
    cluster_fingerprint_payload,
    delete_peer_cluster,
    explain_peer_filter,
    get_peer_group,
    init_database,
    list_cluster_summaries,
    load_cluster_store,
    load_goal_store,
    load_peer_clusters,
    normalize_analysis_snapshot,
    normalize_axis_name,
    normalize_goals_for_display,
    peer_group_key,
    peer_group_subset_counts,
    save_data_cluster,
    save_peer_cluster,
    save_goal_store,
    should_save_data_clusters,
    storage_label,
    utc_now_iso,
    validate_goal,
)


TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


def goal_subset(goal: dict[str, Any], selected_axis_names: list[str] | None = None) -> dict[str, Any]:
    if not selected_axis_names:
        axes = [dict(axis) for axis in goal["axes"]]
    else:
        requested = {normalize_axis_name(name) for name in selected_axis_names if normalize_axis_name(name)}
        axes = [dict(axis) for axis in goal["axes"] if normalize_axis_name(axis["name"]) in requested]
    if not axes:
        raise ValueError("분석에 포함할 Axis를 하나 이상 선택하세요.")
    return {
        "id": goal["id"],
        "name": goal["name"],
        "K_m": float(goal.get("K_m", K_M)),
        "axes": axes,
    }


def bin_index_for_value(value: Any, domain_min: float, domain_max: float, resolution: float) -> tuple[int | None, str]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None, "invalid"
    if not np.isfinite(numeric):
        return None, "invalid"
    if numeric < domain_min or numeric > domain_max:
        return None, "out_of_domain"
    total_bins = max(1, int(np.ceil((domain_max - domain_min) / resolution)))
    if numeric == domain_max:
        return total_bins - 1, "valid"
    index = int(np.floor((numeric - domain_min) / resolution))
    return max(0, min(total_bins - 1, index)), "valid"


def compute_row_level_bin_occupancy(
    rows: list[dict[str, Any]],
    axis_mapping: dict[str, str],
    selected_goal: dict[str, Any],
) -> dict[str, Any]:
    axes = selected_goal["axes"]
    axis_mapping_by_key = {normalize_axis_name(axis_name): column for axis_name, column in axis_mapping.items()}
    bin_occupancy: dict[str, int] = {}
    axis_bin_occupancy: dict[str, dict[str, int]] = {str(axis["name"]): {} for axis in axes}
    valid_multidimensional = 0
    invalid_rows = 0
    out_of_domain_rows = 0

    for row in rows:
        multidimensional_indices: list[int] = []
        row_has_invalid = False
        row_has_out_of_domain = False
        for axis in axes:
            axis_name = str(axis["name"])
            column = axis_mapping.get(axis_name) or axis_mapping_by_key.get(normalize_axis_name(axis_name))
            raw_value = row.get(column, "") if column else ""
            index, status = bin_index_for_value(
                raw_value,
                float(axis["domainMin"]),
                float(axis["domainMax"]),
                float(axis["resolution"]),
            )
            if status == "valid" and index is not None:
                key = str(index)
                axis_counts = axis_bin_occupancy[axis_name]
                axis_counts[key] = axis_counts.get(key, 0) + 1
                multidimensional_indices.append(index)
            elif status == "invalid":
                row_has_invalid = True
            else:
                row_has_out_of_domain = True

        if row_has_invalid:
            invalid_rows += 1
            continue
        if row_has_out_of_domain:
            out_of_domain_rows += 1
            continue
        if len(multidimensional_indices) == len(axes):
            key = json.dumps(multidimensional_indices, separators=(",", ":"))
            bin_occupancy[key] = bin_occupancy.get(key, 0) + 1
            valid_multidimensional += 1

    return {
        "bin_occupancy": bin_occupancy,
        "axis_bin_occupancy": axis_bin_occupancy,
        "bin_occupancy_meta": {
            "version": 1,
            "basis": "row_level",
            "validMultidimensionalRowCount": valid_multidimensional,
            "invalidRowCount": invalid_rows,
            "outOfDomainRowCount": out_of_domain_rows,
            "totalRows": len(rows),
        },
    }


def build_cluster_vector(
    rows: list[dict[str, Any]],
    axis_mapping: dict[str, str],
    selected_goal: dict[str, Any],
    method: str = "mean",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Summarize one CSV file into one cluster vector.

    CSV file = one data cluster. Rows inside the CSV are repeated observations.
    Welford's online algorithm keeps streaming axis-wise mean/variance/std. The
    method argument keeps room for future median/std/IQR summaries without
    changing the caller contract.
    """
    if method != "mean":
        raise ValueError("Only mean cluster summarization is currently supported.")
    if not rows:
        raise ValueError("업로드된 데이터가 비어 있습니다.")

    axes = selected_goal["axes"]
    means = np.zeros(len(axes), dtype=float)
    m2 = np.zeros(len(axes), dtype=float)
    axis_numeric_counts = np.zeros(len(axes), dtype=int)
    cluster_vector_row_count = 0
    columns = list(rows[0].keys()) if rows else []
    axis_mapping_by_key = {normalize_axis_name(axis_name): column for axis_name, column in axis_mapping.items()}
    occupancy = compute_row_level_bin_occupancy(rows, axis_mapping, selected_goal)

    for row in rows:
        row_values: list[float] = []
        row_is_numeric_for_all_axes = True
        for axis_index, axis in enumerate(axes):
            axis_name = axis["name"]
            column = axis_mapping.get(axis_name) or axis_mapping_by_key.get(normalize_axis_name(axis_name))
            if not column:
                raise ValueError(f"Axis '{axis_name}'에 매핑된 CSV column이 없습니다.")
            raw_value = row.get(column, "")
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                row_is_numeric_for_all_axes = False
                row_values.append(0.0)
                continue
            if not np.isfinite(numeric):
                row_is_numeric_for_all_axes = False
                row_values.append(0.0)
                continue
            axis_numeric_counts[axis_index] += 1
            row_values.append(numeric)

        if not row_is_numeric_for_all_axes or len(row_values) != len(axes):
            continue

        cluster_vector_row_count += 1
        for axis_index, numeric in enumerate(row_values):
            delta = numeric - means[axis_index]
            means[axis_index] += delta / cluster_vector_row_count
            delta_after = numeric - means[axis_index]
            m2[axis_index] += delta * delta_after

    if cluster_vector_row_count == 0:
        raise ValueError("선택된 모든 Axis에 사용할 수 있는 multidimensional numeric row가 없습니다.")

    variance = m2 / max(cluster_vector_row_count - 1, 1) if cluster_vector_row_count > 1 else np.zeros_like(m2)
    std = np.sqrt(variance)
    return means, {
        "row_count": len(rows),
        "columns": columns,
        "summary_method": method,
        "values_mean": [round(float(value), 12) for value in means],
        "values_variance": [round(float(value), 12) for value in variance],
        "values_std": [round(float(value), 12) for value in std],
        "cluster_vector_row_count": int(cluster_vector_row_count),
        "cluster_vector_basis": "valid_multidimensional_numeric_rows",
        "axis_numeric_counts": {str(axis["name"]): int(axis_numeric_counts[index]) for index, axis in enumerate(axes)},
        "bin_occupancy": occupancy["bin_occupancy"],
        "axis_bin_occupancy": occupancy["axis_bin_occupancy"],
        "bin_occupancy_meta": occupancy["bin_occupancy_meta"],
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


def build_axis_distribution_from_counts(counts_by_bin: dict[str, Any], total_bins: int) -> dict[str, Any]:
    counts = [0 for _ in range(total_bins)]
    for raw_index, raw_count in (counts_by_bin or {}).items():
        try:
            index = int(raw_index)
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count <= 0 or index < 0 or index >= total_bins:
            continue
        counts[index] += count
    occupied = sum(1 for count in counts if count > 0)
    return {
        "totalBins": total_bins,
        "occupiedBins": occupied,
        "coverage": occupied / total_bins if total_bins else 0.0,
        "bins": counts,
        "observationCount": int(sum(counts)),
    }


def axis_display_label(axis: dict[str, Any]) -> str:
    return f"{axis['name']} ({axis.get('unit')})" if axis.get("unit") else str(axis["name"])


def build_report_visualizations(
    goal: dict[str, Any],
    peer_group: np.ndarray,
    target_vector: np.ndarray,
    result: DiagnosisResult,
    coverage_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    axes = goal["axes"]
    sample_size_items = []
    coverage_axes = []
    equitability_axes = []
    goal_k_m = float(goal.get("K_m", K_M))
    axis_bin_counts = coverage_info.get("axisBinCounts", {}) if isinstance(coverage_info, dict) else {}
    axis_bin_counts_by_key = {normalize_axis_name(axis_name): counts for axis_name, counts in axis_bin_counts.items()}

    for index, axis in enumerate(axes):
        axis_values = peer_group[:, index] if peer_group.ndim == 2 and peer_group.shape[1] > index else np.asarray([], dtype=float)
        total_axis_bins = max(1, int(np.ceil((float(axis["domainMax"]) - float(axis["domainMin"])) / float(axis["resolution"]))))
        row_level_counts = axis_bin_counts_by_key.get(normalize_axis_name(axis["name"]), {})
        if row_level_counts:
            distribution = build_axis_distribution_from_counts(row_level_counts, total_axis_bins)
            distribution_basis = "row_level_bin_occupancy"
            peer_values = []
        else:
            distribution = build_axis_distribution(
                axis_values,
                float(axis["domainMin"]),
                float(axis["domainMax"]),
                float(axis["resolution"]),
            )
            distribution["observationCount"] = int(np.count_nonzero(~np.isnan(axis_values)))
            distribution_basis = "cluster_vector_fallback"
            peer_values = [round(float(value), 6) for value in axis_values]
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
                "peerValues": peer_values,
                "basis": distribution_basis,
                "fallbackReason": "" if distribution_basis == "row_level_bin_occupancy" else "axisBinOccupancy unavailable; displaying saved cluster representative vectors.",
                **distribution,
            }
        )
        equitability_axes.append(
            {
                "axis": axis["name"],
                "label": axis_display_label(axis),
                "unit": axis.get("unit", ""),
                "status": "balanced" if result.equitability_E >= 0.5 else "imbalanced",
                "basis": distribution_basis,
                "bins": distribution["bins"],
                "observationCount": distribution.get("observationCount", 0),
            }
        )

    basis_values = {axis["basis"] for axis in coverage_axes}
    visualization_basis = basis_values.pop() if len(basis_values) == 1 else "mixed"
    return {
        "sampleSize": {"peerGroupCount": int(len(peer_group)), "z": round(float(result.sample_size_Z), 6), "items": sample_size_items},
        "coverage": {"score": round(float(result.coverage_C), 6), "basis": visualization_basis, "axes": coverage_axes},
        "equitability": {
            "score": round(float(result.equitability_E), 6),
            "basis": visualization_basis,
            "status": "balanced" if result.equitability_E >= 0.5 else "imbalanced",
            "axes": equitability_axes,
        },
        "peerRows": [[round(float(value), 6) for value in row] for row in peer_group.tolist()],
        "targetVector": [round(float(value), 6) for value in target_vector.tolist()],
        "axisNames": [axis["name"] for axis in axes],
    }


def confidence_reasons(result: DiagnosisResult, warnings: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
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
            "message": "Coverage는 저장된 peer cluster들의 row-level bin occupancy가 Domain grid를 얼마나 점유했는지 나타냅니다.",
        },
        {
            "label": "Equitability",
            "score": round(float(result.equitability_E), 4),
            "impact": "down" if result.equitability_E < 0.5 else "stable",
            "message": "Equitability는 점유된 row-level bins 안에서 observation count가 얼마나 균등한지 반영합니다.",
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
    if warnings:
        reasons.append(
            {
                "label": "Out-of-domain clipping",
                "score": 0.0,
                "impact": "down",
                "message": f"{len(warnings)} value(s) are outside configured domain ranges. Bin clipping is still applied, but this report should be interpreted with caution.",
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
            "Coverage가 낮습니다. 저장된 peer cluster의 row-level bin occupancy가 아직 domain grid를 충분히 점유하지 못했습니다. "
            "Domain Range가 너무 넓거나 Resolution이 현재 데이터 규모에 비해 너무 세밀할 수 있습니다."
        )
    if result.equitability_E < 0.5:
        messages.append("Equitability가 낮습니다. 점유된 row-level bin 안의 observation count가 일부 bin에 치우쳐 있습니다.")
    if result.w_eff < 0.7:
        messages.append("Engine Robustness가 낮아 최종 confidence가 보수적으로 반영되었습니다.")
    return messages


def make_cluster_record(
    goal: dict[str, Any],
    selected_goal: dict[str, Any],
    cluster_vector: np.ndarray,
    dataset_meta: dict[str, Any],
    source_batch_id: str | None = None,
    analysis_at_upload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    axis_names = [axis["name"] for axis in selected_goal["axes"]]
    values = [round(float(value), 12) for value in cluster_vector]
    key = peer_group_key(str(goal["id"]), axis_names)
    now = utc_now_iso()
    analysis = normalize_analysis_snapshot(analysis_at_upload)
    record = {
        "id": f"cluster_{uuid.uuid4().hex}",
        "goalId": goal["id"],
        "goalName": goal["name"],
        "axisNames": axis_names,
        "axisSignature": axis_subset_key(axis_names),
        "peerGroupKey": key,
        "values": values,
        "valuesMean": dataset_meta.get("values_mean", values),
        "valuesVariance": dataset_meta.get("values_variance", [None for _ in values]),
        "valuesStd": dataset_meta.get("values_std", [None for _ in values]),
        "rowCount": int(dataset_meta["row_count"]),
        "clusterVectorRowCount": int(dataset_meta.get("cluster_vector_row_count") or dataset_meta["row_count"]),
        "clusterVectorBasis": str(dataset_meta.get("cluster_vector_basis") or "valid_multidimensional_numeric_rows"),
        "axisNumericCounts": dataset_meta.get("axis_numeric_counts", {}),
        "binOccupancy": dataset_meta.get("bin_occupancy", {}),
        "axisBinOccupancy": dataset_meta.get("axis_bin_occupancy", {}),
        "binOccupancyMeta": dataset_meta.get(
            "bin_occupancy_meta",
            {
                "version": 1,
                "basis": "row_level",
                "validMultidimensionalRowCount": 0,
                "invalidRowCount": 0,
                "outOfDomainRowCount": 0,
                "totalRows": int(dataset_meta["row_count"]),
            },
        ),
        "createdAt": now,
        "uploadedAt": now,
        "sourceBatchId": source_batch_id,
        "summaryMethod": dataset_meta.get("summary_method", "mean"),
        "storagePolicy": "sanitized_numeric_axis_vector",
        "analysisAtUpload": analysis,
        "peerGroupSizeAtUpload": analysis.get("peerGroupSize"),
        "engineAtUpload": analysis.get("engine"),
        "heterogeneityAtUpload": analysis.get("heterogeneity"),
        "confidenceAtUpload": analysis.get("confidence"),
        "D2AtUpload": analysis.get("D2"),
        "pValueAtUpload": analysis.get("pValue"),
        "sampleSizeZAtUpload": analysis.get("sampleSizeZ"),
        "coverageCAtUpload": analysis.get("coverageC"),
        "equitabilityEAtUpload": analysis.get("equitabilityE"),
        "wEffAtUpload": analysis.get("wEff"),
        "contributionsAtUpload": analysis.get("contributions"),
        "totalBinsAtUpload": analysis.get("totalBins"),
        "occupiedBinsAtUpload": analysis.get("occupiedBins"),
        "mardiaSkewStatAtUpload": analysis.get("mardiaSkewStat"),
        "mardiaSkewPvalAtUpload": analysis.get("mardiaSkewPval"),
        "mardiaKurtStatAtUpload": analysis.get("mardiaKurtStat"),
        "mardiaKurtPvalAtUpload": analysis.get("mardiaKurtPval"),
        "b2pAtUpload": analysis.get("b2p"),
    }
    record["fingerprint"] = cluster_fingerprint(record)
    return record


def experiment_config_from_goal(selected_goal: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        axis_names=[axis["name"] for axis in selected_goal["axes"]],
        domain_range=[(axis["domainMin"], axis["domainMax"]) for axis in selected_goal["axes"]],
        resolution=[axis["resolution"] for axis in selected_goal["axes"]],
        K_m=float(selected_goal.get("K_m", K_M)),
    )


def analysis_peer_rows(
    goal: dict[str, Any],
    axis_names: list[str],
    exclude_cluster_id: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster in load_peer_clusters(str(goal["id"]), axis_names, exclude_cluster_id):
        rows.append({"id": str(cluster.get("id", "")), "source": "stored", "values": [float(value) for value in cluster["values"]]})
    return rows


def analysis_peer_clusters(
    goal: dict[str, Any],
    axis_names: list[str],
    exclude_cluster_id: str | None = None,
) -> list[dict[str, Any]]:
    return load_peer_clusters(str(goal["id"]), axis_names, exclude_cluster_id)


def peer_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.asarray([row["values"] for row in rows], dtype=float)


def merge_count_maps(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in (source or {}).items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        target[str(key)] = target.get(str(key), 0) + count


def count_map_total(source: dict[str, Any]) -> int:
    total = 0
    for value in (source or {}).values():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            total += count
    return total


def build_global_bin_counts(peer_clusters: list[dict[str, Any]], selected_axis_names: list[str]) -> dict[str, Any]:
    selected_signature = axis_subset_key(selected_axis_names)
    bin_counts: dict[str, int] = {}
    axis_bin_counts: dict[str, dict[str, int]] = {}
    eligible_count = 0
    legacy_excluded = 0
    axis_signature_excluded = 0
    row_level_observation_count = 0

    for cluster in peer_clusters:
        stored_signature = str(cluster.get("storedAxisSignature") or cluster.get("axisSignature") or "")
        if stored_signature != selected_signature:
            axis_signature_excluded += 1
            continue
        cluster_bin_counts = cluster.get("binOccupancy") if isinstance(cluster.get("binOccupancy"), dict) else {}
        if not cluster_bin_counts:
            legacy_excluded += 1
            continue
        eligible_count += 1
        merge_count_maps(bin_counts, cluster_bin_counts)
        meta = cluster.get("binOccupancyMeta") if isinstance(cluster.get("binOccupancyMeta"), dict) else {}
        try:
            row_level_observation_count += int(meta.get("validMultidimensionalRowCount") or count_map_total(cluster_bin_counts))
        except (TypeError, ValueError):
            row_level_observation_count += count_map_total(cluster_bin_counts)
        for axis_name, counts in (cluster.get("axisBinOccupancy") or {}).items():
            axis_counts = axis_bin_counts.setdefault(str(axis_name), {})
            merge_count_maps(axis_counts, counts)

    return {
        "binCounts": bin_counts,
        "axisBinCounts": axis_bin_counts,
        "coverageBasis": "row_level_bin_occupancy",
        "coverageEligibleClusterCount": eligible_count,
        "coverageLegacyExcludedClusterCount": legacy_excluded,
        "coverageAxisSignatureExcludedClusterCount": axis_signature_excluded,
        "rowLevelObservationCount": row_level_observation_count,
        "occupiedBins": len(bin_counts),
    }


def out_of_domain_warnings(
    selected_goal: dict[str, Any],
    target_vector: np.ndarray,
    peer_group: np.ndarray | None = None,
    peer_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    axes = selected_goal["axes"]
    for axis_index, axis in enumerate(axes):
        lo = float(axis["domainMin"])
        hi = float(axis["domainMax"])
        axis_name = str(axis["name"])
        target_value = float(target_vector[axis_index])
        if target_value < lo or target_value > hi:
            warnings.append(
                {
                    "axis": axis_name,
                    "value": target_value,
                    "domainMin": lo,
                    "domainMax": hi,
                    "role": "target",
                }
            )
        if peer_group is None:
            continue
        for row_index, peer in enumerate(peer_group, start=1):
            value = float(peer[axis_index])
            if lo <= value <= hi:
                continue
            source = peer_rows[row_index - 1] if peer_rows and row_index - 1 < len(peer_rows) else {}
            warnings.append(
                {
                    "axis": axis_name,
                    "value": value,
                    "domainMin": lo,
                    "domainMax": hi,
                    "role": "peer",
                    "peerIndex": row_index,
                    "clusterId": source.get("id"),
                    "source": source.get("source", "stored"),
                }
            )
    return warnings


def analysis_snapshot(
    result: DiagnosisResult | None,
    axis_names: list[str],
    peer_group_size: int,
    warnings: list[dict[str, Any]] | None = None,
    error: str | None = None,
    coverage_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if result is None:
        snapshot = normalize_analysis_snapshot({})
        snapshot["analysisTimestamp"] = utc_now_iso()
        snapshot["peerGroupSize"] = int(peer_group_size)
        snapshot["outOfDomainWarnings"] = warnings or []
        if coverage_info:
            snapshot.update({key: value for key, value in coverage_info.items() if key != "binCounts" and key != "axisBinCounts"})
        if error:
            snapshot["error"] = error
        return snapshot
    payload = result.to_payload(axis_names)
    snapshot = normalize_analysis_snapshot(
        {
            "analysisTimestamp": utc_now_iso(),
            "peerGroupSize": int(peer_group_size),
            "engine": payload["engine"],
            "isNormal": payload["is_normal"],
            "center": payload["center"],
            "D2": payload["D2"],
            "pValue": payload["p_value"],
            "heterogeneity": payload["heterogeneity"],
            "confidence": payload["confidence"],
            "sampleSizeZ": payload["sample_size_Z"],
            "coverageC": payload["coverage_C"],
            "equitabilityE": payload["equitability_E"],
            "wEff": payload["w_eff"],
            "totalBins": payload["total_bins"],
            "occupiedBins": payload["occupied_bins"],
            "contributions": payload["contributions"],
            "mardiaSkewStat": payload["mardia_skew_stat"],
            "mardiaSkewPval": payload["mardia_skew_pval"],
            "mardiaKurtStat": payload["mardia_kurt_stat"],
            "mardiaKurtPval": payload["mardia_kurt_pval"],
            "b2p": payload["b2p"],
            "outOfDomainWarnings": warnings or [],
        }
    )
    if coverage_info:
        snapshot.update({key: value for key, value in coverage_info.items() if key != "binCounts" and key != "axisBinCounts"})
    return snapshot


def axis_ablation_sensitivity(
    selected_goal: dict[str, Any],
    target_vector: np.ndarray,
    peer_group: np.ndarray,
    base_result: DiagnosisResult,
) -> list[dict[str, Any]]:
    axes = selected_goal["axes"]
    sensitivity: list[dict[str, Any]] = []
    if len(axes) <= 1:
        return [
            {
                "removedAxis": axis["name"],
                "status": "insufficient dimension/sample",
            }
            for axis in axes
        ]

    for removed_index, removed_axis in enumerate(axes):
        kept_indices = [index for index in range(len(axes)) if index != removed_index]
        kept_axes = [axes[index] for index in kept_indices]
        if len(kept_axes) < 2 or len(peer_group) < len(kept_axes) + 1:
            sensitivity.append(
                {
                    "removedAxis": removed_axis["name"],
                    "status": "insufficient dimension/sample",
                }
            )
            continue
        subset_goal = {**selected_goal, "axes": kept_axes}
        try:
            analyzer = DataQualityAnalyzer(experiment_config_from_goal(subset_goal))
            analyzer.add_peers(peer_group[:, kept_indices])
            ablated = analyzer.diagnose(target_vector[kept_indices])
            sensitivity.append(
                {
                    "removedAxis": removed_axis["name"],
                    "status": "ok",
                    "heterogeneity_without_axis": round(float(ablated.heterogeneity), 6),
                    "confidence_without_axis": round(float(ablated.confidence), 6),
                    "D2_without_axis": round(float(ablated.D2), 6),
                    "delta_heterogeneity": round(float(ablated.heterogeneity - base_result.heterogeneity), 6),
                    "delta_confidence": round(float(ablated.confidence - base_result.confidence), 6),
                    "delta_D2": round(float(ablated.D2 - base_result.D2), 6),
                    "interpretation": (
                        f"{removed_axis['name']} strongly drives heterogeneity."
                        if base_result.heterogeneity - ablated.heterogeneity >= 0.15
                        else "No dominant axis-removal effect."
                    ),
                }
            )
        except ValueError:
            sensitivity.append(
                {
                    "removedAxis": removed_axis["name"],
                    "status": "insufficient dimension/sample",
                }
            )
    return sensitivity


def duplicate_status(record: dict[str, Any]) -> dict[str, Any]:
    payload = cluster_fingerprint_payload(record)
    for cluster in load_cluster_store():
        if cluster.get("fingerprint") == record.get("fingerprint") or cluster_fingerprint_payload(cluster) == payload:
            return {"isDuplicate": True, "duplicateClusterId": cluster.get("id")}
    return {"isDuplicate": False, "duplicateClusterId": None}


def peer_filter_diagnostics(goal_id: str, axis_names: list[str], exclude_cluster_id: str | None = None) -> dict[str, Any]:
    return explain_peer_filter(str(goal_id), axis_names, exclude_cluster_id)


def format_peer_filter_error(diagnostics: dict[str, Any]) -> str:
    examples = diagnostics.get("examplesExcluded", [])
    example_text = "; ".join(
        (
            f"{item.get('id') or '(no id)'} reason={item.get('reason')}, "
            f"axisNames={item.get('axisNames')}, missingAxes={item.get('missingAxes', [])}"
        )
        for item in examples[:3]
    )
    return (
        f"totalStoredClusters={diagnostics.get('totalClusters')}, "
        f"sameGoalClusters={diagnostics.get('sameGoalCount')}, "
        f"sameGoalCompatibleAxes={diagnostics.get('compatibleAxisCount')}, "
        f"excludedByGoal={diagnostics.get('excludedByGoal')}, "
        f"excludedByAxis={diagnostics.get('excludedByAxis')}, "
        f"excludedBySelf={diagnostics.get('excludedBySelf')}, "
        f"selectedAxisNames={diagnostics.get('selectedAxisNames')}, "
        f"selectedAxisKeys={diagnostics.get('selectedAxisKeys')}"
        + (f", examplesExcluded=[{example_text}]" if example_text else "")
    )


def run_vector_analysis(
    goal: dict[str, Any],
    selected_goal: dict[str, Any],
    target_vector: np.ndarray,
    exclude_cluster_id: str | None = None,
) -> dict[str, Any]:
    axis_names = [axis["name"] for axis in selected_goal["axes"]]
    config = experiment_config_from_goal(selected_goal)
    peer_clusters = analysis_peer_clusters(goal, axis_names, exclude_cluster_id)
    peer_rows = [{"id": str(cluster.get("id", "")), "source": "stored", "values": [float(value) for value in cluster["values"]]} for cluster in peer_clusters]
    peer_group = peer_matrix(peer_rows)
    if peer_group.size == 0:
        peer_group = np.empty((0, len(axis_names)), dtype=float)
    coverage_info = build_global_bin_counts(peer_clusters, axis_names)
    warnings = out_of_domain_warnings(selected_goal, target_vector, peer_group, peer_rows)
    analyzer = DataQualityAnalyzer(config)
    analyzer.add_peers(peer_group)
    analyzer.add_coverage_bin_counts(coverage_info["binCounts"])
    result = analyzer.diagnose(target_vector)
    result_payload = result.to_payload(config.axis_names)
    result_payload["axisAblationSensitivity"] = axis_ablation_sensitivity(selected_goal, target_vector, peer_group, result)
    result_payload["outOfDomainWarnings"] = warnings
    result_payload["outOfDomainWarningCount"] = len(warnings)
    result_payload["coverageBasis"] = coverage_info["coverageBasis"]
    result_payload["coverageEligibleClusterCount"] = coverage_info["coverageEligibleClusterCount"]
    result_payload["coverageLegacyExcludedClusterCount"] = coverage_info["coverageLegacyExcludedClusterCount"]
    result_payload["coverageAxisSignatureExcludedClusterCount"] = coverage_info["coverageAxisSignatureExcludedClusterCount"]
    result_payload["rowLevelObservationCount"] = coverage_info["rowLevelObservationCount"]
    return {
        "config": config,
        "peerRows": peer_rows,
        "peerClusters": peer_clusters,
        "peerGroup": peer_group,
        "coverageInfo": {
            **coverage_info,
            "totalBins": result.total_bins,
            "occupiedBins": result.occupied_bins,
        },
        "warnings": warnings,
        "result": result,
        "resultPayload": result_payload,
        "snapshot": analysis_snapshot(
            result,
            axis_names,
            len(peer_group),
            warnings,
            coverage_info={**coverage_info, "totalBins": result.total_bins, "occupiedBins": result.occupied_bins},
        ),
    }

def analyze_request_v2(payload: dict[str, Any]) -> dict[str, Any]:
    goals = normalize_goals_for_display(load_goal_store())
    goal_id = str(payload.get("goalId", ""))
    goal = next((item for item in goals if item["id"] == goal_id), None)
    if goal is None:
        raise ValueError("Selected Experiment Goal does not exist.")

    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Uploaded CSV rows are empty.")

    axis_mapping = payload.get("axisMapping", {})
    if not isinstance(axis_mapping, dict):
        raise ValueError("Axis column mapping is missing.")

    selected_axis_names = payload.get("selectedAxes")
    if selected_axis_names is None:
        selected_axis_names = [axis["name"] for axis in goal["axes"]]
    if not isinstance(selected_axis_names, list):
        raise ValueError("selectedAxes must be a list.")

    selected_goal = goal_subset(goal, [str(name) for name in selected_axis_names])
    axis_names = [axis["name"] for axis in selected_goal["axes"]]
    key = peer_group_key(str(goal["id"]), axis_names)
    cluster_vector, dataset_meta = build_cluster_vector(rows, axis_mapping, selected_goal)
    saved_cluster = None
    saved_cluster_is_new = False

    try:
        analysis = run_vector_analysis(goal, selected_goal, cluster_vector)
    except ValueError as exc:
        peer_clusters = analysis_peer_clusters(goal, axis_names)
        peer_rows = [{"id": str(cluster.get("id", "")), "source": "stored", "values": [float(value) for value in cluster["values"]]} for cluster in peer_clusters]
        peer_group = peer_matrix(peer_rows)
        if peer_group.size == 0:
            peer_group = np.empty((0, len(axis_names)), dtype=float)
        coverage_info = build_global_bin_counts(peer_clusters, axis_names)
        coverage_info["totalBins"] = BinGridTracker(
            [(float(axis["domainMin"]), float(axis["domainMax"])) for axis in selected_goal["axes"]],
            [float(axis["resolution"]) for axis in selected_goal["axes"]],
        ).total_bins
        warnings = out_of_domain_warnings(selected_goal, cluster_vector, peer_group, peer_rows)
        pending_cluster = make_cluster_record(
            goal,
            selected_goal,
            cluster_vector,
            dataset_meta,
            analysis_at_upload=analysis_snapshot(None, axis_names, len(peer_group), warnings, str(exc), coverage_info=coverage_info),
        )
        if should_save_data_clusters():
            saved_cluster, saved_cluster_is_new = save_data_cluster(pending_cluster)
        diagnostics = peer_filter_diagnostics(str(goal["id"]), axis_names)
        stored_count = int(diagnostics["compatibleAxisCount"])
        saved_text = "saved" if saved_cluster_is_new else "already stored"
        raise ValueError(
            "Only saved clusters with the same Experiment Goal and Axis configuration are used as the Peer Group. "
            f"Stored Peer Group N={stored_count}, required minimum N={len(axis_names) + 1}. "
            f"Peer filter diagnostics: {format_peer_filter_error(diagnostics)}. "
            f"This CSV was sanitized into a numeric cluster vector and {saved_text}, but analysis is limited until enough prior clusters exist. "
            f"Reason: {exc}"
        ) from exc

    result = analysis["result"]
    result_payload = analysis["resultPayload"]
    peer_group = analysis["peerGroup"]
    config = analysis["config"]
    pending_cluster = make_cluster_record(
        goal,
        selected_goal,
        cluster_vector,
        dataset_meta,
        analysis_at_upload=analysis["snapshot"],
    )
    if should_save_data_clusters():
        saved_cluster, saved_cluster_is_new = save_data_cluster(pending_cluster)

    summary = build_summary(result)
    summary.append("Peer Group uses only saved clusters that match the selected Experiment Goal and Axis configuration.")
    if analysis["warnings"]:
        summary.append(f"{len(analysis['warnings'])} out-of-domain value(s) were clipped for bin calculations and surfaced as warnings.")
    if saved_cluster:
        status = "saved as a new cluster" if saved_cluster_is_new else "detected as an existing duplicate and not saved again"
        summary.append(f"Raw uploaded rows, filename, and unmapped columns were not stored; only the axis vector was {status}.")

    return {
        "meta": {
            "experiment_goal": goal["name"],
            "goal_id": goal["id"],
            "peer_group_key": key,
            "target_rows": dataset_meta["row_count"],
            "cluster_vector_rows": dataset_meta.get("cluster_vector_row_count"),
            "row_level_valid_rows": dataset_meta.get("bin_occupancy_meta", {}).get("validMultidimensionalRowCount"),
            "uploaded_columns": dataset_meta["columns"],
            "summary_method": dataset_meta["summary_method"],
            "peer_group_size": int(len(peer_group)),
            "axis_names": config.axis_names,
            "axes": selected_goal["axes"],
            "available_axes": goal["axes"],
            "config": asdict(config),
            "cluster_definition": dataset_meta["cluster_definition"],
            "analysis_timestamp": analysis["snapshot"]["analysisTimestamp"],
            "coverageBasis": analysis["coverageInfo"]["coverageBasis"],
            "coverageEligibleClusterCount": analysis["coverageInfo"]["coverageEligibleClusterCount"],
            "coverageLegacyExcludedClusterCount": analysis["coverageInfo"]["coverageLegacyExcludedClusterCount"],
            "coverageAxisSignatureExcludedClusterCount": analysis["coverageInfo"]["coverageAxisSignatureExcludedClusterCount"],
            "rowLevelObservationCount": analysis["coverageInfo"]["rowLevelObservationCount"],
            "occupiedBins": analysis["coverageInfo"]["occupiedBins"],
            "totalBins": analysis["coverageInfo"]["totalBins"],
            "storage_policy": "Raw upload rows, filenames, and unmapped columns are not stored.",
        },
        "result": result_payload,
        "summary": summary,
        "confidenceReasons": confidence_reasons(result, analysis["warnings"]),
        "visualizations": build_report_visualizations(selected_goal, peer_group, cluster_vector, result, analysis["coverageInfo"]),
        "clusters": list_cluster_summaries(),
        "peerCounts": bootstrap_peer_counts(),
        "peerSubsetCounts": bootstrap_peer_subset_counts(),
        "savedDataCluster": None
        if saved_cluster is None
        else {
            "id": saved_cluster["id"],
            "isNew": saved_cluster_is_new,
            "axisNames": saved_cluster["axisNames"],
            "rowCount": saved_cluster["rowCount"],
            "clusterVectorRowCount": saved_cluster.get("clusterVectorRowCount"),
            "storeFile": storage_label(CLUSTER_STORE_PATH),
            "storagePolicy": saved_cluster["storagePolicy"],
        },
    }


def find_goal(goal_id: str) -> dict[str, Any]:
    goals = normalize_goals_for_display(load_goal_store())
    goal = next((item for item in goals if item["id"] == goal_id), None)
    if goal is None:
        raise ValueError("Selected Experiment Goal does not exist.")
    return goal


def analyze_batch_request(payload: dict[str, Any]) -> dict[str, Any]:
    goal = find_goal(str(payload.get("goalId", "")))
    files = payload.get("files", [])
    if not isinstance(files, list) or not files:
        raise ValueError("Batch payload must include at least one file.")
    selected_axis_names = payload.get("selectedAxes") or [axis["name"] for axis in goal["axes"]]
    if not isinstance(selected_axis_names, list):
        raise ValueError("selectedAxes must be a list.")
    selected_goal = goal_subset(goal, [str(name) for name in selected_axis_names])
    axis_names = [axis["name"] for axis in selected_goal["axes"]]
    source_batch_id = str(payload.get("sourceBatchId") or f"batch_{uuid.uuid4().hex}")
    seen_fingerprints: set[str] = set()
    items: list[dict[str, Any]] = []

    for index, file_item in enumerate(files, start=1):
        display_name = str(file_item.get("displayName") or file_item.get("name") or f"file_{index}")
        rows = file_item.get("rows", [])
        axis_mapping = file_item.get("axisMapping") or payload.get("axisMapping") or {}
        item_payload: dict[str, Any] = {
            "displayName": display_name,
            "index": index,
            "analysisSuccess": False,
            "saveable": False,
            "duplicate": False,
            "duplicateExisting": False,
            "duplicateInBatch": False,
            "axisMappingStatus": "unmapped",
        }
        try:
            if not isinstance(rows, list) or not rows:
                raise ValueError("Uploaded rows are empty.")
            if not isinstance(axis_mapping, dict):
                raise ValueError("Axis mapping is missing.")
            missing_axes = [axis for axis in axis_names if not axis_mapping.get(axis)]
            if missing_axes:
                raise ValueError(f"Missing mappings for axes: {', '.join(missing_axes)}")
            cluster_vector, dataset_meta = build_cluster_vector(rows, axis_mapping, selected_goal)
            item_payload.update(
                {
                    "rowCount": dataset_meta["row_count"],
                    "clusterVectorRowCount": dataset_meta.get("cluster_vector_row_count"),
                    "rowLevelValidCount": dataset_meta.get("bin_occupancy_meta", {}).get("validMultidimensionalRowCount"),
                    "axisMappingStatus": "mapped",
                    "clusterVector": [round(float(value), 6) for value in cluster_vector],
                }
            )
            try:
                analysis = run_vector_analysis(goal, selected_goal, cluster_vector)
                result_payload = analysis["resultPayload"]
                snapshot = analysis["snapshot"]
                item_payload.update(
                    {
                        "analysisSuccess": True,
                        "analysisSummary": {
                            "heterogeneity": result_payload["heterogeneity"],
                            "confidence": result_payload["confidence"],
                            "engine": result_payload["engine"],
                            "peer_group_size": len(analysis["peerGroup"]),
                            "coverage_eligible_cluster_count": analysis["coverageInfo"]["coverageEligibleClusterCount"],
                            "row_level_observation_count": analysis["coverageInfo"]["rowLevelObservationCount"],
                        },
                        "result": result_payload,
                        "confidenceReasons": confidence_reasons(analysis["result"], analysis["warnings"]),
                        "summary": build_summary(analysis["result"]),
                    }
                )
            except ValueError as exc:
                peer_clusters = analysis_peer_clusters(goal, axis_names)
                peer_rows = [{"id": str(cluster.get("id", "")), "source": "stored", "values": [float(value) for value in cluster["values"]]} for cluster in peer_clusters]
                peer_group = peer_matrix(peer_rows)
                if peer_group.size == 0:
                    peer_group = np.empty((0, len(axis_names)), dtype=float)
                coverage_info = build_global_bin_counts(peer_clusters, axis_names)
                coverage_info["totalBins"] = BinGridTracker(
                    [(float(axis["domainMin"]), float(axis["domainMax"])) for axis in selected_goal["axes"]],
                    [float(axis["resolution"]) for axis in selected_goal["axes"]],
                ).total_bins
                warnings = out_of_domain_warnings(selected_goal, cluster_vector, peer_group, peer_rows)
                snapshot = analysis_snapshot(None, axis_names, len(peer_group), warnings, str(exc), coverage_info=coverage_info)
                item_payload["analysisError"] = str(exc)
                item_payload["analysisSummary"] = {
                    "heterogeneity": None,
                    "confidence": None,
                    "engine": None,
                    "peer_group_size": len(peer_group),
                    "coverage_eligible_cluster_count": coverage_info["coverageEligibleClusterCount"],
                    "row_level_observation_count": coverage_info["rowLevelObservationCount"],
                }

            record = make_cluster_record(
                goal,
                selected_goal,
                cluster_vector,
                dataset_meta,
                source_batch_id=source_batch_id,
                analysis_at_upload=snapshot,
            )
            duplicate = duplicate_status(record)
            duplicate_in_batch = record["fingerprint"] in seen_fingerprints
            seen_fingerprints.add(record["fingerprint"])
            item_payload.update(
                {
                    "fingerprint": record["fingerprint"],
                    "duplicate": bool(duplicate["isDuplicate"] or duplicate_in_batch),
                    "duplicateExisting": bool(duplicate["isDuplicate"]),
                    "duplicateClusterId": duplicate["duplicateClusterId"],
                    "duplicateInBatch": duplicate_in_batch,
                    "saveable": not duplicate["isDuplicate"] and not duplicate_in_batch,
                    "pendingRecord": record,
                }
            )
        except Exception as exc:
            item_payload["error"] = str(exc)
        items.append(item_payload)

    return {
        "sourceBatchId": source_batch_id,
        "meta": {
            "experiment_goal": goal["name"],
            "goal_id": goal["id"],
            "axis_names": axis_names,
            "peer_group_key": peer_group_key(str(goal["id"]), axis_names),
            "storage_policy": "Batch preview keeps display names only in browser memory; saved records contain sanitized numeric vectors only.",
        },
        "items": items,
        "clusters": list_cluster_summaries(),
        "peerCounts": bootstrap_peer_counts(),
        "peerSubsetCounts": bootstrap_peer_subset_counts(),
    }


def batch_save_request(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("No selected cluster records were provided.")
    saved: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        saved_cluster, is_new = save_peer_cluster(record)
        saved.append({"id": saved_cluster["id"], "isNew": is_new, "fingerprint": saved_cluster["fingerprint"]})
    first_record = records[0] if isinstance(records[0], dict) else {}
    goal_id = str(payload.get("goalId") or first_record.get("goalId") or "").strip()
    selected_axis_names = payload.get("selectedAxisNames") or payload.get("selectedAxes") or first_record.get("axisNames") or []
    selected_axis_names = [str(name).strip() for name in selected_axis_names]
    diagnostics = peer_filter_diagnostics(goal_id, selected_axis_names) if goal_id and selected_axis_names else {}
    return {
        "saved": saved,
        "totalStoredClusters": diagnostics.get("totalClusters", len(load_cluster_store())),
        "sameGoalClusterCount": diagnostics.get("sameGoalCount", 0),
        "compatiblePeerCountForSelectedAxes": diagnostics.get("compatibleAxisCount", 0),
        "selectedAxisNames": selected_axis_names,
        "peerGroupKey": peer_group_key(goal_id, selected_axis_names) if goal_id and selected_axis_names else "",
        "peerCounts": bootstrap_peer_counts(),
        "peerSubsetCounts": bootstrap_peer_subset_counts(),
        "clusters": list_cluster_summaries(),
    }


def cluster_by_id(cluster_id: str) -> dict[str, Any]:
    wanted = str(cluster_id or "").strip()
    for cluster in load_cluster_store():
        if str(cluster.get("id")) == wanted:
            return cluster
    raise ValueError("Cluster not found.")


def reevaluate_cluster(cluster_id: str) -> dict[str, Any]:
    cluster = cluster_by_id(cluster_id)
    goal = find_goal(str(cluster["goalId"]))
    selected_goal = goal_subset(goal, [str(name) for name in cluster["axisNames"]])
    target = np.asarray(cluster["values"], dtype=float)
    uploaded = normalize_analysis_snapshot(cluster.get("analysisAtUpload"))
    try:
        analysis = run_vector_analysis(goal, selected_goal, target, exclude_cluster_id=str(cluster["id"]))
        current = analysis["resultPayload"]
        current_peer_group_size = len(analysis["peerGroup"])
        confidence_delta = None if uploaded.get("confidence") is None else round(float(current["confidence"]) - float(uploaded["confidence"]), 6)
        heterogeneity_delta = None if uploaded.get("heterogeneity") is None else round(float(current["heterogeneity"]) - float(uploaded["heterogeneity"]), 6)
        interpretation = reevaluation_interpretation(uploaded, current)
        return {
            "clusterId": cluster["id"],
            "uploadedAt": cluster.get("uploadedAt"),
            "uploaded": uploaded,
            "current": current,
            "currentPeerGroupSize": current_peer_group_size,
            "confidenceDelta": confidence_delta,
            "heterogeneityDelta": heterogeneity_delta,
            "interpretation": interpretation,
        }
    except ValueError as exc:
        return {
            "clusterId": cluster["id"],
            "uploadedAt": cluster.get("uploadedAt"),
            "uploaded": uploaded,
            "current": None,
            "currentPeerGroupSize": len(analysis_peer_rows(goal, [axis["name"] for axis in selected_goal["axes"]], str(cluster["id"]))),
            "error": str(exc),
            "interpretation": ["Current reevaluation is limited because the peer group is still too small after excluding this cluster."],
        }


def reevaluation_interpretation(uploaded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    uploaded_h = uploaded.get("heterogeneity")
    uploaded_c = uploaded.get("confidence")
    current_h = current.get("heterogeneity")
    if uploaded_h is not None and current_h is not None:
        if float(uploaded_h) >= 0.75 and float(current_h) <= 0.55:
            messages.append("처음에는 이질적으로 보였으나 현재 기준에서는 정상 범위에 가까워짐.")
        elif float(current_h) >= 0.75:
            messages.append("현재 누적 기준에서도 계속 이질적임.")
        else:
            messages.append("현재 누적 기준에서는 뚜렷한 이질성이 제한적임.")
    if uploaded_c is not None and float(uploaded_c) < 0.4:
        messages.append("업로드 당시 신뢰도가 낮아 초기 판단은 제한적이었음.")
    return messages or ["업로드 당시 결과와 현재 재평가 결과를 비교할 수 있음."]


def impact_result_payload(goal: dict[str, Any], selected_goal: dict[str, Any], target: np.ndarray, exclude_cluster_id: str | None) -> dict[str, Any]:
    try:
        analysis = run_vector_analysis(goal, selected_goal, target, exclude_cluster_id=exclude_cluster_id)
        return {"ok": True, "result": analysis["resultPayload"], "peerGroupSize": len(analysis["peerGroup"])}
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "peerGroupSize": len(analysis_peer_rows(goal, [axis["name"] for axis in selected_goal["axes"]], exclude_cluster_id)),
        }


def delete_impact_request(payload: dict[str, Any]) -> dict[str, Any]:
    cluster = cluster_by_id(str(payload.get("id", "")))
    goal = find_goal(str(cluster["goalId"]))
    selected_goal = goal_subset(goal, [str(name) for name in cluster["axisNames"]])
    target = np.asarray(cluster["values"], dtype=float)
    all_payload = impact_result_payload(goal, selected_goal, target, None)
    without_payload = impact_result_payload(goal, selected_goal, target, str(cluster["id"]))
    deltas: dict[str, Any] = {}
    if all_payload["ok"] and without_payload["ok"]:
        all_result = all_payload["result"]
        without_result = without_payload["result"]
        deltas = {
            "peerGroupN": [all_payload["peerGroupSize"], without_payload["peerGroupSize"]],
            "deltaConfidence": round(float(without_result["confidence"] - all_result["confidence"]), 6),
            "deltaCoverage": round(float(without_result["coverage_C"] - all_result["coverage_C"]), 6),
            "deltaEquitability": round(float(without_result["equitability_E"] - all_result["equitability_E"]), 6),
            "deltaHeterogeneity": round(float(without_result["heterogeneity"] - all_result["heterogeneity"]), 6),
            "deltaD2": round(float(without_result["D2"] - all_result["D2"]), 6),
            "deltaCenterNorm": round(float(np.linalg.norm(np.asarray(all_result["center"]) - np.asarray(without_result["center"]))), 6),
        }
    config = experiment_config_from_goal(selected_goal)
    tracker = BinGridTracker(config.domain_range, config.resolution)
    for row in analysis_peer_rows(goal, [axis["name"] for axis in selected_goal["axes"]]):
        tracker.add(np.asarray(row["values"], dtype=float))
    bin_uniqueness = tracker.count_for(target) == 1
    return {
        "cluster": {
            "id": cluster["id"],
            "goalName": cluster.get("goalName"),
            "axisNames": cluster.get("axisNames"),
            "values": cluster.get("values"),
            "uploadedAt": cluster.get("uploadedAt"),
        },
        "currentAll": all_payload,
        "withoutCluster": without_payload,
        "deltas": deltas,
        "binUniqueness": bin_uniqueness,
    }


def reevaluate_request(payload: dict[str, Any]) -> dict[str, Any]:
    cluster_id = str(payload.get("id", "")).strip()
    if cluster_id:
        return {"items": [reevaluate_cluster(cluster_id)]}
    return {"items": [reevaluate_cluster(str(cluster["id"])) for cluster in load_cluster_store()]}


def export_report_request(payload: dict[str, Any]) -> dict[str, Any]:
    report = payload.get("report")
    if not isinstance(report, dict):
        raise ValueError("Report payload is required.")
    export_format = str(payload.get("format") or "json").lower()
    timestamp = str(report.get("meta", {}).get("analysis_timestamp") or utc_now_iso()).replace(":", "-")
    if export_format == "json":
        return {
            "filename": f"leesin_report_{timestamp}.json",
            "mime": "application/json",
            "content": json.dumps(report, ensure_ascii=False, indent=2),
        }
    if export_format == "csv":
        meta = report.get("meta", {})
        result = report.get("result", {})
        output = io.StringIO()
        fieldnames = [
            "experiment_goal",
            "goal_id",
            "axis_names",
            "target_vector",
            "peer_group_size",
            "engine",
            "is_normal",
            "D2",
            "p_value",
            "heterogeneity",
            "confidence",
            "sample_size_Z",
            "coverage_C",
            "equitability_E",
            "w_eff",
            "total_bins",
            "occupied_bins",
            "mardia_skew_stat",
            "mardia_skew_pval",
            "mardia_kurt_stat",
            "mardia_kurt_pval",
            "b2p",
            "analysis_timestamp",
            "contributions",
            "axis_ablation_sensitivity",
            "out_of_domain_warnings",
            "confidence_reasons",
            "summary_messages",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "experiment_goal": meta.get("experiment_goal"),
                "goal_id": meta.get("goal_id"),
                "axis_names": json.dumps(meta.get("axis_names", []), ensure_ascii=False),
                "target_vector": json.dumps(report.get("visualizations", {}).get("targetVector", []), ensure_ascii=False),
                "peer_group_size": meta.get("peer_group_size"),
                "engine": result.get("engine"),
                "is_normal": result.get("is_normal"),
                "D2": result.get("D2"),
                "p_value": result.get("p_value"),
                "heterogeneity": result.get("heterogeneity"),
                "confidence": result.get("confidence"),
                "sample_size_Z": result.get("sample_size_Z"),
                "coverage_C": result.get("coverage_C"),
                "equitability_E": result.get("equitability_E"),
                "w_eff": result.get("w_eff"),
                "total_bins": result.get("total_bins"),
                "occupied_bins": result.get("occupied_bins"),
                "mardia_skew_stat": result.get("mardia_skew_stat"),
                "mardia_skew_pval": result.get("mardia_skew_pval"),
                "mardia_kurt_stat": result.get("mardia_kurt_stat"),
                "mardia_kurt_pval": result.get("mardia_kurt_pval"),
                "b2p": result.get("b2p"),
                "analysis_timestamp": meta.get("analysis_timestamp"),
                "contributions": json.dumps(result.get("contributions", []), ensure_ascii=False),
                "axis_ablation_sensitivity": json.dumps(result.get("axisAblationSensitivity", []), ensure_ascii=False),
                "out_of_domain_warnings": json.dumps(result.get("outOfDomainWarnings", []), ensure_ascii=False),
                "confidence_reasons": json.dumps(report.get("confidenceReasons", []), ensure_ascii=False),
                "summary_messages": json.dumps(report.get("summary", []), ensure_ascii=False),
            }
        )
        return {"filename": f"leesin_report_{timestamp}.csv", "mime": "text/csv", "content": output.getvalue()}
    if export_format == "html":
        title = html.escape(str(report.get("meta", {}).get("experiment_goal", "Leesin Report")))
        pretty = html.escape(json.dumps(report, ensure_ascii=False, indent=2))
        return {
            "filename": f"leesin_report_{timestamp}.html",
            "mime": "text/html",
            "content": f"<!doctype html><meta charset='utf-8'><title>{title}</title><h1>{title}</h1><pre>{pretty}</pre>",
        }
    raise ValueError("Unsupported export format.")


analyze_request = analyze_request_v2


def bootstrap_peer_counts() -> dict[str, int]:
    peer_counts: dict[str, int] = {}
    for goal in normalize_goals_for_display(load_goal_store()):
        full_axis_names = [axis["name"] for axis in goal["axes"]]
        peer_counts[goal["id"]] = int(peer_filter_diagnostics(str(goal["id"]), full_axis_names)["compatibleAxisCount"])
    return peer_counts


def bootstrap_peer_subset_counts() -> dict[str, dict[str, int]]:
    counts = {}
    for goal in normalize_goals_for_display(load_goal_store()):
        counts[goal["id"]] = peer_group_subset_counts(goal)
    return counts


def goal_bin_preview(goal: dict[str, Any]) -> dict[str, Any]:
    axis_names = [axis["name"] for axis in goal["axes"]]
    stored_clusters = load_peer_clusters(str(goal["id"]), axis_names)
    coverage_info = build_global_bin_counts(stored_clusters, axis_names)
    axis_previews = []
    total_bins = 1
    for axis_index, axis in enumerate(goal["axes"]):
        total = max(1, int(np.ceil((float(axis["domainMax"]) - float(axis["domainMin"])) / float(axis["resolution"]))))
        total_bins *= total
        occupied = len((coverage_info["axisBinCounts"].get(str(axis["name"])) or {}))
        axis_previews.append(
            {
                "axis": axis["name"],
                "totalBins": total,
                "occupiedBins": occupied,
                "coverage": occupied / total if total else 0.0,
            }
        )
    occupied_bins = len(coverage_info["binCounts"])
    warnings = []
    if total_bins > 100000:
        warnings.append("Resolution이 현재 데이터 수에 비해 과도하게 세밀할 수 있음.")
    if coverage_info["coverageLegacyExcludedClusterCount"]:
        warnings.append("일부 기존 군집은 row-level bin 정보가 없어 coverage 계산에서 제외됩니다. 정확한 coverage를 원하면 CSV를 다시 업로드하세요.")
    if coverage_info["coverageAxisSignatureExcludedClusterCount"]:
        warnings.append("Axis 구성이 다른 군집은 row-level coverage 계산에서 제외되고 heterogeneity peer로만 사용할 수 있습니다.")
    return {
        "basis": coverage_info["coverageBasis"],
        "axisBins": axis_previews,
        "totalBins": total_bins,
        "occupiedBins": occupied_bins,
        "estimatedCoverage": occupied_bins / total_bins if total_bins else 0.0,
        "coverageEligibleClusterCount": coverage_info["coverageEligibleClusterCount"],
        "coverageLegacyExcludedClusterCount": coverage_info["coverageLegacyExcludedClusterCount"],
        "coverageAxisSignatureExcludedClusterCount": coverage_info["coverageAxisSignatureExcludedClusterCount"],
        "rowLevelObservationCount": coverage_info["rowLevelObservationCount"],
        "warning": " ".join(warnings),
        "warnings": warnings,
    }


def build_bootstrap_payload(admin_allowed: bool) -> dict[str, Any]:
    goals = normalize_goals_for_display(load_goal_store())
    peer_counts = bootstrap_peer_counts()
    peer_subset_counts = {}
    for goal in goals:
        peer_subset_counts[goal["id"]] = peer_group_subset_counts(goal)
    admin_auth_required = bool(os.environ.get("ADMIN_TOKEN")) and not admin_allowed
    return {
        "adminAllowed": admin_allowed,
        "adminAuthRequired": admin_auth_required,
        "goals": goals,
        "clusters": list_cluster_summaries(),
        "peerCounts": peer_counts,
        "peerSubsetCounts": peer_subset_counts,
        "goalBinPreview": {goal["id"]: goal_bin_preview(goal) for goal in goals},
        "acceptedUploadTypes": [".csv", ".tsv", ".txt"],
        "storage": {
            "storeDir": storage_label(STORE_DIR),
            "goalStoreFile": storage_label(GOAL_STORE_PATH),
            "clusterStoreFile": storage_label(CLUSTER_STORE_PATH),
            "clusterCount": len(load_cluster_store()),
            "savedItems": ["Experiment Goal 설정", "비식별 numeric cluster vector", "row-level bin count summary"],
            "unsavedItems": ["원본 CSV 파일", "파일명", "비매핑 column", "개인정보 column"],
        },
        "domainDefinitions": {
            "cluster": "CSV 파일 하나는 하나의 데이터 군집으로 취급됩니다. CSV 내부 row들은 해당 군집의 반복 관측값이며, 시스템은 row들을 axis별 대표값으로 집계하여 하나의 cluster vector를 만듭니다.",
            "coverage": "Coverage와 Equitability는 cluster 대표값이 아니라 저장된 row-level bin occupancy summary 기준으로 계산됩니다.",
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
                self._send_json(analyze_request_v2(payload))
                return
            if parsed.path == "/api/analyze-batch":
                self._send_json(analyze_batch_request(payload))
                return
            if parsed.path == "/api/export/report":
                self._send_json(export_report_request(payload))
                return
            if parsed.path == "/api/admin/clusters/batch-save":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin Token is required."}, status=HTTPStatus.FORBIDDEN)
                    return
                self._send_json(batch_save_request(payload))
                return
            if parsed.path == "/api/admin/clusters/impact":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin Token is required."}, status=HTTPStatus.FORBIDDEN)
                    return
                self._send_json(delete_impact_request(payload))
                return
            if parsed.path == "/api/admin/clusters/reevaluate":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin Token is required."}, status=HTTPStatus.FORBIDDEN)
                    return
                self._send_json(reevaluate_request(payload))
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
                self._send_json({"savedGoal": saved_goal, "goals": normalize_goals_for_display(goals), "clusters": list_cluster_summaries(), "peerCounts": bootstrap_peer_counts(), "peerSubsetCounts": bootstrap_peer_subset_counts(), "goalBinPreview": {goal["id"]: goal_bin_preview(goal) for goal in normalize_goals_for_display(goals)}})
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
                normalized_goals = normalize_goals_for_display(goals)
                self._send_json({"goals": normalized_goals, "clusters": list_cluster_summaries(), "peerCounts": bootstrap_peer_counts(), "peerSubsetCounts": bootstrap_peer_subset_counts(), "goalBinPreview": {goal["id"]: goal_bin_preview(goal) for goal in normalized_goals}})
                return
            if parsed.path == "/api/admin/clusters/delete":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin Token이 필요합니다."}, status=HTTPStatus.FORBIDDEN)
                    return
                deleted = delete_peer_cluster(str(payload.get("id", "")))
                if not deleted:
                    raise ValueError("삭제할 군집을 찾을 수 없습니다.")
                self._send_json({"deleted": True, "clusters": list_cluster_summaries(), "peerCounts": bootstrap_peer_counts(), "peerSubsetCounts": bootstrap_peer_subset_counts()})
                return
        except Exception as exc:
            self._send_json(
                {
                    "error": str(exc),
                    "clusters": list_cluster_summaries(),
                    "peerCounts": bootstrap_peer_counts(),
                    "peerSubsetCounts": bootstrap_peer_subset_counts(),
                },
                status=HTTPStatus.BAD_REQUEST,
            )
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
