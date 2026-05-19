from __future__ import annotations

import argparse
import csv
import html
import io
import ipaddress
import itertools
import json
import math
import os
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from feasible_mask import (
    compute_valid_bin_mask_for_axes,
    expression_axis_names,
    is_bin_key_feasible,
    normalize_feasible_expressions,
)
from models import K_M, DensityDiagnosisResult, ExperimentConfig
from stats_engine import BinGridTracker, DensityGridAnalyzer
from storage import (
    CLUSTER_STORE_PATH,
    GOAL_STORE_PATH,
    STORE_DIR,
    axis_subset_key,
    bin_occupancy_hash,
    cluster_fingerprint,
    cluster_fingerprint_payload,
    delete_peer_cluster,
    explain_peer_filter,
    grid_signature_from_axes,
    init_database,
    list_cluster_summaries,
    load_cluster_store,
    load_goal_store,
    load_peer_clusters,
    normalize_analysis_snapshot,
    normalize_axis_name,
    normalize_goals_for_display,
    peer_group_key,
    save_data_cluster,
    save_cluster_store,
    save_peer_cluster,
    save_goal_store,
    should_save_data_clusters,
    storage_label,
    utc_now_iso,
    validate_goal,
)


TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"
PROJECTION_ROW_TUPLE_LIMIT = 50000


def canonical_axis_order(axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(axis) for axis in axes), key=lambda axis: normalize_axis_name(axis.get("name")))


def goal_subset(goal: dict[str, Any], selected_axis_names: list[str] | None = None) -> dict[str, Any]:
    if not selected_axis_names:
        axes = canonical_axis_order(goal["axes"])
    else:
        requested = {normalize_axis_name(name) for name in selected_axis_names if normalize_axis_name(name)}
        axes = canonical_axis_order([axis for axis in goal["axes"] if normalize_axis_name(axis["name"]) in requested])
    if not axes:
        raise ValueError("분석에 포함할 Axis를 하나 이상 선택하세요.")
    axis_name_set = {str(axis["name"]) for axis in axes}
    feasible_expressions = []
    for expression in normalize_feasible_expressions(goal.get("feasibleDomainExpressions")):
        try:
            if expression_axis_names(expression).issubset(axis_name_set):
                feasible_expressions.append(expression)
        except ValueError:
            continue
    return {
        "id": goal["id"],
        "name": goal["name"],
        "K_m": float(goal.get("K_m", K_M)),
        "axes": axes,
        "feasibleDomainRules": list(goal.get("feasibleDomainRules") or []),
        "feasibleDomainAdvancedExpressions": normalize_feasible_expressions(goal.get("feasibleDomainAdvancedExpressions")),
        "feasibleDomainExpressions": feasible_expressions,
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


def feasible_expressions_for_goal(selected_goal: dict[str, Any]) -> list[str]:
    return normalize_feasible_expressions(selected_goal.get("feasibleDomainExpressions"))


def feasible_mask_info_for_goal(selected_goal: dict[str, Any]) -> dict[str, Any]:
    return compute_valid_bin_mask_for_axes(
        canonical_axis_order(selected_goal["axes"]),
        feasible_expressions_for_goal(selected_goal),
    )


def is_bin_key_feasible_for_goal(bin_key: str, selected_goal: dict[str, Any], cache: dict[str, bool] | None = None) -> bool:
    return is_bin_key_feasible(
        bin_key,
        canonical_axis_order(selected_goal["axes"]),
        feasible_expressions_for_goal(selected_goal),
        cache,
    )


def compute_row_level_bin_occupancy(
    rows: list[dict[str, Any]],
    axis_mapping: dict[str, str],
    selected_goal: dict[str, Any],
) -> dict[str, Any]:
    axes = canonical_axis_order(selected_goal["axes"])
    axis_mapping_by_key = {normalize_axis_name(axis_name): column for axis_name, column in axis_mapping.items()}
    bin_occupancy: dict[str, int] = {}
    axis_bin_occupancy: dict[str, dict[str, int]] = {str(axis["name"]): {} for axis in axes}
    row_bin_tuples: list[list[int]] = []
    valid_multidimensional = 0
    invalid_rows = 0
    out_of_domain_rows = 0
    masked_out_rows = 0
    masked_out_bin_occupancy: dict[str, int] = {}
    feasible_cache: dict[str, bool] = {}

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
            if not is_bin_key_feasible_for_goal(key, selected_goal, feasible_cache):
                masked_out_rows += 1
                masked_out_bin_occupancy[key] = masked_out_bin_occupancy.get(key, 0) + 1
                continue
            for axis, index in zip(axes, multidimensional_indices):
                axis_name = str(axis["name"])
                axis_counts = axis_bin_occupancy[axis_name]
                axis_counts[str(index)] = axis_counts.get(str(index), 0) + 1
            bin_occupancy[key] = bin_occupancy.get(key, 0) + 1
            row_bin_tuples.append(list(multidimensional_indices))
            valid_multidimensional += 1

    return {
        "bin_occupancy": bin_occupancy,
        "axis_bin_occupancy": axis_bin_occupancy,
        "row_bin_tuples": row_bin_tuples,
        "bin_occupancy_meta": {
            "version": 1,
            "basis": "row_level",
            "validMultidimensionalRowCount": valid_multidimensional,
            "feasibleValidRowCount": valid_multidimensional,
            "invalidRowCount": invalid_rows,
            "outOfDomainRowCount": out_of_domain_rows,
            "maskedOutRowCount": masked_out_rows,
            "infeasibleRowCount": masked_out_rows,
            "maskedOutBinCount": len(masked_out_bin_occupancy),
            "totalRows": len(rows),
        },
        "masked_out_bin_occupancy": masked_out_bin_occupancy,
    }


def build_dataset_summary(
    rows: list[dict[str, Any]],
    axis_mapping: dict[str, str],
    selected_goal: dict[str, Any],
    method: str = "mean",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a sanitized dataset summary.

    The mean vector is retained for display/storage compatibility, but primary
    density analysis uses row-level bin occupancy in the canonical axis order.
    """
    if method != "mean":
        raise ValueError("Only mean compatibility summarization is currently supported.")
    if not rows:
        raise ValueError("업로드된 데이터가 비어 있습니다.")

    axes = canonical_axis_order(selected_goal["axes"])
    means = np.zeros(len(axes), dtype=float)
    m2 = np.zeros(len(axes), dtype=float)
    axis_numeric_counts = np.zeros(len(axes), dtype=int)
    row_level_vectors: list[list[float]] = []
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

        row_level_vectors.append([float(value) for value in row_values])
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
        "row_level_vectors": row_level_vectors,
        "row_level_vector_axis_order": [str(axis["name"]) for axis in axes],
        "row_level_vector_count": len(row_level_vectors),
        "row_level_vector_basis": "valid_multidimensional_numeric_rows",
        "bin_occupancy": occupancy["bin_occupancy"],
        "axis_bin_occupancy": occupancy["axis_bin_occupancy"],
        "row_bin_tuples": occupancy["row_bin_tuples"],
        "bin_occupancy_meta": occupancy["bin_occupancy_meta"],
        "cluster_definition": "Stored record keeps a mean vector for compatibility; primary analysis re-bins row-level sanitized axis vectors.",
    }


build_cluster_vector = build_dataset_summary
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


def axis_total_bins(axis: dict[str, Any]) -> int:
    domain_min = float(axis["domainMin"])
    domain_max = float(axis["domainMax"])
    resolution = float(axis["resolution"])
    return max(1, int(np.ceil((domain_max - domain_min) / resolution)))


def projection_pair_key(x_axis: str, y_axis: str) -> str:
    return f"{x_axis}|{y_axis}"


def projection_axis_meta(axes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(axis["name"]): {
            "domainMin": float(axis["domainMin"]),
            "domainMax": float(axis["domainMax"]),
            "resolution": float(axis["resolution"]),
            "totalBins": axis_total_bins(axis),
            "unit": axis.get("unit", ""),
            "label": axis_display_label(axis),
        }
        for axis in axes
    }


def empty_projection_matrix(x_bins: int, y_bins: int) -> list[list[int]]:
    return [[0 for _ in range(max(0, x_bins))] for _ in range(max(0, y_bins))]


def normalize_row_bin_tuples(row_bin_tuples: list[Any], expected_length: int) -> list[list[int]]:
    normalized: list[list[int]] = []
    for raw_tuple in row_bin_tuples or []:
        if not isinstance(raw_tuple, (list, tuple)) or len(raw_tuple) != expected_length:
            continue
        try:
            tuple_values = [int(value) for value in raw_tuple]
        except (TypeError, ValueError):
            continue
        if any(value < 0 for value in tuple_values):
            continue
        normalized.append(tuple_values)
    return normalized


def build_pair_projection_from_bin_counts(
    bin_counts: dict[str, Any],
    axis_order: list[str],
    axis_meta: dict[str, dict[str, Any]],
    x_axis: str,
    y_axis: str,
) -> dict[str, Any]:
    x_bins = int(axis_meta.get(x_axis, {}).get("totalBins") or 0)
    y_bins = int(axis_meta.get(y_axis, {}).get("totalBins") or 0)
    counts = empty_projection_matrix(x_bins, y_bins)
    axis_positions = {axis_name: index for index, axis_name in enumerate(axis_order)}
    if x_axis not in axis_positions or y_axis not in axis_positions:
        return {"xAxis": x_axis, "yAxis": y_axis, "xBins": x_bins, "yBins": y_bins, "counts": counts, "maxCount": 0}
    x_position = axis_positions[x_axis]
    y_position = axis_positions[y_axis]

    for raw_key, raw_count in (bin_counts or {}).items():
        try:
            tuple_values = json.loads(str(raw_key))
            count = int(raw_count)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if count <= 0 or not isinstance(tuple_values, list) or len(tuple_values) != len(axis_order):
            continue
        try:
            x_bin = int(tuple_values[x_position])
            y_bin = int(tuple_values[y_position])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= x_bin < x_bins and 0 <= y_bin < y_bins:
            counts[y_bin][x_bin] += count

    max_count = max((max(row) for row in counts), default=0)
    return {"xAxis": x_axis, "yAxis": y_axis, "xBins": x_bins, "yBins": y_bins, "counts": counts, "maxCount": int(max_count)}


def build_pair_projection_from_row_tuples(
    row_bin_tuples: list[Any],
    axis_order: list[str],
    axis_meta: dict[str, dict[str, Any]],
    x_axis: str,
    y_axis: str,
) -> dict[str, Any]:
    x_bins = int(axis_meta.get(x_axis, {}).get("totalBins") or 0)
    y_bins = int(axis_meta.get(y_axis, {}).get("totalBins") or 0)
    counts = empty_projection_matrix(x_bins, y_bins)
    axis_positions = {axis_name: index for index, axis_name in enumerate(axis_order)}
    if x_axis not in axis_positions or y_axis not in axis_positions:
        return {"xAxis": x_axis, "yAxis": y_axis, "xBins": x_bins, "yBins": y_bins, "counts": counts, "maxCount": 0}
    x_position = axis_positions[x_axis]
    y_position = axis_positions[y_axis]

    for tuple_values in normalize_row_bin_tuples(row_bin_tuples, len(axis_order)):
        x_bin = tuple_values[x_position]
        y_bin = tuple_values[y_position]
        if 0 <= x_bin < x_bins and 0 <= y_bin < y_bins:
            counts[y_bin][x_bin] += 1

    max_count = max((max(row) for row in counts), default=0)
    return {"xAxis": x_axis, "yAxis": y_axis, "xBins": x_bins, "yBins": y_bins, "counts": counts, "maxCount": int(max_count)}


def filter_row_tuples_for_axis_pair(
    row_bin_tuples: list[Any],
    axis_order: list[str],
    x_axis: str,
    y_axis: str,
    x_bin: int,
    y_bin: int,
) -> list[list[int]]:
    axis_positions = {axis_name: index for index, axis_name in enumerate(axis_order)}
    if x_axis not in axis_positions or y_axis not in axis_positions:
        return []
    x_position = axis_positions[x_axis]
    y_position = axis_positions[y_axis]
    normalized = normalize_row_bin_tuples(row_bin_tuples, len(axis_order))
    return [tuple_values for tuple_values in normalized if tuple_values[x_position] == x_bin and tuple_values[y_position] == y_bin]


def crosshair_markers_for_selection(axis_pairs: list[list[str]], selection: dict[str, Any] | None) -> dict[str, dict[str, int]]:
    if not selection:
        return {}
    bins_by_axis = selection.get("binsByAxis") if isinstance(selection.get("binsByAxis"), dict) else {}
    markers: dict[str, dict[str, int]] = {}
    for pair in axis_pairs:
        if len(pair) != 2:
            continue
        x_axis, y_axis = str(pair[0]), str(pair[1])
        marker: dict[str, int] = {}
        if x_axis in bins_by_axis:
            marker["xBin"] = int(bins_by_axis[x_axis])
        if y_axis in bins_by_axis:
            marker["yBin"] = int(bins_by_axis[y_axis])
        if marker:
            markers[projection_pair_key(x_axis, y_axis)] = marker
    return markers


def sample_row_bin_tuples_for_payload(row_bin_tuples: list[list[int]], limit: int = PROJECTION_ROW_TUPLE_LIMIT) -> tuple[list[list[int]], bool]:
    if len(row_bin_tuples) <= limit:
        return row_bin_tuples, False
    if limit <= 0:
        return [], True
    step = len(row_bin_tuples) / limit
    sampled = [row_bin_tuples[min(len(row_bin_tuples) - 1, int(index * step))] for index in range(limit)]
    return sampled, True


def build_projection_explorer(
    goal: dict[str, Any],
    coverage_info: dict[str, Any] | None,
    target_row_bin_tuples: list[Any] | None,
) -> dict[str, Any]:
    axes = canonical_axis_order(goal["axes"])
    axis_order = [str(axis["name"]) for axis in axes]
    axis_meta = projection_axis_meta(axes)
    axis_pairs = [[x_axis, y_axis] for x_axis, y_axis in itertools.combinations(axis_order, 2)]
    target_tuples = normalize_row_bin_tuples(target_row_bin_tuples or [], len(axis_order))
    payload_tuples, tuple_sampled = sample_row_bin_tuples_for_payload(target_tuples)
    peer_bin_counts = coverage_info.get("binCounts", {}) if isinstance(coverage_info, dict) else {}

    peer_projections: dict[str, Any] = {}
    target_projections: dict[str, Any] = {}
    for x_axis, y_axis in axis_pairs:
        key = projection_pair_key(x_axis, y_axis)
        peer_projections[key] = build_pair_projection_from_bin_counts(peer_bin_counts, axis_order, axis_meta, x_axis, y_axis)
        target_projections[key] = build_pair_projection_from_row_tuples(target_tuples, axis_order, axis_meta, x_axis, y_axis)

    return {
        "axisOrder": axis_order,
        "axisMeta": axis_meta,
        "axisPairs": axis_pairs,
        "peerProjections": peer_projections,
        "targetProjections": target_projections,
        "feasibleMaskEnabled": bool(coverage_info.get("feasibleMaskEnabled")) if isinstance(coverage_info, dict) else False,
        "validBins": int(coverage_info.get("validBins") or 0) if isinstance(coverage_info, dict) else 0,
        "maskedBins": int(coverage_info.get("maskedBins") or 0) if isinstance(coverage_info, dict) else 0,
        "validDomainRatio": float(coverage_info.get("validDomainRatio") or 0.0) if isinstance(coverage_info, dict) else 0.0,
        "feasibleExpressions": list(coverage_info.get("feasibleExpressions") or []) if isinstance(coverage_info, dict) else [],
        "maskRenderingTodo": "TODO: render masked 2D projection regions as gray overlay.",
        "targetRowBinTuples": payload_tuples,
        "targetRowTupleSampled": tuple_sampled,
        "targetRowTupleLimit": PROJECTION_ROW_TUPLE_LIMIT,
        "targetRowTupleCount": len(target_tuples),
    }


def density_preview_metrics_from_coverage(selected_goal: dict[str, Any], coverage_info: dict[str, Any]) -> dict[str, Any]:
    bin_counts = coverage_info.get("binCounts", {}) if isinstance(coverage_info, dict) else {}
    counts = [int(value) for value in bin_counts.values() if int(value) > 0]
    peer_valid_rows = int(sum(counts))
    mask_info = feasible_mask_info_for_goal(selected_goal)
    total_bins = int(coverage_info.get("totalBins") or mask_info["totalBins"])
    valid_bins = int(coverage_info.get("validBins") or mask_info["validBins"])
    masked_bins = int(coverage_info.get("maskedBins") or mask_info["maskedBins"])
    occupied_bins = len(counts)
    observation_support = peer_valid_rows / (peer_valid_rows + float(selected_goal.get("K_m", K_M))) if peer_valid_rows > 0 else 0.0
    coverage = occupied_bins / valid_bins if valid_bins else 0.0
    if occupied_bins <= 1 or peer_valid_rows <= 0:
        equitability = 0.0
    else:
        proportions = [count / peer_valid_rows for count in counts]
        equitability = -sum(p * math.log(p) for p in proportions if p > 0) / math.log(occupied_bins)
    confidence = float((observation_support * coverage * equitability) ** (1.0 / 3.0)) if observation_support and coverage and equitability else 0.0
    return {
        "totalBins": int(total_bins),
        "validBins": int(valid_bins),
        "maskedBins": int(masked_bins),
        "validDomainRatio": round(float(valid_bins / total_bins), 6) if total_bins else 0.0,
        "feasibleMaskEnabled": bool(mask_info.get("feasibleMaskEnabled")),
        "feasibleExpressions": list(mask_info.get("feasibleExpressions") or []),
        "occupiedBins": int(occupied_bins),
        "peerValidRows": int(peer_valid_rows),
        "observationSupportZ": round(float(observation_support), 6),
        "coverageC": round(float(coverage), 6),
        "equitabilityE": round(float(equitability), 6),
        "confidence": round(float(confidence), 6),
        "eligibleRecords": int(coverage_info.get("coverageEligibleClusterCount") or 0),
        "legacyExcluded": int(coverage_info.get("coverageLegacyExcludedClusterCount") or 0),
        "gridSignatureExcluded": int(coverage_info.get("coverageGridSignatureExcludedClusterCount") or 0),
        "gridSignature": str(coverage_info.get("gridSignature") or ""),
    }


def build_report_visualizations(
    goal: dict[str, Any],
    peer_group: np.ndarray,
    target_vector: np.ndarray,
    result: DensityDiagnosisResult,
    coverage_info: dict[str, Any] | None = None,
    target_bin_counts: dict[str, int] | None = None,
    target_meta: dict[str, Any] | None = None,
    target_row_bin_tuples: list[Any] | None = None,
    target_row_vectors: list[Any] | None = None,
    target_row_vector_axis_order: list[str] | None = None,
) -> dict[str, Any]:
    axes = canonical_axis_order(goal["axes"])
    coverage_axes = []
    equitability_axes = []
    axis_bin_counts = coverage_info.get("axisBinCounts", {}) if isinstance(coverage_info, dict) else {}
    axis_bin_counts_by_key = {normalize_axis_name(axis_name): counts for axis_name, counts in axis_bin_counts.items()}

    for index, axis in enumerate(axes):
        total_axis_bins = max(1, int(np.ceil((float(axis["domainMax"]) - float(axis["domainMin"])) / float(axis["resolution"]))))
        row_level_counts = axis_bin_counts_by_key.get(normalize_axis_name(axis["name"]), {})
        distribution = build_axis_distribution_from_counts(row_level_counts, total_axis_bins)
        distribution_basis = "row_level_bin_occupancy"
        coverage_axes.append(
            {
                "axis": axis["name"],
                "label": axis_display_label(axis),
                "unit": axis.get("unit", ""),
                "domainMin": float(axis["domainMin"]),
                "domainMax": float(axis["domainMax"]),
                "resolution": float(axis["resolution"]),
                "targetValue": round(float(target_vector[index]), 6),
                "peerValues": [],
                "basis": distribution_basis,
                "fallbackReason": "",
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
    normalized_target_bins = {str(key): int(value) for key, value in (target_bin_counts or {}).items()}
    target_meta = target_meta or {}
    return {
        "observationSupport": {
            "peerObservationCount": int(result.peer_observation_count),
            "score": round(float(result.observation_support_S), 6),
        },
        "coverage": {"score": round(float(result.coverage_C), 6), "basis": visualization_basis, "axes": coverage_axes},
        "equitability": {
            "score": round(float(result.equitability_E), 6),
            "basis": visualization_basis,
            "status": "balanced" if result.equitability_E >= 0.5 else "imbalanced",
            "axes": equitability_axes,
        },
        "peerRows": [[round(float(value), 6) for value in row] for row in peer_group.tolist()],
        "targetVector": [round(float(value), 6) for value in target_vector.tolist()],
        "targetBinOccupancy": normalized_target_bins,
        "targetBinOccupancyMeta": {
            "validMultidimensionalRowCount": int(target_meta.get("validMultidimensionalRowCount") or result.valid_target_rows),
            "invalidRowCount": int(target_meta.get("invalidRowCount") or result.invalid_target_rows),
            "outOfDomainRowCount": int(target_meta.get("outOfDomainRowCount") or result.out_of_domain_rows),
            "maskedOutRowCount": int(target_meta.get("maskedOutRowCount") or result.masked_out_target_rows),
            "infeasibleRowCount": int(target_meta.get("infeasibleRowCount") or result.masked_out_target_rows),
            "totalRows": int(target_meta.get("totalRows") or result.target_total_rows),
        },
        "projectionExplorer": build_projection_explorer(goal, coverage_info or {}, target_row_bin_tuples or []),
        "gridPreviewMetrics": density_preview_metrics_from_coverage(goal, coverage_info or {}),
        "targetRowVectors": target_row_vectors or [],
        "targetRowVectorAxisOrder": target_row_vector_axis_order or [axis["name"] for axis in axes],
        "axisNames": [axis["name"] for axis in axes],
    }


def confidence_reasons(result: DensityDiagnosisResult, warnings: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    reasons = [
        {
            "label": "Observation Support",
            "score": round(float(result.observation_support_S), 4),
            "impact": "down" if result.observation_support_S < 0.6 else "stable",
            "message": "Observation Support uses peer row-level observations: S = peer rows / (peer rows + K_density). K_density currently reuses the goal K_m value.",
        },
        {
            "label": "Coverage",
            "score": round(float(result.coverage_C), 4),
            "impact": "down" if result.coverage_C < 0.3 else "stable",
            "message": "Coverage measures how much of the configured density grid is occupied by eligible peer row-level observations.",
        },
        {
            "label": "Equitability",
            "score": round(float(result.equitability_E), 4),
            "impact": "down" if result.equitability_E < 0.5 else "stable",
            "message": "Equitability measures whether peer observations are balanced across occupied density bins.",
        },
    ]
    if warnings:
        reasons.append(
            {
                "label": "Out-of-domain rows",
                "score": 0.0,
                "impact": "down",
                "message": f"{len(warnings)} target row-level warning(s) were reported. Out-of-domain rows are excluded from target density occupancy and reflected in out_of_domain_rate.",
            }
        )
    return reasons


def build_summary(result: DensityDiagnosisResult) -> list[str]:
    messages: list[str] = []
    if result.specificity_score > 0.75 and result.confidence > 0.7:
        messages.append("Specificity Score and confidence are both high. Target rows fall in low-density or unseen bins relative to the occupied peer bin count distribution.")
    elif result.specificity_score > 0.75 and result.confidence <= 0.4:
        messages.append("Target rows are rare against the current peer density map, but the peer map itself has limited support. Interpret the outlier signal cautiously.")
    elif result.specificity_score <= 0.5:
        messages.append("Target rows mostly fall in dense or ordinary occupied peer bins.")
    else:
        messages.append("The target shows moderate specificity. Additional peer observations can make the density baseline sharper.")

    if result.observation_support_S < 0.5:
        messages.append("Observation Support is low. Add saved records with row-level sanitized axis vectors.")
    if result.coverage_C < 0.3:
        messages.append("Coverage is low. The peer density map occupies only a small portion of the configured domain-resolution grid.")
    if result.equitability_E < 0.5:
        messages.append("Equitability is low. Peer observations are concentrated in a small number of occupied bins.")
    if result.unseen_bin_rate > 0:
        messages.append(f"{result.unseen_bin_rate:.1%} of valid target rows fall in bins unseen in the peer density map.")
    if result.extreme_specificity_rate > 0:
        messages.append(f"{result.extreme_specificity_rate:.1%} of valid target rows have bin-level specificity >= 0.95.")
    if result.masked_out_target_rows > 0:
        messages.append(f"{result.masked_out_target_rows} target rows were inside the axis Domain Range but outside the Feasible Domain Mask and were excluded from specificity scoring.")
    return messages


def make_cluster_record(
    goal: dict[str, Any],
    selected_goal: dict[str, Any],
    cluster_vector: np.ndarray,
    dataset_meta: dict[str, Any],
    source_batch_id: str | None = None,
    analysis_at_upload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_axes = canonical_axis_order(selected_goal["axes"])
    axis_names = [axis["name"] for axis in canonical_axes]
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
        "gridSignature": grid_signature_from_axes(canonical_axes),
        "peerGroupKey": key,
        "values": values,
        "valuesMean": dataset_meta.get("values_mean", values),
        "valuesVariance": dataset_meta.get("values_variance", [None for _ in values]),
        "valuesStd": dataset_meta.get("values_std", [None for _ in values]),
        "rowCount": int(dataset_meta["row_count"]),
        "clusterVectorRowCount": int(dataset_meta.get("cluster_vector_row_count") or dataset_meta["row_count"]),
        "clusterVectorBasis": str(dataset_meta.get("cluster_vector_basis") or "valid_multidimensional_numeric_rows"),
        "axisNumericCounts": dataset_meta.get("axis_numeric_counts", {}),
        "rowLevelVectors": dataset_meta.get("row_level_vectors", []),
        "rowLevelVectorAxisOrder": dataset_meta.get("row_level_vector_axis_order", axis_names),
        "rowLevelVectorCount": int(dataset_meta.get("row_level_vector_count") or 0),
        "rowLevelVectorBasis": str(dataset_meta.get("row_level_vector_basis") or "valid_multidimensional_numeric_rows"),
        "hasRowLevelVectors": bool(dataset_meta.get("row_level_vectors")),
        "binOccupancy": dataset_meta.get("bin_occupancy", {}),
        "binOccupancyHash": bin_occupancy_hash(dataset_meta.get("bin_occupancy", {})),
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
        "storagePolicy": "sanitized_row_level_axis_vectors",
        "analysisAtUpload": analysis,
        "peerGroupSizeAtUpload": analysis.get("peerGroupSize"),
        "engineAtUpload": analysis.get("engine"),
        "specificityScoreAtUpload": analysis.get("specificityScore"),
        "specificityMethodAtUpload": analysis.get("specificityMethod"),
        "meanBinSpecificityAtUpload": analysis.get("meanBinSpecificity"),
        "maxSpecificityAtUpload": analysis.get("maxSpecificity"),
        "extremeSpecificityRateAtUpload": analysis.get("extremeSpecificityRate"),
        "meanRarityAtUpload": analysis.get("meanRarity"),
        "maxRarityAtUpload": analysis.get("maxRarity"),
        "unseenBinRateAtUpload": analysis.get("unseenBinRate"),
        "rareBinRateAtUpload": analysis.get("rareBinRate"),
        "outOfDomainRateAtUpload": analysis.get("outOfDomainRate"),
        "confidenceAtUpload": analysis.get("confidence"),
        "observationSupportSAtUpload": analysis.get("observationSupportS"),
        "peerObservationCountAtUpload": analysis.get("peerObservationCount"),
        "validTargetRowsAtUpload": analysis.get("validTargetRows"),
        "maskedOutTargetRowsAtUpload": analysis.get("maskedOutTargetRows"),
        "coverageCAtUpload": analysis.get("coverageC"),
        "equitabilityEAtUpload": analysis.get("equitabilityE"),
        "totalBinsAtUpload": analysis.get("totalBins"),
        "validBinsAtUpload": analysis.get("validBins"),
        "maskedBinsAtUpload": analysis.get("maskedBins"),
        "occupiedBinsAtUpload": analysis.get("occupiedBins"),
    }
    record["fingerprint"] = cluster_fingerprint(record)
    return record


def experiment_config_from_goal(selected_goal: dict[str, Any]) -> ExperimentConfig:
    axes = canonical_axis_order(selected_goal["axes"])
    return ExperimentConfig(
        axis_names=[axis["name"] for axis in axes],
        domain_range=[(axis["domainMin"], axis["domainMax"]) for axis in axes],
        resolution=[axis["resolution"] for axis in axes],
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


def row_vectors_for_axis_order(
    row_vectors: list[Any],
    source_axis_order: list[str],
    target_axis_order: list[str],
) -> list[list[float]] | None:
    source_positions: dict[str, int] = {}
    for index, axis_name in enumerate(source_axis_order or []):
        axis_key = normalize_axis_name(axis_name)
        if axis_key and axis_key not in source_positions:
            source_positions[axis_key] = index
    target_positions: list[int] = []
    for axis_name in target_axis_order:
        axis_key = normalize_axis_name(axis_name)
        if axis_key not in source_positions:
            return None
        target_positions.append(source_positions[axis_key])

    normalized: list[list[float]] = []
    for row in row_vectors or []:
        if not isinstance(row, list):
            continue
        try:
            vector = [float(row[position]) for position in target_positions]
        except (TypeError, ValueError, IndexError):
            continue
        if all(np.isfinite(value) for value in vector):
            normalized.append(vector)
    return normalized


def recompute_bin_occupancy_from_row_vectors(
    row_vectors: list[Any],
    axis_order: list[str],
    selected_goal: dict[str, Any],
) -> dict[str, Any] | None:
    axes = canonical_axis_order(selected_goal["axes"])
    target_axis_order = [str(axis["name"]) for axis in axes]
    reordered_rows = row_vectors_for_axis_order(row_vectors, axis_order, target_axis_order)
    if reordered_rows is None:
        return None

    bin_occupancy: dict[str, int] = {}
    axis_bin_occupancy: dict[str, dict[str, int]] = {str(axis["name"]): {} for axis in axes}
    row_bin_tuples: list[list[int]] = []
    out_of_domain_rows = 0
    masked_out_rows = 0
    masked_out_bin_occupancy: dict[str, int] = {}
    feasible_cache: dict[str, bool] = {}

    for vector in reordered_rows:
        multidimensional_indices: list[int] = []
        row_out_of_domain = False
        for axis, value in zip(axes, vector):
            index, status = bin_index_for_value(
                value,
                float(axis["domainMin"]),
                float(axis["domainMax"]),
                float(axis["resolution"]),
            )
            if status == "valid" and index is not None:
                multidimensional_indices.append(index)
            else:
                row_out_of_domain = True
                break
        if row_out_of_domain:
            out_of_domain_rows += 1
            continue
        key = json.dumps(multidimensional_indices, separators=(",", ":"))
        if not is_bin_key_feasible_for_goal(key, selected_goal, feasible_cache):
            masked_out_rows += 1
            masked_out_bin_occupancy[key] = masked_out_bin_occupancy.get(key, 0) + 1
            continue
        for axis, index in zip(axes, multidimensional_indices):
            axis_counts = axis_bin_occupancy[str(axis["name"])]
            axis_counts[str(index)] = axis_counts.get(str(index), 0) + 1
        bin_occupancy[key] = bin_occupancy.get(key, 0) + 1
        row_bin_tuples.append(list(multidimensional_indices))

    return {
        "bin_occupancy": bin_occupancy,
        "axis_bin_occupancy": axis_bin_occupancy,
        "row_bin_tuples": row_bin_tuples,
        "bin_occupancy_meta": {
            "version": 1,
            "basis": "row_level_recomputed_from_sanitized_vectors",
            "validMultidimensionalRowCount": int(sum(bin_occupancy.values())),
            "feasibleValidRowCount": int(sum(bin_occupancy.values())),
            "invalidRowCount": 0,
            "outOfDomainRowCount": out_of_domain_rows,
            "maskedOutRowCount": masked_out_rows,
            "infeasibleRowCount": masked_out_rows,
            "maskedOutBinCount": len(masked_out_bin_occupancy),
            "totalRows": len(reordered_rows),
        },
        "masked_out_bin_occupancy": masked_out_bin_occupancy,
    }


def build_global_bin_counts(peer_clusters: list[dict[str, Any]], selected_goal: dict[str, Any] | list[str]) -> dict[str, Any]:
    if isinstance(selected_goal, dict):
        canonical_axes = canonical_axis_order(selected_goal["axes"])
        selected_axis_names = [str(axis["name"]) for axis in canonical_axes]
        current_grid_signature = grid_signature_from_axes(canonical_axes)
        mask_info = feasible_mask_info_for_goal(selected_goal)
    else:
        selected_axis_names = [str(name) for name in selected_goal]
        current_grid_signature = ""
        mask_info = {
            "totalBins": 0,
            "validBins": 0,
            "maskedBins": 0,
            "feasibleExpressions": [],
            "feasibleMaskEnabled": False,
            "validDomainRatio": 0.0,
        }
    selected_signature = axis_subset_key(selected_axis_names)
    bin_counts: dict[str, int] = {}
    axis_bin_counts: dict[str, dict[str, int]] = {}
    eligible_count = 0
    legacy_excluded = 0
    grid_signature_excluded = 0
    row_level_observation_count = 0
    infeasible_peer_rows = 0
    infeasible_peer_bin_keys: set[str] = set()

    for cluster in peer_clusters:
        recomputed: dict[str, Any] | None = None
        if current_grid_signature:
            row_vectors = cluster.get("rowLevelVectors") if isinstance(cluster.get("rowLevelVectors"), list) else []
            source_axis_order = [str(name) for name in (cluster.get("rowLevelVectorAxisOrder") or cluster.get("axisNames") or [])]
            if not row_vectors:
                legacy_excluded += 1
                continue
            recomputed = recompute_bin_occupancy_from_row_vectors(row_vectors, source_axis_order, selected_goal)
            if recomputed is None:
                grid_signature_excluded += 1
                continue
            cluster_bin_counts = recomputed["bin_occupancy"]
            cluster_axis_counts = recomputed["axis_bin_occupancy"]
            meta = recomputed["bin_occupancy_meta"]
            infeasible_peer_rows += int(meta.get("maskedOutRowCount") or meta.get("infeasibleRowCount") or 0)
            infeasible_peer_bin_keys.update(str(key) for key in (recomputed.get("masked_out_bin_occupancy") or {}).keys())
        else:
            cluster_bin_counts = cluster.get("binOccupancy") if isinstance(cluster.get("binOccupancy"), dict) else {}
            cluster_grid_signature = str(cluster.get("gridSignature") or "").strip()
            if not cluster_bin_counts or not cluster_grid_signature:
                legacy_excluded += 1
                continue
            stored_signature = str(cluster.get("storedAxisSignature") or cluster.get("axisSignature") or "")
            if stored_signature != selected_signature:
                grid_signature_excluded += 1
                continue
            cluster_axis_counts = cluster.get("axisBinOccupancy") or {}
            meta = cluster.get("binOccupancyMeta") if isinstance(cluster.get("binOccupancyMeta"), dict) else {}
            infeasible_peer_rows += int(meta.get("maskedOutRowCount") or meta.get("infeasibleRowCount") or 0)
        eligible_count += 1
        merge_count_maps(bin_counts, cluster_bin_counts)
        try:
            row_level_observation_count += int(meta.get("validMultidimensionalRowCount") or count_map_total(cluster_bin_counts))
        except (TypeError, ValueError):
            row_level_observation_count += count_map_total(cluster_bin_counts)
        for axis_name, counts in (cluster_axis_counts or {}).items():
            axis_counts = axis_bin_counts.setdefault(str(axis_name), {})
            merge_count_maps(axis_counts, counts)

    return {
        "binCounts": bin_counts,
        "axisBinCounts": axis_bin_counts,
        "coverageBasis": "row_level_bin_occupancy",
        "coverageEligibleClusterCount": eligible_count,
        "coverageLegacyExcludedClusterCount": legacy_excluded,
        "coverageGridSignatureExcludedClusterCount": grid_signature_excluded,
        "coverageAxisSignatureExcludedClusterCount": grid_signature_excluded,
        "rowLevelObservationCount": row_level_observation_count,
        "occupiedBins": len(bin_counts),
        "totalBins": int(mask_info.get("totalBins") or 0),
        "validBins": int(mask_info.get("validBins") or mask_info.get("totalBins") or 0),
        "maskedBins": int(mask_info.get("maskedBins") or 0),
        "validDomainRatio": float(mask_info.get("validDomainRatio") or 0.0),
        "feasibleMaskEnabled": bool(mask_info.get("feasibleMaskEnabled")),
        "feasibleExpressions": list(mask_info.get("feasibleExpressions") or []),
        "infeasiblePeerRows": int(infeasible_peer_rows),
        "infeasiblePeerBins": int(len(infeasible_peer_bin_keys)),
        "gridSignature": current_grid_signature,
    }


def out_of_domain_warnings(
    selected_goal: dict[str, Any],
    target_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    meta = target_meta or {}
    out_of_domain_count = int(meta.get("outOfDomainRowCount") or 0)
    masked_out_count = int(meta.get("maskedOutRowCount") or meta.get("infeasibleRowCount") or 0)
    total_rows = int(meta.get("totalRows") or 0)
    if out_of_domain_count > 0:
        warnings.append(
            {
                "role": "target",
                "outOfDomainRowCount": out_of_domain_count,
                "totalRows": total_rows,
                "axes": [axis["name"] for axis in canonical_axis_order(selected_goal["axes"])],
                "message": "Target rows outside the configured domain range were excluded from density occupancy.",
            }
        )
    if masked_out_count > 0:
        warnings.append(
            {
                "role": "target",
                "maskedOutRowCount": masked_out_count,
                "infeasibleRowCount": masked_out_count,
                "totalRows": total_rows,
                "axes": [axis["name"] for axis in canonical_axis_order(selected_goal["axes"])],
                "message": "Some target rows are inside the axis Domain Range but outside the Feasible Domain Mask. These rows were excluded from specificity scoring and reported separately.",
            }
        )
    return warnings


def analysis_snapshot(
    result: DensityDiagnosisResult | None,
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
            "specificityMethod": payload.get("specificity_method"),
            "specificityInterpretation": payload.get("specificity_interpretation"),
            "specificityScore": payload["specificity_score"],
            "meanBinSpecificity": payload.get("mean_bin_specificity"),
            "maxSpecificity": payload.get("max_specificity"),
            "extremeSpecificityRate": payload.get("extreme_specificity_rate"),
            "meanRarity": payload["mean_rarity"],
            "maxRarity": payload["max_rarity"],
            "unseenBinRate": payload["unseen_bin_rate"],
            "rareBinRate": payload["rare_bin_rate"],
            "outOfDomainRate": payload["out_of_domain_rate"],
            "confidence": payload["confidence"],
            "observationSupportS": payload["observation_support_S"],
            "coverageC": payload["coverage_C"],
            "equitabilityE": payload["equitability_E"],
            "peerObservationCount": payload["peer_observation_count"],
            "validTargetRows": payload["valid_target_rows"],
            "invalidTargetRows": payload["invalid_target_rows"],
            "outOfDomainRows": payload["out_of_domain_rows"],
            "maskedOutTargetRows": payload.get("masked_out_target_rows"),
            "infeasibleTargetRows": payload.get("infeasible_target_rows"),
            "totalBins": payload["total_bins"],
            "validBins": payload.get("valid_bins"),
            "maskedBins": payload.get("masked_bins"),
            "occupiedBins": payload["occupied_bins"],
            "feasibleMaskEnabled": payload.get("feasible_mask_enabled"),
            "feasibleExpressions": coverage_info.get("feasibleExpressions", []) if coverage_info else [],
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
    base_result: DensityDiagnosisResult,
) -> list[dict[str, Any]]:
    return []


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


def run_density_analysis(
    goal: dict[str, Any],
    selected_goal: dict[str, Any],
    target_bin_counts: dict[str, int],
    target_meta: dict[str, Any],
    target_vector: np.ndarray | None = None,
    exclude_cluster_id: str | None = None,
) -> dict[str, Any]:
    axis_names = [axis["name"] for axis in canonical_axis_order(selected_goal["axes"])]
    config = experiment_config_from_goal(selected_goal)
    peer_clusters = analysis_peer_clusters(goal, axis_names, exclude_cluster_id)
    peer_rows = [{"id": str(cluster.get("id", "")), "source": "stored", "values": [float(value) for value in cluster["values"]]} for cluster in peer_clusters]
    peer_group = peer_matrix(peer_rows)
    if peer_group.size == 0:
        peer_group = np.empty((0, len(axis_names)), dtype=float)
    coverage_info = build_global_bin_counts(peer_clusters, selected_goal)
    warnings = out_of_domain_warnings(selected_goal, target_meta)
    analyzer = DensityGridAnalyzer(config)
    analyzer.set_peer_bin_counts(coverage_info["binCounts"])
    analyzer.set_feasible_domain(
        valid_bins=int(coverage_info.get("validBins") or coverage_info.get("totalBins") or 0),
        masked_bins=int(coverage_info.get("maskedBins") or 0),
        feasible_mask_enabled=bool(coverage_info.get("feasibleMaskEnabled")),
    )
    result = analyzer.diagnose(target_bin_counts, target_meta)
    result_payload = result.to_payload(config.axis_names)
    result_payload["totalBins"] = result_payload.get("total_bins")
    result_payload["validTargetRows"] = result_payload.get("valid_target_rows")
    result_payload["outOfDomainRows"] = result_payload.get("out_of_domain_rows")
    result_payload["maskedOutTargetRows"] = result_payload.get("masked_out_target_rows")
    result_payload["infeasibleTargetRows"] = result_payload.get("infeasible_target_rows")
    result_payload["occupiedBins"] = result_payload.get("occupied_bins")
    result_payload["outOfDomainWarnings"] = warnings
    result_payload["outOfDomainWarningCount"] = len(warnings)
    result_payload["coverageBasis"] = coverage_info["coverageBasis"]
    result_payload["coverageEligibleClusterCount"] = coverage_info["coverageEligibleClusterCount"]
    result_payload["coverageLegacyExcludedClusterCount"] = coverage_info["coverageLegacyExcludedClusterCount"]
    result_payload["coverageGridSignatureExcludedClusterCount"] = coverage_info["coverageGridSignatureExcludedClusterCount"]
    result_payload["coverageAxisSignatureExcludedClusterCount"] = coverage_info["coverageGridSignatureExcludedClusterCount"]
    result_payload["rowLevelObservationCount"] = coverage_info["rowLevelObservationCount"]
    result_payload["gridSignature"] = coverage_info["gridSignature"]
    result_payload["validBins"] = coverage_info.get("validBins", result.valid_bins)
    result_payload["maskedBins"] = coverage_info.get("maskedBins", result.masked_bins)
    result_payload["validDomainRatio"] = coverage_info.get("validDomainRatio", 1.0)
    result_payload["valid_domain_ratio"] = coverage_info.get("validDomainRatio", result_payload.get("valid_domain_ratio", 1.0))
    result_payload["feasibleMaskEnabled"] = coverage_info.get("feasibleMaskEnabled", False)
    result_payload["feasibleExpressions"] = coverage_info.get("feasibleExpressions", [])
    result_payload["infeasiblePeerRows"] = coverage_info.get("infeasiblePeerRows", 0)
    result_payload["infeasiblePeerBins"] = coverage_info.get("infeasiblePeerBins", 0)
    return {
        "config": config,
        "peerRows": peer_rows,
        "peerClusters": peer_clusters,
        "peerGroup": peer_group,
        "coverageInfo": {
            **coverage_info,
            "totalBins": result.total_bins,
            "validBins": result.valid_bins,
            "maskedBins": result.masked_bins,
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
    axis_names = [axis["name"] for axis in canonical_axis_order(selected_goal["axes"])]
    key = peer_group_key(str(goal["id"]), axis_names)
    cluster_vector, dataset_meta = build_dataset_summary(rows, axis_mapping, selected_goal)
    saved_cluster = None
    saved_cluster_is_new = False

    try:
        analysis = run_density_analysis(
            goal,
            selected_goal,
            dataset_meta.get("bin_occupancy", {}),
            dataset_meta.get("bin_occupancy_meta", {}),
            cluster_vector,
        )
    except ValueError as exc:
        peer_clusters = analysis_peer_clusters(goal, axis_names)
        peer_rows = [{"id": str(cluster.get("id", "")), "source": "stored", "values": [float(value) for value in cluster["values"]]} for cluster in peer_clusters]
        peer_group = peer_matrix(peer_rows)
        if peer_group.size == 0:
            peer_group = np.empty((0, len(axis_names)), dtype=float)
        coverage_info = build_global_bin_counts(peer_clusters, selected_goal)
        coverage_info["totalBins"] = BinGridTracker(
            [(float(axis["domainMin"]), float(axis["domainMax"])) for axis in canonical_axis_order(selected_goal["axes"])],
            [float(axis["resolution"]) for axis in canonical_axis_order(selected_goal["axes"])],
        ).total_bins
        warnings = out_of_domain_warnings(selected_goal, dataset_meta.get("bin_occupancy_meta", {}))
        pending_cluster = make_cluster_record(
            goal,
            selected_goal,
            cluster_vector,
            dataset_meta,
            analysis_at_upload=analysis_snapshot(None, axis_names, len(peer_group), warnings, str(exc), coverage_info=coverage_info),
        )
        if should_save_data_clusters():
            saved_cluster, saved_cluster_is_new = save_data_cluster(pending_cluster)
        stored_count = int(coverage_info["coverageEligibleClusterCount"])
        saved_text = "saved" if saved_cluster_is_new else "already stored"
        raise ValueError(
            "Density analysis uses saved records with row-level sanitized axis vectors re-binned into the current grid. "
            f"Eligible density records={stored_count}, peer row-level observations={coverage_info['rowLevelObservationCount']}. "
            f"Excluded legacy records={coverage_info['coverageLegacyExcludedClusterCount']}, "
            f"axis mismatches={coverage_info['coverageGridSignatureExcludedClusterCount']}. "
            f"This CSV was sanitized into row-level axis vectors and bin occupancy and {saved_text}, but analysis is limited until eligible peer density exists. "
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
    summary.append("Density scoring re-bins saved row-level sanitized axis vectors into the current grid.")
    if analysis["warnings"]:
        summary.append(f"{len(analysis['warnings'])} out-of-domain value(s) were clipped for bin calculations and surfaced as warnings.")
    if saved_cluster:
        status = "saved as a new record" if saved_cluster_is_new else "detected as an existing duplicate and not saved again"
        summary.append(f"Raw uploaded rows, filename, and unmapped columns were not stored; only the sanitized density summary was {status}.")

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
            "axes": canonical_axis_order(selected_goal["axes"]),
            "available_axes": goal["axes"],
            "config": asdict(config),
            "cluster_definition": dataset_meta["cluster_definition"],
            "analysis_timestamp": analysis["snapshot"]["analysisTimestamp"],
            "coverageBasis": analysis["coverageInfo"]["coverageBasis"],
            "coverageEligibleClusterCount": analysis["coverageInfo"]["coverageEligibleClusterCount"],
            "coverageLegacyExcludedClusterCount": analysis["coverageInfo"]["coverageLegacyExcludedClusterCount"],
            "coverageGridSignatureExcludedClusterCount": analysis["coverageInfo"]["coverageGridSignatureExcludedClusterCount"],
            "coverageAxisSignatureExcludedClusterCount": analysis["coverageInfo"]["coverageGridSignatureExcludedClusterCount"],
            "rowLevelObservationCount": analysis["coverageInfo"]["rowLevelObservationCount"],
            "occupiedBins": analysis["coverageInfo"]["occupiedBins"],
            "totalBins": analysis["coverageInfo"]["totalBins"],
            "storage_policy": "Raw upload rows, filenames, and unmapped columns are not stored.",
        },
        "result": result_payload,
        "summary": summary,
        "confidenceReasons": confidence_reasons(result, analysis["warnings"]),
        "visualizations": build_report_visualizations(
            selected_goal,
            peer_group,
            cluster_vector,
            result,
            analysis["coverageInfo"],
            dataset_meta.get("bin_occupancy", {}),
            dataset_meta.get("bin_occupancy_meta", {}),
            dataset_meta.get("row_bin_tuples", []),
            dataset_meta.get("row_level_vectors", []),
            dataset_meta.get("row_level_vector_axis_order", []),
        ),
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
            "rowLevelVectorCount": saved_cluster.get("rowLevelVectorCount"),
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
    axis_names = [axis["name"] for axis in canonical_axis_order(selected_goal["axes"])]
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
            cluster_vector, dataset_meta = build_dataset_summary(rows, axis_mapping, selected_goal)
            item_payload.update(
                {
                    "rowCount": dataset_meta["row_count"],
                    "clusterVectorRowCount": dataset_meta.get("cluster_vector_row_count"),
                    "rowLevelVectorCount": dataset_meta.get("row_level_vector_count"),
                    "rowLevelValidCount": dataset_meta.get("bin_occupancy_meta", {}).get("validMultidimensionalRowCount"),
                    "axisMappingStatus": "mapped",
                    "clusterVector": [round(float(value), 6) for value in cluster_vector],
                }
            )
            try:
                analysis = run_density_analysis(
                    goal,
                    selected_goal,
                    dataset_meta.get("bin_occupancy", {}),
                    dataset_meta.get("bin_occupancy_meta", {}),
                    cluster_vector,
                )
                result_payload = analysis["resultPayload"]
                snapshot = analysis["snapshot"]
                item_payload.update(
                    {
                        "analysisSuccess": True,
                        "analysisSummary": {
                            "specificity_score": result_payload["specificity_score"],
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
                coverage_info = build_global_bin_counts(peer_clusters, selected_goal)
                coverage_info["totalBins"] = BinGridTracker(
                    [(float(axis["domainMin"]), float(axis["domainMax"])) for axis in canonical_axis_order(selected_goal["axes"])],
                    [float(axis["resolution"]) for axis in canonical_axis_order(selected_goal["axes"])],
                ).total_bins
                warnings = out_of_domain_warnings(selected_goal, dataset_meta.get("bin_occupancy_meta", {}))
                snapshot = analysis_snapshot(None, axis_names, len(peer_group), warnings, str(exc), coverage_info=coverage_info)
                item_payload["analysisError"] = str(exc)
                item_payload["analysisSummary"] = {
                    "specificity_score": None,
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
    compatible_density_count = 0
    if goal_id and selected_axis_names:
        try:
            selected_goal = goal_subset(find_goal(goal_id), selected_axis_names)
            compatible_density_count = int(
                build_global_bin_counts(load_peer_clusters(goal_id, selected_axis_names), selected_goal)["coverageEligibleClusterCount"]
            )
        except ValueError:
            compatible_density_count = 0
    return {
        "saved": saved,
        "totalStoredClusters": diagnostics.get("totalClusters", len(load_cluster_store())),
        "sameGoalClusterCount": diagnostics.get("sameGoalCount", 0),
        "compatiblePeerCountForSelectedAxes": compatible_density_count,
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


def record_density_counts_for_goal(cluster: dict[str, Any], selected_goal: dict[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
    row_vectors = cluster.get("rowLevelVectors") if isinstance(cluster.get("rowLevelVectors"), list) else []
    source_axis_order = [str(name) for name in (cluster.get("rowLevelVectorAxisOrder") or cluster.get("axisNames") or [])]
    if row_vectors:
        recomputed = recompute_bin_occupancy_from_row_vectors(row_vectors, source_axis_order, selected_goal)
        if recomputed is not None:
            return recomputed["bin_occupancy"], recomputed["bin_occupancy_meta"]
    return (
        cluster.get("binOccupancy") if isinstance(cluster.get("binOccupancy"), dict) else {},
        cluster.get("binOccupancyMeta") if isinstance(cluster.get("binOccupancyMeta"), dict) else {},
    )


def reevaluate_cluster(cluster_id: str) -> dict[str, Any]:
    cluster = cluster_by_id(cluster_id)
    goal = find_goal(str(cluster["goalId"]))
    selected_goal = goal_subset(goal, [str(name) for name in cluster["axisNames"]])
    target = np.asarray(cluster["values"], dtype=float)
    target_counts, target_meta = record_density_counts_for_goal(cluster, selected_goal)
    uploaded = normalize_analysis_snapshot(cluster.get("analysisAtUpload"))
    try:
        analysis = run_density_analysis(
            goal,
            selected_goal,
            target_counts,
            target_meta,
            target,
            exclude_cluster_id=str(cluster["id"]),
        )
        current = analysis["resultPayload"]
        current_peer_group_size = len(analysis["peerGroup"])
        confidence_delta = None if uploaded.get("confidence") is None else round(float(current["confidence"]) - float(uploaded["confidence"]), 6)
        specificity_delta = None if uploaded.get("specificityScore") is None else round(float(current["specificity_score"]) - float(uploaded["specificityScore"]), 6)
        interpretation = reevaluation_interpretation(uploaded, current)
        return {
            "clusterId": cluster["id"],
            "uploadedAt": cluster.get("uploadedAt"),
            "uploaded": uploaded,
            "current": current,
            "currentPeerGroupSize": current_peer_group_size,
            "confidenceDelta": confidence_delta,
            "specificityDelta": specificity_delta,
            "interpretation": interpretation,
        }
    except ValueError as exc:
        return {
            "clusterId": cluster["id"],
            "uploadedAt": cluster.get("uploadedAt"),
            "uploaded": uploaded,
            "current": None,
            "currentPeerGroupSize": len(analysis_peer_rows(goal, [axis["name"] for axis in canonical_axis_order(selected_goal["axes"])], str(cluster["id"]))),
            "error": str(exc),
            "interpretation": ["Current reevaluation is limited because no eligible peer density remains after excluding this cluster."],
        }


def reevaluation_interpretation(uploaded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    uploaded_h = uploaded.get("specificityScore")
    uploaded_c = uploaded.get("confidence")
    current_h = current.get("specificity_score")
    if uploaded_h is not None and current_h is not None:
        if float(uploaded_h) >= 0.75 and float(current_h) <= 0.55:
            messages.append("The record looked highly specific at upload time, but is closer to the current density baseline now.")
        elif float(current_h) >= 0.75:
            messages.append("The record remains highly specific against the current density baseline.")
        else:
            messages.append("The record is only mildly specific against the current density baseline.")
    if uploaded_c is not None and float(uploaded_c) < 0.4:
        messages.append("Upload-time confidence was low, so the earlier judgment was support-limited.")
    return messages or ["Upload-time and current density results are broadly similar."]


def impact_result_payload(goal: dict[str, Any], selected_goal: dict[str, Any], cluster: dict[str, Any], exclude_cluster_id: str | None) -> dict[str, Any]:
    try:
        target = np.asarray(cluster.get("values", []), dtype=float)
        target_counts, target_meta = record_density_counts_for_goal(cluster, selected_goal)
        analysis = run_density_analysis(
            goal,
            selected_goal,
            target_counts,
            target_meta,
            target,
            exclude_cluster_id=exclude_cluster_id,
        )
        return {"ok": True, "result": analysis["resultPayload"], "peerGroupSize": len(analysis["peerGroup"])}
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "peerGroupSize": len(analysis_peer_rows(goal, [axis["name"] for axis in canonical_axis_order(selected_goal["axes"])], exclude_cluster_id)),
        }


def delete_impact_request(payload: dict[str, Any]) -> dict[str, Any]:
    cluster = cluster_by_id(str(payload.get("id", "")))
    goal = find_goal(str(cluster["goalId"]))
    selected_goal = goal_subset(goal, [str(name) for name in cluster["axisNames"]])
    target = np.asarray(cluster["values"], dtype=float)
    all_payload = impact_result_payload(goal, selected_goal, cluster, None)
    without_payload = impact_result_payload(goal, selected_goal, cluster, str(cluster["id"]))
    deltas: dict[str, Any] = {}
    if all_payload["ok"] and without_payload["ok"]:
        all_result = all_payload["result"]
        without_result = without_payload["result"]
        deltas = {
            "peerGroupN": [all_payload["peerGroupSize"], without_payload["peerGroupSize"]],
            "deltaConfidence": round(float(without_result["confidence"] - all_result["confidence"]), 6),
            "deltaCoverage": round(float(without_result["coverage_C"] - all_result["coverage_C"]), 6),
            "deltaEquitability": round(float(without_result["equitability_E"] - all_result["equitability_E"]), 6),
            "deltaSpecificity": round(float(without_result["specificity_score"] - all_result["specificity_score"]), 6),
            "deltaMeanRarity": round(float(without_result["mean_rarity"] - all_result["mean_rarity"]), 6),
        }
    without_peer_clusters = analysis_peer_clusters(goal, [axis["name"] for axis in canonical_axis_order(selected_goal["axes"])], str(cluster["id"]))
    without_density = build_global_bin_counts(without_peer_clusters, selected_goal)
    target_counts, _target_meta = record_density_counts_for_goal(cluster, selected_goal)
    bin_uniqueness = bool(target_counts) and all(int(without_density["binCounts"].get(str(key), 0)) == 0 for key in target_counts)
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
            "specificity_method",
            "specificity_score",
            "mean_bin_specificity",
            "max_specificity",
            "extreme_specificity_rate",
            "mean_rarity",
            "max_rarity",
            "unseen_bin_rate",
            "rare_bin_rate",
            "out_of_domain_rate",
            "masked_out_target_rows",
            "infeasible_target_rows",
            "feasible_mask_enabled",
            "feasible_expressions",
            "valid_bins",
            "masked_bins",
            "valid_domain_ratio",
            "infeasible_peer_rows",
            "infeasible_peer_bins",
            "confidence",
            "observation_support_S",
            "coverage_C",
            "equitability_E",
            "peer_observation_count",
            "valid_target_rows",
            "invalid_target_rows",
            "out_of_domain_rows",
            "total_bins",
            "occupied_bins",
            "analysis_timestamp",
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
                "specificity_method": result.get("specificity_method"),
                "specificity_score": result.get("specificity_score"),
                "mean_bin_specificity": result.get("mean_bin_specificity"),
                "max_specificity": result.get("max_specificity"),
                "extreme_specificity_rate": result.get("extreme_specificity_rate"),
                "mean_rarity": result.get("mean_rarity"),
                "max_rarity": result.get("max_rarity"),
                "unseen_bin_rate": result.get("unseen_bin_rate"),
                "rare_bin_rate": result.get("rare_bin_rate"),
                "out_of_domain_rate": result.get("out_of_domain_rate"),
                "masked_out_target_rows": result.get("masked_out_target_rows"),
                "infeasible_target_rows": result.get("infeasible_target_rows"),
                "feasible_mask_enabled": result.get("feasible_mask_enabled"),
                "feasible_expressions": json.dumps(result.get("feasibleExpressions", result.get("feasible_expressions", [])), ensure_ascii=False),
                "valid_bins": result.get("valid_bins"),
                "masked_bins": result.get("masked_bins"),
                "valid_domain_ratio": result.get("validDomainRatio", result.get("valid_domain_ratio")),
                "infeasible_peer_rows": result.get("infeasiblePeerRows", result.get("infeasible_peer_rows")),
                "infeasible_peer_bins": result.get("infeasiblePeerBins", result.get("infeasible_peer_bins")),
                "confidence": result.get("confidence"),
                "observation_support_S": result.get("observation_support_S"),
                "coverage_C": result.get("coverage_C"),
                "equitability_E": result.get("equitability_E"),
                "peer_observation_count": result.get("peer_observation_count"),
                "valid_target_rows": result.get("valid_target_rows"),
                "invalid_target_rows": result.get("invalid_target_rows"),
                "out_of_domain_rows": result.get("out_of_domain_rows"),
                "total_bins": result.get("total_bins"),
                "occupied_bins": result.get("occupied_bins"),
                "analysis_timestamp": meta.get("analysis_timestamp"),
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
        full_axis_names = [axis["name"] for axis in canonical_axis_order(goal["axes"])]
        selected_goal = goal_subset(goal, full_axis_names)
        peer_counts[goal["id"]] = int(
            build_global_bin_counts(load_peer_clusters(str(goal["id"]), full_axis_names), selected_goal)["coverageEligibleClusterCount"]
        )
    return peer_counts


def bootstrap_peer_subset_counts() -> dict[str, dict[str, int]]:
    counts = {}
    for goal in normalize_goals_for_display(load_goal_store()):
        axis_names = [axis["name"] for axis in canonical_axis_order(goal["axes"])]
        goal_counts: dict[str, int] = {}
        for size in range(1, len(axis_names) + 1):
            for subset in itertools.combinations(axis_names, size):
                names = list(subset)
                selected_goal = goal_subset(goal, names)
                peer_clusters = load_peer_clusters(str(goal["id"]), names)
                try:
                    goal_counts[axis_subset_key(names)] = int(build_global_bin_counts(peer_clusters, selected_goal)["coverageEligibleClusterCount"])
                except ValueError:
                    goal_counts[axis_subset_key(names)] = 0
        counts[goal["id"]] = goal_counts
    return counts


def goal_bin_preview(goal: dict[str, Any]) -> dict[str, Any]:
    canonical_axes = canonical_axis_order(goal["axes"])
    axis_names = [axis["name"] for axis in canonical_axes]
    stored_clusters = load_peer_clusters(str(goal["id"]), axis_names)
    coverage_info = build_global_bin_counts(stored_clusters, goal)
    axis_previews = []
    total_bins = 1
    for axis_index, axis in enumerate(canonical_axes):
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
    valid_bins = int(coverage_info.get("validBins") or total_bins)
    masked_bins = int(coverage_info.get("maskedBins") or 0)
    warnings = []
    if total_bins > 100000:
        warnings.append("Resolution이 현재 데이터 수에 비해 과도하게 세밀할 수 있음.")
    if coverage_info["coverageLegacyExcludedClusterCount"]:
        warnings.append("일부 기존 record는 row-level vectors가 없어 coverage 계산에서 제외됩니다. 정확한 coverage를 원하면 CSV를 다시 업로드하세요.")
    if coverage_info["coverageGridSignatureExcludedClusterCount"]:
        warnings.append("Axis 구성이 맞지 않는 record는 density 계산에서 제외됩니다.")
    if coverage_info.get("feasibleMaskEnabled"):
        warnings.append("Feasible Domain Mask가 활성화되어 coverage는 occupied bins / valid bins 기준입니다.")
    return {
        "basis": coverage_info["coverageBasis"],
        "axisBins": axis_previews,
        "totalBins": total_bins,
        "validBins": valid_bins,
        "maskedBins": masked_bins,
        "validDomainRatio": coverage_info.get("validDomainRatio", valid_bins / total_bins if total_bins else 0.0),
        "feasibleMaskEnabled": coverage_info.get("feasibleMaskEnabled", False),
        "feasibleExpressions": coverage_info.get("feasibleExpressions", []),
        "occupiedBins": occupied_bins,
        "estimatedCoverage": occupied_bins / valid_bins if valid_bins else 0.0,
        "coverageEligibleClusterCount": coverage_info["coverageEligibleClusterCount"],
        "coverageLegacyExcludedClusterCount": coverage_info["coverageLegacyExcludedClusterCount"],
        "coverageGridSignatureExcludedClusterCount": coverage_info["coverageGridSignatureExcludedClusterCount"],
        "coverageAxisSignatureExcludedClusterCount": coverage_info["coverageGridSignatureExcludedClusterCount"],
        "rowLevelObservationCount": coverage_info["rowLevelObservationCount"],
        "warning": " ".join(warnings),
        "warnings": warnings,
    }


def grid_preview_goal(goal: dict[str, Any], selected_axis_names: list[str], preview_axes: list[dict[str, Any]]) -> dict[str, Any]:
    selected_goal = goal_subset(goal, selected_axis_names)
    preview_by_key = {normalize_axis_name(axis.get("name")): axis for axis in preview_axes if normalize_axis_name(axis.get("name"))}
    axes: list[dict[str, Any]] = []
    for axis in canonical_axis_order(selected_goal["axes"]):
        preview = preview_by_key.get(normalize_axis_name(axis["name"]), {})
        axes.append(
            {
                "name": axis["name"],
                "unit": axis.get("unit", ""),
                "domainMin": float(preview.get("domainMin", axis["domainMin"])),
                "domainMax": float(preview.get("domainMax", axis["domainMax"])),
                "resolution": float(preview.get("resolution", axis["resolution"])),
            }
        )
    preview_goal = {
        "id": selected_goal["id"],
        "name": selected_goal["name"],
        "K_m": float(selected_goal.get("K_m", K_M)),
        "axes": axes,
        "feasibleDomainExpressions": feasible_expressions_for_goal(selected_goal),
    }
    experiment_config_from_goal(preview_goal)
    return preview_goal


def grid_preview_request(payload: dict[str, Any]) -> dict[str, Any]:
    goal = find_goal(str(payload.get("goalId", "")))
    selected_axis_names = payload.get("selectedAxes") or [axis["name"] for axis in goal["axes"]]
    if not isinstance(selected_axis_names, list):
        raise ValueError("selectedAxes must be a list.")
    preview_axes = payload.get("previewAxes")
    if not isinstance(preview_axes, list):
        raise ValueError("previewAxes must be a list.")
    preview_goal = grid_preview_goal(goal, [str(name) for name in selected_axis_names], preview_axes)
    axis_names = [axis["name"] for axis in canonical_axis_order(preview_goal["axes"])]
    peer_clusters = analysis_peer_clusters(goal, axis_names)
    coverage_info = build_global_bin_counts(peer_clusters, preview_goal)
    coverage_info["totalBins"] = density_preview_metrics_from_coverage(preview_goal, coverage_info)["totalBins"]

    target_axis_order = [str(name) for name in (payload.get("targetRowVectorAxisOrder") or axis_names)]
    target_vectors = payload.get("targetRowVectors") if isinstance(payload.get("targetRowVectors"), list) else []
    target_recomputed = recompute_bin_occupancy_from_row_vectors(target_vectors, target_axis_order, preview_goal) or {
        "bin_occupancy": {},
        "axis_bin_occupancy": {},
        "row_bin_tuples": [],
        "bin_occupancy_meta": {"validMultidimensionalRowCount": 0, "invalidRowCount": 0, "outOfDomainRowCount": 0, "totalRows": 0},
    }
    metrics = density_preview_metrics_from_coverage(preview_goal, coverage_info)
    projection_explorer = build_projection_explorer(preview_goal, coverage_info, target_recomputed["row_bin_tuples"])
    projection_explorer["gridPreviewMetrics"] = metrics

    result_payload: dict[str, Any] | None = None
    if metrics["peerValidRows"] > 0 and target_recomputed["bin_occupancy_meta"]["validMultidimensionalRowCount"] > 0:
        analyzer = DensityGridAnalyzer(experiment_config_from_goal(preview_goal))
        analyzer.set_peer_bin_counts(coverage_info["binCounts"])
        analyzer.set_feasible_domain(
            valid_bins=metrics["validBins"],
            masked_bins=metrics["maskedBins"],
            feasible_mask_enabled=metrics["feasibleMaskEnabled"],
        )
        result_payload = analyzer.diagnose(target_recomputed["bin_occupancy"], target_recomputed["bin_occupancy_meta"]).to_payload(axis_names)

    return {
        "previewGoal": preview_goal,
        "metrics": metrics,
        "result": result_payload,
        "projectionExplorer": projection_explorer,
        "targetBinOccupancy": target_recomputed["bin_occupancy"],
        "targetBinOccupancyMeta": target_recomputed["bin_occupancy_meta"],
        "coverageInfo": {key: value for key, value in coverage_info.items() if key not in {"binCounts", "axisBinCounts"}},
    }


def rebuild_record_for_goal_grid(record: dict[str, Any], selected_goal: dict[str, Any]) -> tuple[dict[str, Any], bool, str]:
    axes = canonical_axis_order(selected_goal["axes"])
    axis_names = [str(axis["name"]) for axis in axes]
    row_vectors = record.get("rowLevelVectors") if isinstance(record.get("rowLevelVectors"), list) else []
    source_axis_order = [str(name) for name in (record.get("rowLevelVectorAxisOrder") or record.get("axisNames") or [])]
    reordered_rows = row_vectors_for_axis_order(row_vectors, source_axis_order, axis_names)
    if not row_vectors or reordered_rows is None:
        return dict(record), False, "legacy_missing_row_level_vectors"

    recomputed = recompute_bin_occupancy_from_row_vectors(reordered_rows, axis_names, selected_goal)
    if recomputed is None:
        return dict(record), False, "axis_mismatch"

    array = np.asarray(reordered_rows, dtype=float)
    if array.size:
        means = array.mean(axis=0)
        variance = array.var(axis=0, ddof=1) if len(array) > 1 else np.zeros(len(axis_names), dtype=float)
        std = np.sqrt(variance)
    else:
        means = np.asarray(record.get("values", [0.0 for _ in axis_names]), dtype=float)
        variance = np.zeros(len(axis_names), dtype=float)
        std = np.zeros(len(axis_names), dtype=float)

    updated = dict(record)
    updated["axisNames"] = axis_names
    updated["axisSignature"] = axis_subset_key(axis_names)
    updated["gridSignature"] = grid_signature_from_axes(axes)
    updated["peerGroupKey"] = peer_group_key(str(record["goalId"]), axis_names)
    updated["values"] = [round(float(value), 12) for value in means]
    updated["valuesMean"] = [round(float(value), 12) for value in means]
    updated["valuesVariance"] = [round(float(value), 12) for value in variance]
    updated["valuesStd"] = [round(float(value), 12) for value in std]
    updated["rowLevelVectors"] = [[round(float(value), 12) for value in row] for row in reordered_rows]
    updated["rowLevelVectorAxisOrder"] = axis_names
    updated["rowLevelVectorCount"] = len(reordered_rows)
    updated["rowLevelVectorBasis"] = "valid_multidimensional_numeric_rows"
    updated["hasRowLevelVectors"] = bool(reordered_rows)
    updated["binOccupancy"] = recomputed["bin_occupancy"]
    updated["axisBinOccupancy"] = recomputed["axis_bin_occupancy"]
    updated["binOccupancyMeta"] = recomputed["bin_occupancy_meta"]
    updated["binOccupancyHash"] = bin_occupancy_hash(recomputed["bin_occupancy"])
    updated["storagePolicy"] = "sanitized_row_level_axis_vectors"
    updated["fingerprint"] = cluster_fingerprint(updated)
    return updated, True, "updated"


def apply_goal_grid_defaults_request(payload: dict[str, Any]) -> dict[str, Any]:
    goal_id = str(payload.get("goalId", "")).strip()
    goals = normalize_goals_for_display(load_goal_store())
    goal = next((item for item in goals if item["id"] == goal_id), None)
    if goal is None:
        raise ValueError("Selected Experiment Goal does not exist.")
    selected_axis_names = payload.get("selectedAxes") or [axis["name"] for axis in goal["axes"]]
    if not isinstance(selected_axis_names, list):
        raise ValueError("selectedAxes must be a list.")
    preview_axes = payload.get("previewAxes")
    if not isinstance(preview_axes, list):
        raise ValueError("previewAxes must be a list.")

    preview_goal = grid_preview_goal(goal, [str(name) for name in selected_axis_names], preview_axes)
    preview_by_key = {normalize_axis_name(axis["name"]): axis for axis in preview_goal["axes"]}
    updated_goal = dict(goal)
    updated_axes: list[dict[str, Any]] = []
    for axis in goal["axes"]:
        preview = preview_by_key.get(normalize_axis_name(axis["name"]))
        if preview:
            updated = dict(axis)
            updated["domainMin"] = float(preview["domainMin"])
            updated["domainMax"] = float(preview["domainMax"])
            updated["resolution"] = float(preview["resolution"])
            updated_axes.append(updated)
        else:
            updated_axes.append(dict(axis))
    updated_goal["axes"] = updated_axes
    updated_goal = validate_goal(updated_goal)
    updated_goals = [updated_goal if item["id"] == goal_id else item for item in goals]

    updated_clusters: list[dict[str, Any]] = []
    updated_count = 0
    legacy_excluded = 0
    axis_excluded = 0
    for cluster in load_cluster_store():
        if str(cluster.get("goalId")) != goal_id:
            updated_clusters.append(cluster)
            continue
        try:
            cluster_goal = goal_subset(updated_goal, [str(name) for name in cluster.get("axisNames", [])])
        except ValueError:
            axis_excluded += 1
            updated_clusters.append(cluster)
            continue
        rebuilt, changed, reason = rebuild_record_for_goal_grid(cluster, cluster_goal)
        if changed:
            updated_count += 1
        elif reason == "legacy_missing_row_level_vectors":
            legacy_excluded += 1
        else:
            axis_excluded += 1
        updated_clusters.append(rebuilt)

    save_goal_store(updated_goals)
    save_cluster_store(updated_clusters)
    normalized_goals = normalize_goals_for_display(updated_goals)
    return {
        "savedGoal": updated_goal,
        "updatedRecordCount": updated_count,
        "legacyExcludedRecordCount": legacy_excluded,
        "axisExcludedRecordCount": axis_excluded,
        "goals": normalized_goals,
        "clusters": list_cluster_summaries(),
        "peerCounts": bootstrap_peer_counts(),
        "peerSubsetCounts": bootstrap_peer_subset_counts(),
        "goalBinPreview": {goal["id"]: goal_bin_preview(goal) for goal in normalized_goals},
    }


def build_bootstrap_payload(admin_allowed: bool) -> dict[str, Any]:
    goals = normalize_goals_for_display(load_goal_store())
    peer_counts = bootstrap_peer_counts()
    peer_subset_counts = bootstrap_peer_subset_counts()
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
            "savedItems": ["Experiment Goal 설정", "selected-axis row-level sanitized numeric vectors", "비식별 mean vector 호환 필드", "row-level bin count summary"],
            "unsavedItems": ["원본 CSV 파일", "파일명", "비매핑 column", "개인정보 column"],
        },
        "domainDefinitions": {
            "cluster": "CSV 파일 하나는 저장 record로 처리됩니다. 저장/삭제는 record 단위지만, density calculation은 record 내부 row-level sanitized axis vectors를 현재 grid로 다시 binning해서 수행합니다.",
            "coverage": "Coverage and Equitability are calculated from row-level vectors re-binned into the current density grid.",
            "K_m": "K_m is currently reused as K_density for Observation Support: S = peer row-level observations / (peer row-level observations + K_density).",
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
            if parsed.path == "/api/grid-preview":
                self._send_json(grid_preview_request(payload))
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
            if parsed.path == "/api/admin/goals/validate-expressions":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin Token이 필요합니다."}, status=HTTPStatus.FORBIDDEN)
                    return
                validated_goal = validate_goal(payload)
                self._send_json({"ok": True, "goal": validated_goal, "maskInfo": feasible_mask_info_for_goal(validated_goal)})
                return
            if parsed.path == "/api/admin/goals/apply-grid":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin Token이 필요합니다."}, status=HTTPStatus.FORBIDDEN)
                    return
                self._send_json(apply_goal_grid_defaults_request(payload))
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
                    raise ValueError("삭제할 record를 찾을 수 없습니다.")
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

