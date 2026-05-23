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
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from feasible_box_counter import (
    axis_bin_counts,
    bin_tuple_masked_by_boxes,
    compile_rules_to_mask_boxes,
    compute_a_valid_for_rules,
    compute_rectangular_total_bins,
    mask_signature,
)
from feasible_mask import normalize_feasible_expressions
from models import K_M, DensityDiagnosisResult, ExperimentConfig
from stats_engine import BinGridTracker, DensityGridAnalyzer, compute_density_confidence
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
    storage_status,
    utc_now_iso,
    validate_goal,
)


TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"
PROJECTION_ROW_TUPLE_LIMIT = int(os.environ.get("MAX_TARGET_TUPLES", "5000"))
MAX_PROJECTION_CELLS = int(os.environ.get("MAX_PROJECTION_CELLS", "50000"))
MAX_PROJECTION_PAIRS = int(os.environ.get("MAX_PROJECTION_PAIRS", "36"))
MAX_GRID_PREVIEW_BINS = int(os.environ.get("MAX_GRID_PREVIEW_BINS", "500000"))
CERTIFIED_FEASIBLE_DOMAIN_MESSAGE = (
    "Feasible Domain is defined by exact 2D Projection Mask boxes. "
    "Leesin does not materialize the full multidimensional grid."
)
A_VALID_JOBS: dict[str, dict[str, Any]] = {}
SERVER_INSTANCE_ID = os.environ.get("RENDER_INSTANCE_ID") or os.environ.get("HOSTNAME") or uuid.uuid4().hex[:12]
SERVER_STARTED_AT = datetime.now(timezone.utc).isoformat()


def canonical_axis_order(axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(axis) for axis in axes), key=lambda axis: normalize_axis_name(axis.get("name")))


def axis_role(axis: dict[str, Any]) -> str:
    return "output" if str(axis.get("role") or "").strip().lower() == "output" else "input"


def input_axes_for_goal(goal: dict[str, Any]) -> list[dict[str, Any]]:
    return [axis for axis in canonical_axis_order(goal.get("axes", [])) if axis_role(axis) == "input"]


def output_axes_for_goal(goal: dict[str, Any]) -> list[dict[str, Any]]:
    return [axis for axis in canonical_axis_order(goal.get("axes", [])) if axis_role(axis) == "output"]


def goal_subset(goal: dict[str, Any], selected_axis_names: list[str] | None = None) -> dict[str, Any]:
    if not selected_axis_names:
        axes = canonical_axis_order(goal["axes"])
    else:
        requested = {normalize_axis_name(name) for name in selected_axis_names if normalize_axis_name(name)}
        axes = canonical_axis_order([axis for axis in goal["axes"] if normalize_axis_name(axis["name"]) in requested])
    if not axes:
        raise ValueError("분석에 포함할 Axis를 하나 이상 선택하세요.")
    axis_name_set = {str(axis["name"]) for axis in axes}
    input_axis_name_set = {str(axis["name"]) for axis in axes if axis_role(axis) == "input"}
    if not input_axis_name_set:
        raise ValueError("At least one input axis is required for Confidence and feasible design masks.")
    feasible_rules = [
        dict(rule)
        for rule in goal.get("feasibleDomainRules") or []
        if rule.get("enabled", True) and rule_axes(rule).issubset(input_axis_name_set)
    ]
    feasible_expressions = [
        str(rule.get("expression") or "").strip()
        for rule in feasible_rules
        if rule.get("enabled", True) and str(rule.get("expression") or "").strip()
    ]
    return {
        "id": goal["id"],
        "name": goal["name"],
        "K_m": float(goal.get("K_m", K_M)),
        "axes": axes,
        "inputAxisNames": [axis["name"] for axis in axes if axis_role(axis) == "input"],
        "outputAxisNames": [axis["name"] for axis in axes if axis_role(axis) == "output"],
        "feasibleDomainRules": feasible_rules,
        "legacyAdvancedExpressions": normalize_feasible_expressions(goal.get("legacyAdvancedExpressions")),
        "feasibleDomainAdvancedExpressions": [],
        "generatedFeasibleExpressions": feasible_expressions,
        "feasibleDomainExpressions": feasible_expressions,
        "aValidCache": dict(goal.get("aValidCache") or {}) if isinstance(goal.get("aValidCache"), dict) else {},
        "aValid": goal.get("aValid"),
        "maskedBins": goal.get("maskedBins"),
        "rectangularTotalBins": goal.get("rectangularTotalBins"),
        "aValidStatus": goal.get("aValidStatus"),
        "aValidProgressPercent": goal.get("aValidProgressPercent"),
        "aValidMode": goal.get("aValidMode"),
        "maskSignature": goal.get("maskSignature"),
        "aValidComputedAt": goal.get("aValidComputedAt"),
    }


def rule_axes(rule: dict[str, Any]) -> set[str]:
    gui_spec = rule.get("guiSpec") if isinstance(rule.get("guiSpec"), dict) else {}
    rule_type = str(gui_spec.get("type") or rule.get("sourceType") or "")
    axes: set[str] = set()
    if rule_type == "focused_2d_mask" or rule.get("sourceType") == "focused_2d_mask":
        axes.update(
            axis
            for axis in (str(gui_spec.get("xAxis") or "").strip(), str(gui_spec.get("yAxis") or "").strip())
            if axis
        )
        scope = gui_spec.get("scope") if isinstance(gui_spec.get("scope"), dict) else {}
        for axis, scope_spec in scope.items():
            if isinstance(scope_spec, dict) and str(scope_spec.get("mode") or "all").lower() == "range":
                axes.add(str(axis))
        return axes
    if_spec = gui_spec.get("if") if isinstance(gui_spec.get("if"), dict) else {}
    then_spec = gui_spec.get("then") if isinstance(gui_spec.get("then"), dict) else {}
    axes.update(axis for axis in (str(if_spec.get("axis") or "").strip(), str(then_spec.get("axis") or "").strip()) if axis)
    return axes


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


def mask_info_for_display(mask_info: dict[str, Any]) -> dict[str, Any]:
    payload = dict(mask_info)
    if payload.get("feasibleMaskEvaluationSkipped"):
        payload["validBins"] = None
        payload["maskedBins"] = None
        payload["validDomainRatio"] = None
    return payload


def api_error_payload(exc: Exception) -> dict[str, Any]:
    message = str(exc) or "Request failed."
    error_type = exc.__class__.__name__
    lower = message.lower()
    recoverable = any(
        marker in lower
        for marker in ("too large", "skipped", "invalid feasible", "unknown axis", "timeout", "requires", "must")
    )
    hint = ""
    if "too large" in lower or "skipped" in lower:
        hint = "The grid may be too large; try coarser resolution or narrower selected axes."
    elif "unknown axis" in lower or "invalid feasible" in lower:
        hint = "Check that expression variables exactly match the current goal axis names."
    return {
        "error": message,
        "errorType": error_type,
        "hint": hint,
        "recoverable": bool(recoverable),
    }


def nullable_numeric_delta(current: Any, previous: Any) -> float | None:
    if current is None or previous is None:
        return None
    try:
        return round(float(current) - float(previous), 6)
    except (TypeError, ValueError):
        return None


def feasible_mask_info_for_goal(selected_goal: dict[str, Any]) -> dict[str, Any]:
    axes = input_axes_for_goal(selected_goal)
    if not axes:
        raise ValueError("At least one input axis is required for a_valid.")
    rules = selected_goal.get("feasibleDomainRules") or []
    boxes = compile_rules_to_mask_boxes(rules, axes)
    total_bins = compute_rectangular_total_bins(axes)
    current_signature = mask_signature(axes, boxes)
    cache = selected_goal.get("aValidCache") if isinstance(selected_goal.get("aValidCache"), dict) else {}
    cache_ready = (
        cache.get("maskSignature") == current_signature
        and cache.get("aValid") is not None
        and cache.get("maskedBins") is not None
        and cache.get("rectangularTotalBins") == total_bins
    )
    if not boxes:
        cache_ready = True
        cache = {
            "maskSignature": current_signature,
            "rectangularTotalBins": total_bins,
            "maskedBins": 0,
            "aValid": total_bins,
            "aValidMode": "exact_box_union",
            "computedAt": selected_goal.get("aValidComputedAt"),
            "durationMs": 0,
        }
    info = {
        "totalBins": int(total_bins),
        "rectangularTotalBins": int(total_bins),
        "validBins": int(cache["aValid"]) if cache_ready else None,
        "aValid": int(cache["aValid"]) if cache_ready else None,
        "maskedBins": int(cache["maskedBins"]) if cache_ready else None,
        "maskBoxes": boxes,
        "maskBoxCount": len(boxes),
        "feasibleMaskEnabled": bool(boxes),
        "validDomainRatio": float(cache["aValid"] / total_bins) if cache_ready and total_bins else None,
        "aValidStatus": "ready" if cache_ready else str(selected_goal.get("aValidStatus") or "stale"),
        "aValidProgressPercent": 100 if cache_ready else int(selected_goal.get("aValidProgressPercent") if selected_goal.get("aValidProgressPercent") is not None else 0),
        "aValidMode": "exact_box_union",
        "coverageDenominator": int(cache["aValid"]) if cache_ready else None,
        "maskSignature": current_signature,
        "aValidComputedAt": cache.get("computedAt") if cache_ready else None,
        "aValidCache": dict(cache) if cache else {},
        "feasibleMaskEvaluationSkipped": False,
        "feasibleMaskWarning": "",
        "coverageWarning": "",
    }
    expressions = feasible_expressions_for_goal(selected_goal)
    info["feasibleExpressions"] = expressions
    info["generatedFeasibleExpressions"] = expressions
    info["certifiedFeasibleDomain"] = True
    info["feasibleMaskMessage"] = CERTIFIED_FEASIBLE_DOMAIN_MESSAGE
    info.setdefault("feasibleMaskEvaluationSkipped", False)
    info.setdefault("feasibleMaskWarning", "")
    info.setdefault("coverageWarning", "")
    return info


def is_bin_key_feasible_for_goal(bin_key: str, selected_goal: dict[str, Any], cache: dict[str, bool] | None = None) -> bool:
    cache_key = str(bin_key)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    axes = canonical_axis_order(selected_goal["axes"])
    axis_order = [str(axis["name"]) for axis in axes]
    try:
        indices = json.loads(cache_key)
    except json.JSONDecodeError:
        result = False
    else:
        boxes_key = "__maskBoxes"
        counts_key = "__axisCounts"
        if cache is not None and boxes_key in cache:  # type: ignore[operator]
            boxes = cache[boxes_key]  # type: ignore[index]
            counts = cache[counts_key]  # type: ignore[index]
        else:
            boxes = compile_rules_to_mask_boxes(selected_goal.get("feasibleDomainRules") or [], axes)
            counts = axis_bin_counts(axes)
            if cache is not None:
                cache[boxes_key] = boxes  # type: ignore[index]
                cache[counts_key] = counts  # type: ignore[index]
        result = not bin_tuple_masked_by_boxes(indices, axis_order, boxes, counts)
    if cache is not None:
        cache[cache_key] = bool(result)
    return bool(result)


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


def projection_cell_count(axis_meta: dict[str, dict[str, Any]], x_axis: str, y_axis: str) -> tuple[int, int, int]:
    x_bins = int(axis_meta.get(x_axis, {}).get("totalBins") or 0)
    y_bins = int(axis_meta.get(y_axis, {}).get("totalBins") or 0)
    return x_bins, y_bins, int(max(0, x_bins) * max(0, y_bins))


def skipped_projection_payload(x_axis: str, y_axis: str, x_bins: int, y_bins: int, reason: str) -> dict[str, Any]:
    return {
        "xAxis": x_axis,
        "yAxis": y_axis,
        "xBins": int(x_bins),
        "yBins": int(y_bins),
        "counts": [],
        "maxCount": 0,
        "projectionSkipped": True,
        "reason": reason,
    }


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
    x_bins, y_bins, cell_count = projection_cell_count(axis_meta, x_axis, y_axis)
    if cell_count > MAX_PROJECTION_CELLS:
        return skipped_projection_payload(
            x_axis,
            y_axis,
            x_bins,
            y_bins,
            "Projection cell count exceeds MAX_PROJECTION_CELLS.",
        )
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
    x_bins, y_bins, cell_count = projection_cell_count(axis_meta, x_axis, y_axis)
    if cell_count > MAX_PROJECTION_CELLS:
        return skipped_projection_payload(
            x_axis,
            y_axis,
            x_bins,
            y_bins,
            "Projection cell count exceeds MAX_PROJECTION_CELLS.",
        )
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
    all_axes = canonical_axis_order(goal["axes"])
    axes = [axis for axis in all_axes if axis_role(axis) == "input"]
    if not axes:
        axes = all_axes
    axis_index_by_name = {axis["name"]: index for index, axis in enumerate(all_axes)}
    axis_order = [str(axis["name"]) for axis in axes]
    axis_meta = projection_axis_meta(axes)
    all_axis_pairs = [[x_axis, y_axis] for x_axis, y_axis in itertools.combinations(axis_order, 2)]
    axis_pairs = all_axis_pairs[:MAX_PROJECTION_PAIRS]
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
        "allAxisPairCount": len(all_axis_pairs),
        "projectionPairTruncated": len(all_axis_pairs) > len(axis_pairs),
        "maxProjectionPairs": MAX_PROJECTION_PAIRS,
        "maxProjectionCells": MAX_PROJECTION_CELLS,
        "peerProjections": peer_projections,
        "targetProjections": target_projections,
        "feasibleMaskEnabled": bool(coverage_info.get("feasibleMaskEnabled")) if isinstance(coverage_info, dict) else False,
        "validBins": int(coverage_info.get("validBins")) if isinstance(coverage_info, dict) and coverage_info.get("validBins") is not None else None,
        "aValid": int(coverage_info.get("aValid")) if isinstance(coverage_info, dict) and coverage_info.get("aValid") is not None else None,
        "rectangularTotalBins": int(coverage_info.get("rectangularTotalBins") or coverage_info.get("totalBins") or 0) if isinstance(coverage_info, dict) else 0,
        "maskedBins": int(coverage_info.get("maskedBins")) if isinstance(coverage_info, dict) and coverage_info.get("maskedBins") is not None else None,
        "maskBoxCount": int(coverage_info.get("maskBoxCount") or 0) if isinstance(coverage_info, dict) else 0,
        "validDomainRatio": float(coverage_info.get("validDomainRatio")) if isinstance(coverage_info, dict) and coverage_info.get("validDomainRatio") is not None else None,
        "feasibleExpressions": list(coverage_info.get("feasibleExpressions") or []) if isinstance(coverage_info, dict) else [],
        "feasibleMaskEvaluationSkipped": bool(coverage_info.get("feasibleMaskEvaluationSkipped")) if isinstance(coverage_info, dict) else False,
        "feasibleMaskWarning": str(coverage_info.get("feasibleMaskWarning") or "") if isinstance(coverage_info, dict) else "",
        "coverageWarning": str(coverage_info.get("coverageWarning") or "") if isinstance(coverage_info, dict) else "",
        "feasibleMaskMessage": str(coverage_info.get("feasibleMaskMessage") or CERTIFIED_FEASIBLE_DOMAIN_MESSAGE) if isinstance(coverage_info, dict) else CERTIFIED_FEASIBLE_DOMAIN_MESSAGE,
        "targetIncludedInReference": bool(coverage_info.get("targetIncludedInReference", True)) if isinstance(coverage_info, dict) else True,
        "internalDensityMode": bool(coverage_info.get("internalDensityMode")) if isinstance(coverage_info, dict) else False,
        "externalPeerObservationCount": int(coverage_info.get("externalPeerObservationCount") or 0) if isinstance(coverage_info, dict) else 0,
        "referenceObservationCount": int(coverage_info.get("referenceObservationCount") or 0) if isinstance(coverage_info, dict) else 0,
        "maskRenderingTodo": "TODO: render masked 2D projection regions as gray overlay.",
        "targetRowBinTuples": payload_tuples,
        "targetRowTupleSampled": tuple_sampled,
        "targetRowTupleLimit": PROJECTION_ROW_TUPLE_LIMIT,
        "targetRowTupleCount": len(target_tuples),
    }


def density_preview_metrics_from_coverage(selected_goal: dict[str, Any], coverage_info: dict[str, Any]) -> dict[str, Any]:
    bin_counts = (coverage_info.get("designBinCounts") or coverage_info.get("binCounts", {})) if isinstance(coverage_info, dict) else {}
    counts = [int(value) for value in bin_counts.values() if int(value) > 0]
    peer_valid_rows = int(sum(counts))
    mask_info = feasible_mask_info_for_goal(selected_goal)
    total_bins = int(coverage_info.get("totalBins") or mask_info["totalBins"])
    valid_bins = coverage_info.get("validBins", mask_info.get("validBins"))
    valid_bins = int(valid_bins) if valid_bins is not None else None
    masked_bins = coverage_info.get("maskedBins", mask_info.get("maskedBins"))
    masked_bins = int(masked_bins) if masked_bins is not None else None
    occupied_bins = len(counts)
    observation_support = peer_valid_rows / (peer_valid_rows + float(selected_goal.get("K_m", K_M))) if peer_valid_rows > 0 else 0.0
    coverage = occupied_bins / valid_bins if valid_bins else None
    if occupied_bins <= 1 or peer_valid_rows <= 0:
        equitability = 0.0
    else:
        proportions = [count / peer_valid_rows for count in counts]
        equitability = -sum(p * math.log(p) for p in proportions if p > 0) / math.log(occupied_bins)
    confidence = float((observation_support * coverage * equitability) ** (1.0 / 3.0)) if observation_support and coverage and equitability else None
    return {
        "totalBins": int(total_bins),
        "validBins": valid_bins,
        "maskedBins": masked_bins,
        "validDomainRatio": round(float(valid_bins / total_bins), 6) if total_bins and valid_bins is not None else None,
        "feasibleMaskEnabled": bool(mask_info.get("feasibleMaskEnabled")),
        "feasibleExpressions": list(mask_info.get("feasibleExpressions") or []),
        "feasibleMaskEvaluationSkipped": False,
        "feasibleMaskWarning": "",
        "coverageWarning": "",
        "aValid": mask_info.get("aValid") if mask_info.get("aValid") is not None else valid_bins,
        "rectangularTotalBins": int(mask_info.get("rectangularTotalBins") or total_bins),
        "maskBoxCount": int(mask_info.get("maskBoxCount") or 0),
        "aValidStatus": str(mask_info.get("aValidStatus") or "stale"),
        "aValidProgressPercent": int(mask_info.get("aValidProgressPercent") if mask_info.get("aValidProgressPercent") is not None else 0),
        "aValidMode": str(mask_info.get("aValidMode") or "exact_box_union"),
        "feasibleMaskMessage": str(mask_info.get("feasibleMaskMessage") or CERTIFIED_FEASIBLE_DOMAIN_MESSAGE),
        "occupiedBins": int(occupied_bins),
        "peerValidRows": int(peer_valid_rows),
        "referenceObservationCount": int(peer_valid_rows),
        "targetIncludedInReference": True,
        "observationSupportZ": round(float(observation_support), 6),
        "coverageC": None if coverage is None else round(float(coverage), 6),
        "equitabilityE": round(float(equitability), 6),
        "confidence": None if confidence is None else round(float(confidence), 6),
        "coveragePending": coverage is None,
        "confidencePending": confidence is None,
        "confidenceExact": confidence is not None,
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
    all_axes = canonical_axis_order(goal["axes"])
    axes = [axis for axis in all_axes if axis_role(axis) == "input"] or all_axes
    axis_index_by_name = {axis["name"]: index for index, axis in enumerate(all_axes)}
    coverage_axes = []
    equitability_axes = []
    axis_bin_counts = coverage_info.get("axisBinCounts", {}) if isinstance(coverage_info, dict) else {}
    axis_bin_counts_by_key = {normalize_axis_name(axis_name): counts for axis_name, counts in axis_bin_counts.items()}

    for axis in axes:
        index = axis_index_by_name.get(axis["name"], 0)
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
        "coverage": {
            "score": None if result.coverage_C is None else round(float(result.coverage_C), 6),
            "pending": result.coverage_C is None,
            "basis": visualization_basis,
            "axes": coverage_axes,
        },
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
    coverage_reason = {
        "label": "Coverage",
        "score": None,
        "impact": "pending",
        "message": "a_valid exact calculation is not ready. Coverage and Confidence will be available after Compute a_valid finishes.",
    }
    if result.coverage_C is not None:
        coverage_reason = {
            "label": "Coverage",
            "score": round(float(result.coverage_C), 4),
            "impact": "down" if result.coverage_C < 0.3 else "stable",
            "message": "Coverage measures how much of the feasible input design grid is occupied by target-included reference row-level observations. Output axes are excluded from Coverage.",
        }
    reasons = [
        {
            "label": "Observation Support",
            "score": round(float(result.observation_support_S), 4),
            "impact": "down" if result.observation_support_S < 0.6 else "stable",
            "message": "Observation Support uses target-included reference row-level observations: S = reference rows / (reference rows + K_density). K_density currently reuses the goal K_m value.",
        },
        coverage_reason,
        {
            "label": "Equitability",
            "score": round(float(result.equitability_E), 4),
            "impact": "down" if result.equitability_E < 0.5 else "stable",
            "message": "Equitability measures whether reference observations are balanced across occupied input design bins.",
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
    if result.confidence is None:
        messages.append("a_valid exact calculation is not ready. Specificity, Observation Support, and Equitability are available now; Coverage and Confidence will be available after Compute a_valid finishes.")
    if result.specificity_score > 0.75 and result.confidence is not None and result.confidence > 0.7:
        messages.append("Specificity Score and confidence are both high. Target rows fall in low-density bins relative to the target-included reference count distribution.")
    elif result.specificity_score > 0.75 and (result.confidence is None or result.confidence <= 0.4):
        messages.append("Target rows are rare against the current target-included reference density map, but the map itself has limited support. Interpret the outlier signal cautiously.")
    elif result.specificity_score <= 0.5:
        messages.append("Target rows mostly fall in dense or ordinary occupied reference bins.")
    else:
        messages.append("The target shows moderate specificity. Additional external peer observations can make the reference density baseline sharper.")

    if result.observation_support_S < 0.5:
        messages.append("Observation Support is low. Add more row-level observations or saved external records with row-level sanitized axis vectors.")
    if result.coverage_C is not None and result.coverage_C < 0.3:
        messages.append("Coverage is low. The target-included reference density map occupies only a small portion of the feasible input design grid.")
    if result.equitability_E < 0.5:
        messages.append("Equitability is low. Reference observations are concentrated in a small number of occupied bins.")
    if result.unseen_bin_rate > 0:
        messages.append(f"{result.unseen_bin_rate:.1%} of valid target rows fall in bins unseen in the reference density map.")
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


def rectangular_total_bins_for_axes(axes: list[dict[str, Any]]) -> int:
    total = 1
    for axis in canonical_axis_order(axes):
        total *= max(1, int(np.ceil((float(axis["domainMax"]) - float(axis["domainMin"])) / float(axis["resolution"]))))
    return int(total)


def skipped_grid_preview_payload(preview_goal: dict[str, Any], total_bins: int) -> dict[str, Any]:
    mask_info = mask_info_for_display(feasible_mask_info_for_goal(preview_goal))
    warning = (
        "Projection preview recalculation was limited because total bin count exceeds MAX_GRID_PREVIEW_BINS. "
        "Apply as Goal Default is still available after confirmation."
    )
    metrics = {
        "totalBins": int(total_bins),
        "validBins": mask_info.get("validBins"),
        "maskedBins": mask_info.get("maskedBins"),
        "validDomainRatio": mask_info.get("validDomainRatio"),
        "feasibleMaskEnabled": bool(mask_info.get("feasibleMaskEnabled")),
        "feasibleExpressions": list(mask_info.get("feasibleExpressions") or []),
        "feasibleMaskEvaluationSkipped": False,
        "feasibleMaskWarning": "",
        "coverageWarning": "",
        "aValid": mask_info.get("aValid", mask_info.get("validBins")),
        "rectangularTotalBins": mask_info.get("rectangularTotalBins", total_bins),
        "maskBoxCount": mask_info.get("maskBoxCount", 0),
        "aValidStatus": mask_info.get("aValidStatus", "stale"),
        "gridPreviewSkipped": True,
        "previewWarning": warning,
        "occupiedBins": 0,
        "peerValidRows": 0,
        "referenceObservationCount": 0,
        "targetIncludedInReference": True,
        "observationSupportZ": 0.0,
        "coverageC": None,
        "equitabilityE": 0.0,
        "confidence": None,
        "coveragePending": True,
        "confidencePending": True,
        "confidenceExact": False,
        "eligibleRecords": 0,
        "legacyExcluded": 0,
        "gridSignatureExcluded": 0,
        "gridSignature": grid_signature_from_axes(canonical_axis_order(preview_goal["axes"])),
    }
    projection_explorer = build_projection_explorer(
        preview_goal,
        {
            "binCounts": {},
            "axisBinCounts": {},
            "totalBins": int(total_bins),
            "validBins": mask_info.get("validBins"),
            "maskedBins": mask_info.get("maskedBins"),
            "validDomainRatio": mask_info.get("validDomainRatio"),
            "feasibleMaskEnabled": bool(mask_info.get("feasibleMaskEnabled")),
            "feasibleExpressions": list(mask_info.get("feasibleExpressions") or []),
            "feasibleMaskEvaluationSkipped": False,
            "feasibleMaskWarning": "",
            "coverageWarning": "",
            "aValid": mask_info.get("aValid", mask_info.get("validBins")),
            "rectangularTotalBins": mask_info.get("rectangularTotalBins", total_bins),
            "maskBoxCount": mask_info.get("maskBoxCount", 0),
        },
        [],
    )
    projection_explorer["gridPreviewMetrics"] = metrics
    return {
        "previewGoal": preview_goal,
        "metrics": metrics,
        "result": None,
        "projectionExplorer": projection_explorer,
        "targetBinOccupancy": {},
        "targetBinOccupancyMeta": {
            "validMultidimensionalRowCount": 0,
            "invalidRowCount": 0,
            "outOfDomainRowCount": 0,
            "maskedOutRowCount": 0,
            "totalRows": 0,
        },
        "coverageInfo": {key: value for key, value in metrics.items() if key not in {"feasibleExpressions"}},
    }


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


def design_goal_for_selected(goal: dict[str, Any], selected_goal: dict[str, Any]) -> dict[str, Any]:
    design_axis_names = [axis["name"] for axis in canonical_axis_order(selected_goal["axes"]) if axis_role(axis) == "input"]
    if not design_axis_names:
        raise ValueError("At least one input axis is required for Confidence.")
    return goal_subset(goal, design_axis_names)


def recompute_design_counts_from_vectors(
    row_vectors: list[Any],
    axis_order: list[str],
    goal: dict[str, Any],
    selected_goal: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    design_goal = design_goal_for_selected(goal, selected_goal)
    recomputed = recompute_bin_occupancy_from_row_vectors(row_vectors, axis_order, design_goal)
    if recomputed is None:
        return {}, {"validMultidimensionalRowCount": 0, "totalRows": 0}
    return recomputed["bin_occupancy"], recomputed["bin_occupancy_meta"]


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
            "rectangularTotalBins": 0,
            "validBins": 0,
            "aValid": 0,
            "maskedBins": 0,
            "maskBoxes": [],
            "maskBoxCount": 0,
            "feasibleExpressions": [],
            "feasibleMaskEnabled": False,
            "validDomainRatio": 0.0,
            "feasibleMaskEvaluationSkipped": False,
            "feasibleMaskWarning": "",
            "coverageWarning": "",
            "aValidStatus": "ready",
            "aValidProgressPercent": 100,
            "aValidMode": "exact_box_union",
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
        "validBins": int(mask_info["validBins"]) if mask_info.get("validBins") is not None else None,
        "aValid": int(mask_info["aValid"]) if mask_info.get("aValid") is not None else None,
        "rectangularTotalBins": int(mask_info.get("rectangularTotalBins") or mask_info.get("totalBins") or 0),
        "maskedBins": int(mask_info["maskedBins"]) if mask_info.get("maskedBins") is not None else None,
        "maskBoxes": list(mask_info.get("maskBoxes") or []),
        "maskBoxCount": int(mask_info.get("maskBoxCount") or 0),
        "validDomainRatio": None if mask_info.get("validDomainRatio") is None else float(mask_info.get("validDomainRatio") or 0.0),
        "feasibleMaskEnabled": bool(mask_info.get("feasibleMaskEnabled")),
        "feasibleExpressions": list(mask_info.get("feasibleExpressions") or []),
        "generatedFeasibleExpressions": list(mask_info.get("generatedFeasibleExpressions") or mask_info.get("feasibleExpressions") or []),
        "certifiedFeasibleDomain": bool(mask_info.get("certifiedFeasibleDomain", True)),
        "feasibleMaskMessage": str(mask_info.get("feasibleMaskMessage") or CERTIFIED_FEASIBLE_DOMAIN_MESSAGE),
        "feasibleMaskEvaluationSkipped": bool(mask_info.get("feasibleMaskEvaluationSkipped")),
        "feasibleMaskWarning": str(mask_info.get("feasibleMaskWarning") or ""),
        "coverageWarning": str(mask_info.get("coverageWarning") or ""),
        "aValidStatus": str(mask_info.get("aValidStatus") or "stale"),
        "aValidProgressPercent": int(mask_info.get("aValidProgressPercent") if mask_info.get("aValidProgressPercent") is not None else 0),
        "aValidMode": str(mask_info.get("aValidMode") or "exact_box_union"),
        "maskSignature": str(mask_info.get("maskSignature") or ""),
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
    target_design_bin_counts: dict[str, int] | None = None,
    target_design_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    axis_names = [axis["name"] for axis in canonical_axis_order(selected_goal["axes"])]
    design_goal = design_goal_for_selected(goal, selected_goal)
    design_axis_names = [axis["name"] for axis in canonical_axis_order(design_goal["axes"])]
    output_axis_names = [axis["name"] for axis in canonical_axis_order(selected_goal["axes"]) if axis_role(axis) == "output"]
    if target_design_bin_counts is None:
        target_design_bin_counts = target_bin_counts if design_axis_names == axis_names else {}
    if target_design_meta is None:
        target_design_meta = target_meta if design_axis_names == axis_names else {}
    config = experiment_config_from_goal(selected_goal)
    peer_clusters = analysis_peer_clusters(goal, axis_names, exclude_cluster_id)
    peer_rows = [{"id": str(cluster.get("id", "")), "source": "stored", "values": [float(value) for value in cluster["values"]]} for cluster in peer_clusters]
    peer_group = peer_matrix(peer_rows)
    if peer_group.size == 0:
        peer_group = np.empty((0, len(axis_names)), dtype=float)
    coverage_info = build_global_bin_counts(peer_clusters, selected_goal)
    design_coverage_info = build_global_bin_counts(peer_clusters, design_goal)
    warnings = out_of_domain_warnings(selected_goal, target_meta)
    external_bin_counts = {str(key): int(value) for key, value in (coverage_info.get("binCounts") or {}).items()}
    reference_bin_counts = dict(external_bin_counts)
    merge_count_maps(reference_bin_counts, target_bin_counts)
    design_external_bin_counts = {str(key): int(value) for key, value in (design_coverage_info.get("binCounts") or {}).items()}
    design_reference_bin_counts = dict(design_external_bin_counts)
    merge_count_maps(design_reference_bin_counts, target_design_bin_counts or {})
    external_peer_record_count = int(coverage_info.get("coverageEligibleClusterCount") or 0)
    external_peer_observation_count = count_map_total(external_bin_counts)
    reference_observation_count = count_map_total(reference_bin_counts)
    reference_occupied_bins = len(reference_bin_counts)
    internal_density_mode = external_peer_record_count == 0 or external_peer_observation_count == 0
    self_contained_warning = bool(internal_density_mode and reference_observation_count > 0)
    if self_contained_warning:
        warnings.append(
            {
                "role": "reference",
                "internalDensityMode": True,
                "message": (
                    "특이도는 표적 군집을 포함한 reference density map 기준으로 계산됩니다. "
                    "외부 peer record가 없는 경우, 이 값은 과거 실험 대비 이상치가 아니라 "
                    "업로드된 데이터 내부의 상대적 희소 구간을 의미합니다."
                ),
            }
        )
    analyzer = DensityGridAnalyzer(config)
    analyzer.set_peer_bin_counts(reference_bin_counts)
    analyzer.set_feasible_domain(
        valid_bins=coverage_info.get("validBins"),
        masked_bins=int(coverage_info.get("maskedBins") or 0) if coverage_info.get("maskedBins") is not None else 0,
        feasible_mask_enabled=bool(coverage_info.get("feasibleMaskEnabled")),
    )
    result = analyzer.diagnose(target_bin_counts, target_meta)
    design_observation_support, design_coverage, design_equitability, design_confidence = compute_density_confidence(
        design_reference_bin_counts,
        design_coverage_info.get("validBins"),
        float(selected_goal.get("K_m", K_M)),
    )
    result.observation_support_S = float(design_observation_support)
    result.coverage_C = None if design_coverage is None else float(design_coverage)
    result.equitability_E = float(design_equitability)
    result.confidence = design_confidence
    result.peer_observation_count = count_map_total(design_reference_bin_counts)
    result.valid_bins = int(design_coverage_info["validBins"]) if design_coverage_info.get("validBins") is not None else None
    result.masked_bins = int(design_coverage_info["maskedBins"]) if design_coverage_info.get("maskedBins") is not None else 0
    result.occupied_bins = len(design_reference_bin_counts)
    result_payload = result.to_payload(config.axis_names)
    result_payload["specificityTotalBins"] = result_payload.get("total_bins")
    result_payload["totalBins"] = design_coverage_info.get("rectangularTotalBins", result_payload.get("total_bins"))
    result_payload["validTargetRows"] = result_payload.get("valid_target_rows")
    result_payload["outOfDomainRows"] = result_payload.get("out_of_domain_rows")
    result_payload["maskedOutTargetRows"] = result_payload.get("masked_out_target_rows")
    result_payload["infeasibleTargetRows"] = result_payload.get("infeasible_target_rows")
    result_payload["occupiedBins"] = result_payload.get("occupied_bins")
    result_payload["outOfDomainWarnings"] = warnings
    result_payload["outOfDomainWarningCount"] = len(warnings)
    result_payload["coverageBasis"] = "input_design_row_level_bin_occupancy"
    result_payload["coverageEligibleClusterCount"] = design_coverage_info["coverageEligibleClusterCount"]
    result_payload["coverageLegacyExcludedClusterCount"] = design_coverage_info["coverageLegacyExcludedClusterCount"]
    result_payload["coverageGridSignatureExcludedClusterCount"] = design_coverage_info["coverageGridSignatureExcludedClusterCount"]
    result_payload["coverageAxisSignatureExcludedClusterCount"] = design_coverage_info["coverageGridSignatureExcludedClusterCount"]
    result_payload["rowLevelObservationCount"] = reference_observation_count
    result_payload["gridSignature"] = coverage_info["gridSignature"]
    result_payload["validBins"] = design_coverage_info.get("validBins", result.valid_bins)
    result_payload["maskedBins"] = design_coverage_info.get("maskedBins", result.masked_bins)
    result_payload["aValid"] = design_coverage_info.get("aValid", design_coverage_info.get("validBins", result.valid_bins))
    result_payload["rectangularTotalBins"] = design_coverage_info.get("rectangularTotalBins")
    result_payload["aValidStatus"] = design_coverage_info.get("aValidStatus", "stale")
    result_payload["aValidProgressPercent"] = design_coverage_info.get("aValidProgressPercent", 0)
    result_payload["aValidMode"] = design_coverage_info.get("aValidMode", "exact_box_union")
    result_payload["maskBoxCount"] = design_coverage_info.get("maskBoxCount", 0)
    result_payload["coveragePending"] = result_payload.get("coverage_C") is None
    result_payload["confidencePending"] = result_payload.get("confidence") is None
    result_payload["confidenceExact"] = result_payload.get("confidence") is not None
    result_payload["validDomainRatio"] = design_coverage_info.get("validDomainRatio")
    result_payload["valid_domain_ratio"] = design_coverage_info.get("validDomainRatio", result_payload.get("valid_domain_ratio"))
    result_payload["feasibleMaskEnabled"] = design_coverage_info.get("feasibleMaskEnabled", False)
    result_payload["feasibleExpressions"] = design_coverage_info.get("feasibleExpressions", [])
    result_payload["feasibleMaskEvaluationSkipped"] = design_coverage_info.get("feasibleMaskEvaluationSkipped", False)
    result_payload["feasibleMaskWarning"] = design_coverage_info.get("feasibleMaskWarning", "")
    result_payload["coverageWarning"] = design_coverage_info.get("coverageWarning", "")
    result_payload["infeasiblePeerRows"] = design_coverage_info.get("infeasiblePeerRows", 0)
    result_payload["infeasiblePeerBins"] = design_coverage_info.get("infeasiblePeerBins", 0)
    axis_roles = {axis["name"]: axis_role(axis) for axis in canonical_axis_order(selected_goal["axes"])}
    result_payload["axisRoles"] = axis_roles
    result_payload["inputAxisNames"] = design_axis_names
    result_payload["outputAxisNames"] = output_axis_names
    result_payload["specificityAxisNames"] = axis_names
    result_payload["confidenceAxisNames"] = design_axis_names
    result_payload["designOccupiedBins"] = len(design_reference_bin_counts)
    result_payload["designValidBins"] = design_coverage_info.get("validBins")
    result_payload["designTotalBins"] = design_coverage_info.get("rectangularTotalBins")
    result_payload["designMaskedBins"] = design_coverage_info.get("maskedBins")
    result_payload["designCoverageC"] = result_payload.get("coverage_C")
    result_payload["designEquitabilityE"] = result_payload.get("equitability_E")
    result_payload["designConfidence"] = result_payload.get("confidence")
    result_payload["targetIncludedInReference"] = True
    result_payload["target_included_in_reference"] = True
    result_payload["internalDensityMode"] = internal_density_mode
    result_payload["internal_density_mode"] = internal_density_mode
    result_payload["selfContainedReferenceWarning"] = self_contained_warning
    result_payload["self_contained_reference_warning"] = self_contained_warning
    result_payload["externalPeerRecordCount"] = external_peer_record_count
    result_payload["external_peer_record_count"] = external_peer_record_count
    result_payload["peerRecordCount"] = external_peer_record_count
    result_payload["peer_record_count"] = external_peer_record_count
    result_payload["externalPeerObservationCount"] = external_peer_observation_count
    result_payload["external_peer_observation_count"] = external_peer_observation_count
    result_payload["referenceObservationCount"] = reference_observation_count
    result_payload["reference_observation_count"] = reference_observation_count
    result_payload["referenceOccupiedBins"] = reference_occupied_bins
    result_payload["reference_occupied_bins"] = reference_occupied_bins
    result_payload["referenceDensityPolicy"] = "target_included"
    coverage_for_result = {
            **coverage_info,
            "designCoverageInfo": {key: value for key, value in design_coverage_info.items() if key not in {"binCounts", "axisBinCounts"}},
            "externalBinCounts": external_bin_counts,
            "storedPeerBinCounts": external_bin_counts,
            "binCounts": reference_bin_counts,
            "referenceBinCounts": reference_bin_counts,
            "designBinCounts": design_reference_bin_counts,
            "designExternalBinCounts": design_external_bin_counts,
            "designAxisNames": design_axis_names,
            "outputAxisNames": output_axis_names,
            "specificityAxisNames": axis_names,
            "confidenceAxisNames": design_axis_names,
            "designOccupiedBins": len(design_reference_bin_counts),
            "designValidBins": design_coverage_info.get("validBins"),
            "designTotalBins": design_coverage_info.get("rectangularTotalBins"),
            "designMaskedBins": design_coverage_info.get("maskedBins"),
            "designCoverageC": result_payload.get("coverage_C"),
            "designEquitabilityE": result_payload.get("equitability_E"),
            "designConfidence": result_payload.get("confidence"),
            "specificityTotalBins": result_payload.get("specificityTotalBins"),
            "externalPeerRecordCount": external_peer_record_count,
        "peerRecordCount": external_peer_record_count,
        "externalPeerObservationCount": external_peer_observation_count,
        "referenceObservationCount": reference_observation_count,
        "referenceOccupiedBins": reference_occupied_bins,
        "targetIncludedInReference": True,
        "internalDensityMode": internal_density_mode,
        "selfContainedReferenceWarning": self_contained_warning,
        "rowLevelObservationCount": reference_observation_count,
    }
    return {
        "config": config,
        "peerRows": peer_rows,
        "peerClusters": peer_clusters,
        "peerGroup": peer_group,
        "coverageInfo": {
            **coverage_for_result,
            "specificityTotalBins": coverage_info.get("totalBins", result.total_bins),
            "totalBins": design_coverage_info.get("rectangularTotalBins", result.total_bins),
            "validBins": result.valid_bins,
            "maskedBins": result.masked_bins,
            "aValid": design_coverage_info.get("aValid", result.valid_bins),
            "rectangularTotalBins": design_coverage_info.get("rectangularTotalBins"),
            "aValidStatus": design_coverage_info.get("aValidStatus", "stale"),
            "aValidProgressPercent": design_coverage_info.get("aValidProgressPercent", 0),
            "aValidMode": design_coverage_info.get("aValidMode", "exact_box_union"),
            "maskBoxCount": design_coverage_info.get("maskBoxCount", 0),
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
            coverage_info={
                **coverage_for_result,
                "specificityTotalBins": result.total_bins,
                "totalBins": design_coverage_info.get("rectangularTotalBins", result.total_bins),
                "occupiedBins": result.occupied_bins,
            },
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
    target_design_counts, target_design_meta = recompute_design_counts_from_vectors(
        dataset_meta.get("row_level_vectors", []),
        dataset_meta.get("row_level_vector_axis_order", []),
        goal,
        selected_goal,
    )
    saved_cluster = None
    saved_cluster_is_new = False

    try:
        analysis = run_density_analysis(
            goal,
            selected_goal,
            dataset_meta.get("bin_occupancy", {}),
            dataset_meta.get("bin_occupancy_meta", {}),
            cluster_vector,
            target_design_bin_counts=target_design_counts,
            target_design_meta=target_design_meta,
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
            "Density analysis uses target-included reference row-level bin occupancy in the current grid. "
            f"Eligible external density records={stored_count}, external peer row-level observations={coverage_info['rowLevelObservationCount']}. "
            f"Excluded legacy records={coverage_info['coverageLegacyExcludedClusterCount']}, "
            f"axis mismatches={coverage_info['coverageGridSignatureExcludedClusterCount']}. "
            f"This CSV was sanitized into row-level axis vectors and bin occupancy and {saved_text}, but analysis could not run because the target/reference rows did not satisfy the analysis requirements. "
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
    if result_payload.get("selfContainedReferenceWarning"):
        summary.append("특이도는 표적 군집을 포함한 reference density map 기준으로 계산됩니다. 외부 peer record가 없는 경우, 이 값은 과거 실험 대비 이상치가 아니라 업로드된 데이터 내부의 상대적 희소 구간을 의미합니다.")
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
            "peerRecordCount": result_payload.get("peerRecordCount"),
            "externalPeerRecordCount": result_payload.get("externalPeerRecordCount"),
            "externalPeerObservationCount": result_payload.get("externalPeerObservationCount"),
            "referenceObservationCount": result_payload.get("referenceObservationCount"),
            "referenceOccupiedBins": result_payload.get("referenceOccupiedBins"),
            "targetIncludedInReference": True,
            "internalDensityMode": result_payload.get("internalDensityMode"),
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
            "feasibleMaskEnabled": analysis["coverageInfo"].get("feasibleMaskEnabled"),
            "feasibleMaskEvaluationSkipped": analysis["coverageInfo"].get("feasibleMaskEvaluationSkipped"),
            "feasibleMaskWarning": analysis["coverageInfo"].get("feasibleMaskWarning"),
            "coverageWarning": analysis["coverageInfo"].get("coverageWarning"),
            "validBins": analysis["coverageInfo"].get("validBins"),
            "maskedBins": analysis["coverageInfo"].get("maskedBins"),
            "validDomainRatio": analysis["coverageInfo"].get("validDomainRatio"),
            "rowLevelObservationCount": analysis["coverageInfo"]["referenceObservationCount"],
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
                target_design_counts, target_design_meta = recompute_design_counts_from_vectors(
                    dataset_meta.get("row_level_vectors", []),
                    dataset_meta.get("row_level_vector_axis_order", []),
                    goal,
                    selected_goal,
                )
                analysis = run_density_analysis(
                    goal,
                    selected_goal,
                    dataset_meta.get("bin_occupancy", {}),
                    dataset_meta.get("bin_occupancy_meta", {}),
                    cluster_vector,
                    target_design_bin_counts=target_design_counts,
                    target_design_meta=target_design_meta,
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
                            "external_peer_record_count": result_payload.get("externalPeerRecordCount"),
                            "external_peer_observation_count": result_payload.get("externalPeerObservationCount"),
                            "reference_observation_count": result_payload.get("referenceObservationCount"),
                            "target_included_in_reference": result_payload.get("targetIncludedInReference"),
                            "internal_density_mode": result_payload.get("internalDensityMode"),
                            "coverage_eligible_cluster_count": analysis["coverageInfo"]["coverageEligibleClusterCount"],
                            "row_level_observation_count": analysis["coverageInfo"]["referenceObservationCount"],
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


def record_design_density_counts_for_goal(
    cluster: dict[str, Any],
    goal: dict[str, Any],
    selected_goal: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    row_vectors = cluster.get("rowLevelVectors") if isinstance(cluster.get("rowLevelVectors"), list) else []
    source_axis_order = [str(name) for name in (cluster.get("rowLevelVectorAxisOrder") or cluster.get("axisNames") or [])]
    if row_vectors:
        return recompute_design_counts_from_vectors(row_vectors, source_axis_order, goal, selected_goal)
    design_goal = design_goal_for_selected(goal, selected_goal)
    if [axis["name"] for axis in canonical_axis_order(design_goal["axes"])] == [axis["name"] for axis in canonical_axis_order(selected_goal["axes"])]:
        return record_density_counts_for_goal(cluster, selected_goal)
    return {}, {"validMultidimensionalRowCount": 0, "totalRows": 0}


def reevaluate_cluster(cluster_id: str) -> dict[str, Any]:
    cluster = cluster_by_id(cluster_id)
    goal = find_goal(str(cluster["goalId"]))
    selected_goal = goal_subset(goal, [str(name) for name in cluster["axisNames"]])
    target = np.asarray(cluster["values"], dtype=float)
    target_counts, target_meta = record_density_counts_for_goal(cluster, selected_goal)
    target_design_counts, target_design_meta = record_design_density_counts_for_goal(cluster, goal, selected_goal)
    uploaded = normalize_analysis_snapshot(cluster.get("analysisAtUpload"))
    try:
        analysis = run_density_analysis(
            goal,
            selected_goal,
            target_counts,
            target_meta,
            target,
            exclude_cluster_id=str(cluster["id"]),
            target_design_bin_counts=target_design_counts,
            target_design_meta=target_design_meta,
        )
        current = analysis["resultPayload"]
        current_peer_group_size = len(analysis["peerGroup"])
        confidence_delta = nullable_numeric_delta(current.get("confidence"), uploaded.get("confidence"))
        specificity_delta = nullable_numeric_delta(current.get("specificity_score"), uploaded.get("specificityScore"))
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
        target_design_counts, target_design_meta = record_design_density_counts_for_goal(cluster, goal, selected_goal)
        # The target is always added into the reference map by run_density_analysis.
        # Exclude this saved record from the external peer side to avoid counting it twice.
        effective_exclude_cluster_id = str(exclude_cluster_id or cluster.get("id") or "").strip() or None
        analysis = run_density_analysis(
            goal,
            selected_goal,
            target_counts,
            target_meta,
            target,
            exclude_cluster_id=effective_exclude_cluster_id,
            target_design_bin_counts=target_design_counts,
            target_design_meta=target_design_meta,
        )
        return {"ok": True, "result": analysis["resultPayload"], "peerGroupSize": len(analysis["peerGroup"])}
    except ValueError as exc:
        effective_exclude_cluster_id = str(exclude_cluster_id or cluster.get("id") or "").strip() or None
        return {
            "ok": False,
            "error": str(exc),
            "peerGroupSize": len(analysis_peer_rows(goal, [axis["name"] for axis in canonical_axis_order(selected_goal["axes"])], effective_exclude_cluster_id)),
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
            "deltaConfidence": nullable_numeric_delta(without_result.get("confidence"), all_result.get("confidence")),
            "deltaCoverage": nullable_numeric_delta(without_result.get("coverage_C"), all_result.get("coverage_C")),
            "deltaEquitability": nullable_numeric_delta(without_result.get("equitability_E"), all_result.get("equitability_E")),
            "deltaSpecificity": nullable_numeric_delta(without_result.get("specificity_score"), all_result.get("specificity_score")),
            "deltaMeanRarity": nullable_numeric_delta(without_result.get("mean_rarity"), all_result.get("mean_rarity")),
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
            "peer_record_count",
            "external_peer_record_count",
            "external_peer_observation_count",
            "reference_observation_count",
            "reference_occupied_bins",
            "reference_density_policy",
            "target_included_in_reference",
            "internal_density_mode",
            "self_contained_reference_warning",
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
            "feasible_mask_evaluation_skipped",
            "feasible_mask_warning",
            "coverage_warning",
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
                "peer_record_count": result.get("peerRecordCount", result.get("peer_record_count")),
                "external_peer_record_count": result.get("externalPeerRecordCount", result.get("external_peer_record_count")),
                "external_peer_observation_count": result.get("externalPeerObservationCount", result.get("external_peer_observation_count")),
                "reference_observation_count": result.get("referenceObservationCount", result.get("reference_observation_count")),
                "reference_occupied_bins": result.get("referenceOccupiedBins", result.get("reference_occupied_bins")),
                "reference_density_policy": result.get("referenceDensityPolicy", result.get("reference_density_policy")),
                "target_included_in_reference": result.get("targetIncludedInReference", result.get("target_included_in_reference")),
                "internal_density_mode": result.get("internalDensityMode", result.get("internal_density_mode")),
                "self_contained_reference_warning": result.get("selfContainedReferenceWarning", result.get("self_contained_reference_warning")),
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
                "feasible_mask_evaluation_skipped": result.get("feasibleMaskEvaluationSkipped", result.get("feasible_mask_evaluation_skipped")),
                "feasible_mask_warning": result.get("feasibleMaskWarning", result.get("feasible_mask_warning")),
                "coverage_warning": result.get("coverageWarning", result.get("coverage_warning")),
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
                try:
                    selected_goal = goal_subset(goal, names)
                    peer_clusters = load_peer_clusters(str(goal["id"]), names)
                    goal_counts[axis_subset_key(names)] = int(build_global_bin_counts(peer_clusters, selected_goal)["coverageEligibleClusterCount"])
                except ValueError:
                    goal_counts[axis_subset_key(names)] = 0
        counts[goal["id"]] = goal_counts
    return counts


def goal_bin_preview(goal: dict[str, Any]) -> dict[str, Any]:
    canonical_axes = canonical_axis_order(goal["axes"])
    full_axis_names = [axis["name"] for axis in canonical_axes]
    design_axes = input_axes_for_goal(goal)
    design_axis_names = [axis["name"] for axis in design_axes]
    design_goal = goal_subset(goal, design_axis_names)
    stored_clusters = load_peer_clusters(str(goal["id"]), full_axis_names)
    coverage_info = build_global_bin_counts(stored_clusters, design_goal)
    axis_previews = []
    total_bins = 1
    for axis_index, axis in enumerate(design_axes):
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
    valid_bins = int(coverage_info.get("validBins")) if coverage_info.get("validBins") is not None else None
    masked_bins = int(coverage_info.get("maskedBins")) if coverage_info.get("maskedBins") is not None else None
    warnings = []
    if total_bins > 100000:
        warnings.append("Resolution이 현재 데이터 수에 비해 과도하게 세밀할 수 있음.")
    if coverage_info["coverageLegacyExcludedClusterCount"]:
        warnings.append("일부 기존 record는 row-level vectors가 없어 coverage 계산에서 제외됩니다. 정확한 coverage를 원하면 CSV를 다시 업로드하세요.")
    if coverage_info["coverageGridSignatureExcludedClusterCount"]:
        warnings.append("Axis 구성이 맞지 않는 record는 density 계산에서 제외됩니다.")
    if coverage_info.get("feasibleMaskEnabled"):
        warnings.append("Feasible Domain은 2D Projection Mask box들의 조합으로 정의되며, a_valid는 mask box 합집합으로 exact 계산됩니다.")
    return {
        "basis": coverage_info["coverageBasis"],
        "axisBins": axis_previews,
        "totalBins": total_bins,
        "validBins": valid_bins,
        "aValid": coverage_info.get("aValid") if coverage_info.get("aValid") is not None else valid_bins,
        "rectangularTotalBins": coverage_info.get("rectangularTotalBins", total_bins),
        "designTotalBins": coverage_info.get("rectangularTotalBins", total_bins),
        "designAxes": design_axis_names,
        "outputAxisNames": [axis["name"] for axis in canonical_axes if axis_role(axis) == "output"],
        "maskedBins": masked_bins,
        "maskBoxCount": coverage_info.get("maskBoxCount", 0),
        "validDomainRatio": coverage_info.get("validDomainRatio"),
        "feasibleMaskEnabled": coverage_info.get("feasibleMaskEnabled", False),
        "feasibleExpressions": coverage_info.get("feasibleExpressions", []),
        "feasibleMaskEvaluationSkipped": False,
        "feasibleMaskWarning": coverage_info.get("feasibleMaskWarning", ""),
        "coverageWarning": coverage_info.get("coverageWarning", ""),
        "aValidStatus": coverage_info.get("aValidStatus", "stale"),
        "aValidProgressPercent": coverage_info.get("aValidProgressPercent", 0),
        "aValidMode": coverage_info.get("aValidMode", "exact_box_union"),
        "occupiedBins": occupied_bins,
        "estimatedCoverage": occupied_bins / valid_bins if valid_bins else None,
        "coveragePending": valid_bins is None,
        "confidencePending": valid_bins is None,
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
        "feasibleDomainRules": list(selected_goal.get("feasibleDomainRules") or []),
        "legacyAdvancedExpressions": normalize_feasible_expressions(selected_goal.get("legacyAdvancedExpressions")),
        "generatedFeasibleExpressions": feasible_expressions_for_goal(selected_goal),
        "feasibleDomainExpressions": feasible_expressions_for_goal(selected_goal),
        "aValidCache": dict(selected_goal.get("aValidCache") or {}) if isinstance(selected_goal.get("aValidCache"), dict) else {},
        "aValidStatus": selected_goal.get("aValidStatus"),
        "aValidProgressPercent": selected_goal.get("aValidProgressPercent"),
        "aValidMode": selected_goal.get("aValidMode"),
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
    preview_total_bins = rectangular_total_bins_for_axes(preview_goal["axes"])
    if preview_total_bins > MAX_GRID_PREVIEW_BINS:
        return skipped_grid_preview_payload(preview_goal, preview_total_bins)
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
    external_bin_counts = {str(key): int(value) for key, value in (coverage_info.get("binCounts") or {}).items()}
    reference_bin_counts = dict(external_bin_counts)
    merge_count_maps(reference_bin_counts, target_recomputed["bin_occupancy"])
    preview_coverage_info = {
        **coverage_info,
        "externalBinCounts": external_bin_counts,
        "storedPeerBinCounts": external_bin_counts,
        "binCounts": reference_bin_counts,
        "referenceBinCounts": reference_bin_counts,
        "externalPeerRecordCount": int(coverage_info.get("coverageEligibleClusterCount") or 0),
        "externalPeerObservationCount": count_map_total(external_bin_counts),
        "referenceObservationCount": count_map_total(reference_bin_counts),
        "referenceOccupiedBins": len(reference_bin_counts),
        "targetIncludedInReference": True,
        "internalDensityMode": count_map_total(external_bin_counts) == 0,
    }
    metrics = density_preview_metrics_from_coverage(preview_goal, preview_coverage_info)
    projection_explorer = build_projection_explorer(preview_goal, preview_coverage_info, target_recomputed["row_bin_tuples"])
    projection_explorer["gridPreviewMetrics"] = metrics

    result_payload: dict[str, Any] | None = None
    if metrics["peerValidRows"] > 0 and target_recomputed["bin_occupancy_meta"]["validMultidimensionalRowCount"] > 0:
        analyzer = DensityGridAnalyzer(experiment_config_from_goal(preview_goal))
        analyzer.set_peer_bin_counts(reference_bin_counts)
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
        "coverageInfo": {key: value for key, value in preview_coverage_info.items() if key not in {"binCounts", "axisBinCounts", "referenceBinCounts", "externalBinCounts", "storedPeerBinCounts"}},
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


def a_valid_status_for_goal(goal: dict[str, Any]) -> dict[str, Any]:
    goal_id = str(goal.get("id") or "")
    if goal_id in A_VALID_JOBS:
        job = dict(A_VALID_JOBS[goal_id])
        job.setdefault("goalId", goal_id)
        return job
    mask_info = feasible_mask_info_for_goal(goal)
    return {
        "goalId": goal_id,
        "aValidStatus": mask_info.get("aValidStatus", "stale"),
        "aValidProgressPercent": mask_info.get("aValidProgressPercent", 0),
        "message": "a_valid ready." if mask_info.get("aValidStatus") == "ready" else "Mask or axis settings changed. Recompute a_valid.",
        "aValid": mask_info.get("aValid"),
        "maskedBins": mask_info.get("maskedBins"),
        "rectangularTotalBins": mask_info.get("rectangularTotalBins"),
        "aValidMode": mask_info.get("aValidMode", "exact_box_union"),
        "maskSignature": mask_info.get("maskSignature"),
        "computedAt": mask_info.get("aValidComputedAt"),
    }


def compute_a_valid_for_goal_request(goal_id: str) -> dict[str, Any]:
    goals = normalize_goals_for_display(load_goal_store())
    goal = next((item for item in goals if str(item.get("id")) == goal_id), None)
    if goal is None:
        raise ValueError("Selected Experiment Goal does not exist.")

    started = datetime.now(timezone.utc)
    A_VALID_JOBS[goal_id] = {
        "goalId": goal_id,
        "aValidStatus": "calculating",
        "status": "calculating",
        "aValidProgressPercent": 0,
        "percent": 0,
        "message": "a_valid calculation started.",
        "rectangularTotalBins": compute_rectangular_total_bins(input_axes_for_goal(goal)),
    }

    def progress(percent: int, message: str) -> None:
        bounded = max(0, min(100, int(percent)))
        A_VALID_JOBS[goal_id].update(
            {
                "aValidStatus": "calculating",
                "status": "calculating",
                "aValidProgressPercent": bounded,
                "percent": bounded,
                "message": message,
            }
        )

    try:
        result = compute_a_valid_for_rules(input_axes_for_goal(goal), goal.get("feasibleDomainRules") or [], progress_callback=progress)
        computed_at = datetime.now(timezone.utc)
        duration_ms = int((computed_at - started).total_seconds() * 1000)
        cache = {
            "maskSignature": result["maskSignature"],
            "rectangularTotalBins": result["rectangularTotalBins"],
            "maskedBins": result["maskedBins"],
            "aValid": result["aValid"],
            "aValidMode": "exact_box_union",
            "computedAt": computed_at.isoformat(),
            "durationMs": duration_ms,
        }
        updated_goal = dict(goal)
        updated_goal["aValidCache"] = cache
        updated_goal = validate_goal(updated_goal)
        updated_goals = [updated_goal if str(item.get("id")) == goal_id else item for item in goals]
        save_goal_store(updated_goals)
        normalized_goals = normalize_goals_for_display(updated_goals)
        payload = {
            "ok": True,
            "goalId": goal_id,
            "aValidStatus": "ready",
            "status": "ready",
            "aValidProgressPercent": 100,
            "percent": 100,
            "rectangularTotalBins": cache["rectangularTotalBins"],
            "maskedBins": cache["maskedBins"],
            "aValid": cache["aValid"],
            "validBins": cache["aValid"],
            "aValidMode": cache["aValidMode"],
            "maskSignature": cache["maskSignature"],
            "computedAt": cache["computedAt"],
            "durationMs": duration_ms,
            "savedGoal": updated_goal,
            "goals": normalized_goals,
            "goalBinPreview": {goal["id"]: goal_bin_preview(goal) for goal in normalized_goals},
        }
        A_VALID_JOBS[goal_id] = {**payload, "message": "a_valid ready."}
        return payload
    except Exception as exc:
        A_VALID_JOBS[goal_id] = {
            "goalId": goal_id,
            "aValidStatus": "failed",
            "status": "failed",
            "aValidProgressPercent": 100,
            "percent": 100,
            "message": str(exc),
        }
        raise


def build_storage_payload(cluster_count: int = 0, goal_count: int | None = None) -> dict[str, Any]:
    status = storage_status()
    return {
        **status,
        "backend": status["storageBackend"],
        "goalStoreFile": status["goalStorePath"],
        "clusterStoreFile": status["clusterStorePath"],
        "goalCount": goal_count,
        "clusterCount": cluster_count,
        "recordCount": cluster_count,
        "loadedAt": datetime.now(timezone.utc).isoformat(),
        "serverInstance": SERVER_INSTANCE_ID,
        "serverStartedAt": SERVER_STARTED_AT,
        "savedItems": [
            "Experiment Goal 설정",
            "selected-axis row-level sanitized numeric vectors",
            "비식별 mean vector 호환 필드",
            "row-level bin count summary",
        ],
        "unsavedItems": ["원본 CSV 파일", "파일명", "비매핑 column", "개인정보 column"],
    }


def build_bootstrap_payload(admin_allowed: bool) -> dict[str, Any]:
    goals = normalize_goals_for_display(load_goal_store())
    peer_counts = bootstrap_peer_counts()
    peer_subset_counts = bootstrap_peer_subset_counts()
    clusters = list_cluster_summaries()
    admin_auth_required = bool(os.environ.get("ADMIN_TOKEN")) and not admin_allowed
    return {
        "adminAllowed": admin_allowed,
        "adminAuthRequired": admin_auth_required,
        "goals": goals,
        "clusters": clusters,
        "peerCounts": peer_counts,
        "peerSubsetCounts": peer_subset_counts,
        "goalBinPreview": {goal["id"]: goal_bin_preview(goal) for goal in goals},
        "acceptedUploadTypes": [".csv", ".tsv", ".txt"],
        "storage": build_storage_payload(cluster_count=len(clusters), goal_count=len(goals)),
        "domainDefinitions": {
            "cluster": "CSV 파일 하나는 저장 record로 처리됩니다. 저장/삭제는 record 단위지만, density calculation은 record 내부 row-level sanitized axis vectors를 현재 grid로 다시 binning해서 수행합니다.",
            "coverage": "Coverage and Equitability are calculated from row-level vectors re-binned into the current density grid.",
            "K_m": "K_m is currently reused as K_density for Observation Support: S = peer row-level observations / (peer row-level observations + K_density).",
        },
    }


def build_shell_bootstrap_payload(admin_allowed: bool) -> dict[str, Any]:
    admin_auth_required = bool(os.environ.get("ADMIN_TOKEN")) and not admin_allowed
    return {
        "adminAllowed": admin_allowed,
        "adminAuthRequired": admin_auth_required,
        "goals": [],
        "clusters": [],
        "peerCounts": {},
        "peerSubsetCounts": {},
        "goalBinPreview": {},
        "acceptedUploadTypes": [".csv", ".tsv", ".txt"],
        "deferBootstrap": True,
        "storage": build_storage_payload(cluster_count=0, goal_count=None),
        "domainDefinitions": {
            "cluster": "CSV 파일 하나는 저장 record로 처리됩니다. 저장/삭제는 record 단위지만, density calculation은 record 내부 row-level sanitized axis vectors를 현재 grid로 다시 binning해서 수행합니다.",
            "coverage": "Coverage and Equitability are calculated from row-level vectors re-binned into the current density grid.",
            "K_m": "K_m is currently reused as K_density for Observation Support.",
        },
    }


def render_page(admin_allowed: bool) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    bootstrap = json.dumps(build_shell_bootstrap_payload(admin_allowed), ensure_ascii=False)
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
        if parsed.path == "/api/storage-status":
            self._send_json(build_storage_payload(cluster_count=len(list_cluster_summaries()), goal_count=len(load_goal_store())))
            return
        if parsed.path == "/healthz":
            self._send_json({"ok": True})
            return
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 5 and parts[:3] == ["api", "admin", "goals"] and parts[4] == "a-valid-status":
            if not self._admin_allowed():
                self._send_json({"error": "Admin Token이 필요합니다."}, status=HTTPStatus.FORBIDDEN)
                return
            goal_id = unquote(parts[3])
            goal = find_goal(goal_id)
            self._send_json(a_valid_status_for_goal(goal))
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
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 5 and parts[:3] == ["api", "admin", "goals"] and parts[4] == "compute-a-valid":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin Token이 필요합니다."}, status=HTTPStatus.FORBIDDEN)
                    return
                self._send_json(compute_a_valid_for_goal_request(unquote(parts[3])))
                return
            if parsed.path == "/api/admin/goals/compute-a-valid":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin Token이 필요합니다."}, status=HTTPStatus.FORBIDDEN)
                    return
                goal_id = str(payload.get("id") or payload.get("goalId") or "").strip()
                if not goal_id:
                    validated_goal = validate_goal(payload)
                    mask_info = feasible_mask_info_for_goal(validated_goal)
                    self._send_json({"ok": True, "goal": validated_goal, "maskInfo": mask_info})
                    return
                self._send_json(compute_a_valid_for_goal_request(goal_id))
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
                mask_info = feasible_mask_info_for_goal(validated_goal)
                self._send_json({"ok": True, "goal": validated_goal, "maskInfo": mask_info_for_display(mask_info)})
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
            self._send_json(api_error_payload(exc), status=HTTPStatus.BAD_REQUEST)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(data)))
        try:
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        size = len(data)
        if size >= 1_000_000:
            print(f"Large HTML response: {size} bytes. Consider lazy loading.")
        else:
            print(f"HTML response size: {size} bytes.")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(size))
        try:
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    # TODO: This app currently uses BaseHTTPRequestHandler rather than a WSGI app.
    # Add a WSGI adapter before switching a Render start command to gunicorn.
    init_database()
    status = storage_status()
    try:
        loaded_goal_count = len(load_goal_store())
    except Exception as exc:
        loaded_goal_count = -1
        print(f"Storage goal load check failed: {exc}")
    try:
        loaded_cluster_count = len(list_cluster_summaries())
    except Exception as exc:
        loaded_cluster_count = -1
        print(f"Storage record load check failed: {exc}")
    print(
        "Storage status: "
        f"backend={status.get('storageBackend')} "
        f"storeDir={status.get('storeDir')} "
        f"goalStorePath={status.get('goalStorePath')} "
        f"clusterStorePath={status.get('clusterStorePath')} "
        f"goals={loaded_goal_count} "
        f"records={loaded_cluster_count}"
    )
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

