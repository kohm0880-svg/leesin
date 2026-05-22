from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable


GUI_RULE_OPERATORS = {">", ">=", "<", "<=", "==", "!="}


def _axis_key(name: Any) -> str:
    return str(name or "").strip().lower()


def canonical_axes(axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(axis) for axis in axes), key=lambda axis: _axis_key(axis.get("name")))


def axis_total_bins(axis: dict[str, Any]) -> int:
    domain_min = float(axis["domainMin"])
    domain_max = float(axis["domainMax"])
    resolution = float(axis["resolution"])
    return max(1, int(math.ceil((domain_max - domain_min) / resolution)))


def axis_bin_counts(axes: list[dict[str, Any]]) -> dict[str, int]:
    return {str(axis["name"]): axis_total_bins(axis) for axis in canonical_axes(axes)}


def compute_rectangular_total_bins(axes: list[dict[str, Any]]) -> int:
    counts = [axis_total_bins(axis) for axis in canonical_axes(axes)]
    return int(math.prod(counts)) if counts else 0


def _finite_float(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite.")
    return numeric


def _format_number(value: Any) -> str:
    return format(_finite_float(value, "Rule value"), ".12g")


def _value_range_to_bin_range(axis: dict[str, Any], min_value: float, max_value: float) -> tuple[int, int]:
    """Return half-open bin index range whose bin centers are inside [min_value, max_value]."""
    if min_value > max_value:
        return (0, 0)
    domain_min = float(axis["domainMin"])
    resolution = float(axis["resolution"])
    total = axis_total_bins(axis)
    if math.isinf(min_value) and min_value < 0:
        start = 0
    else:
        start = math.ceil(((min_value - domain_min) / resolution) - 0.5 - 1e-12)
    if math.isinf(max_value) and max_value > 0:
        end = total
    else:
        end = math.floor(((max_value - domain_min) / resolution) - 0.5 + 1e-12) + 1
    return (max(0, min(total, int(start))), max(0, min(total, int(end))))


def _condition_to_bin_ranges(axis: dict[str, Any], operator: str, value: float) -> list[tuple[int, int]]:
    total = axis_total_bins(axis)
    if operator == ">":
        return [_value_range_to_bin_range(axis, value + 1e-12, float("inf"))]
    if operator == ">=":
        return [_value_range_to_bin_range(axis, value, float("inf"))]
    if operator == "<":
        return [_value_range_to_bin_range(axis, float("-inf"), value - 1e-12)]
    if operator == "<=":
        return [_value_range_to_bin_range(axis, float("-inf"), value)]
    if operator == "==":
        return [_value_range_to_bin_range(axis, value, value)]
    if operator == "!=":
        equal_ranges = _condition_to_bin_ranges(axis, "==", value)
        if not equal_ranges:
            return [(0, total)]
        eq_start, eq_end = equal_ranges[0]
        ranges = []
        if eq_start > 0:
            ranges.append((0, eq_start))
        if eq_end < total:
            ranges.append((eq_end, total))
        return ranges
    raise ValueError(f"Unsupported operator '{operator}'.")


def _axis_lookup(axes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(axis["name"]): axis for axis in canonical_axes(axes)}


def _non_empty_box(ranges: dict[str, tuple[int, int]]) -> bool:
    return all(int(end) > int(start) for start, end in ranges.values())


def focused_2d_rule_to_mask_boxes(rule: dict[str, Any], axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    axis_by_name = _axis_lookup(axes)
    allowed = set(axis_by_name)
    gui_spec = rule.get("guiSpec") if isinstance(rule.get("guiSpec"), dict) else {}
    if str(gui_spec.get("type") or "") != "focused_2d_mask":
        raise ValueError("Focused mask rule must have guiSpec.type focused_2d_mask.")
    x_axis = str(gui_spec.get("xAxis") or "").strip()
    y_axis = str(gui_spec.get("yAxis") or "").strip()
    if x_axis not in allowed or y_axis not in allowed:
        raise ValueError("Focused mask axes must exist in the goal.")
    if x_axis == y_axis:
        raise ValueError("Focused 2D Mask requires different X and Y axes.")
    x_min = _finite_float(gui_spec.get("xMin"), "xMin")
    x_max = _finite_float(gui_spec.get("xMax"), "xMax")
    y_min = _finite_float(gui_spec.get("yMin"), "yMin")
    y_max = _finite_float(gui_spec.get("yMax"), "yMax")
    if x_min > x_max or y_min > y_max:
        raise ValueError("Focused mask ranges must have min <= max.")
    ranges: dict[str, tuple[int, int]] = {
        x_axis: _value_range_to_bin_range(axis_by_name[x_axis], x_min, x_max),
        y_axis: _value_range_to_bin_range(axis_by_name[y_axis], y_min, y_max),
    }
    scope = gui_spec.get("scope") if isinstance(gui_spec.get("scope"), dict) else {}
    for raw_axis, raw_scope in scope.items():
        axis_name = str(raw_axis or "").strip()
        if axis_name not in allowed:
            raise ValueError(f"Unknown focused mask scope axis '{axis_name}'.")
        if axis_name in {x_axis, y_axis}:
            continue
        scope_spec = raw_scope if isinstance(raw_scope, dict) else {}
        mode = str(scope_spec.get("mode") or "all").strip().lower()
        if mode == "all":
            continue
        if mode != "range":
            raise ValueError(f"Unsupported focused mask scope mode '{mode}'.")
        min_value = _finite_float(scope_spec.get("min"), f"{axis_name} min")
        max_value = _finite_float(scope_spec.get("max"), f"{axis_name} max")
        if min_value > max_value:
            raise ValueError(f"Focused mask scope range for '{axis_name}' is invalid.")
        ranges[axis_name] = _value_range_to_bin_range(axis_by_name[axis_name], min_value, max_value)
    return [{"ranges": ranges}] if _non_empty_box(ranges) else []


def conditional_rule_to_mask_boxes(rule: dict[str, Any], axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    axis_by_name = _axis_lookup(axes)
    allowed = set(axis_by_name)
    gui_spec = rule.get("guiSpec") if isinstance(rule.get("guiSpec"), dict) else {}
    if str(gui_spec.get("type") or "conditional_range") != "conditional_range":
        raise ValueError("Conditional rule must have guiSpec.type conditional_range.")
    if_spec = gui_spec.get("if") if isinstance(gui_spec.get("if"), dict) else {}
    then_spec = gui_spec.get("then") if isinstance(gui_spec.get("then"), dict) else {}
    if_axis = str(if_spec.get("axis") or "").strip()
    then_axis = str(then_spec.get("axis") or "").strip()
    operator = str(if_spec.get("op") or "").strip()
    if if_axis not in allowed or then_axis not in allowed:
        raise ValueError("Conditional rule axes must exist in the goal.")
    if operator not in GUI_RULE_OPERATORS:
        raise ValueError(f"Unsupported conditional operator '{operator}'.")
    if_value = _finite_float(if_spec.get("value"), "IF value")
    min_value = _finite_float(then_spec.get("min"), "THEN min")
    max_value = _finite_float(then_spec.get("max"), "THEN max")
    if min_value > max_value:
        raise ValueError("THEN minimum must be less than or equal to maximum.")
    if_ranges = _condition_to_bin_ranges(axis_by_name[if_axis], operator, if_value)
    then_outside_ranges = (
        _condition_to_bin_ranges(axis_by_name[then_axis], "<", min_value)
        + _condition_to_bin_ranges(axis_by_name[then_axis], ">", max_value)
    )
    boxes: list[dict[str, Any]] = []
    for if_range in if_ranges:
        for then_range in then_outside_ranges:
            ranges = {if_axis: if_range, then_axis: then_range}
            if _non_empty_box(ranges):
                boxes.append({"ranges": ranges})
    return boxes


def compile_rules_to_mask_boxes(rules: list[dict[str, Any]] | None, axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for rule in rules or []:
        if not isinstance(rule, dict) or not rule.get("enabled", True):
            continue
        source_type = str(rule.get("sourceType") or "")
        gui_spec = rule.get("guiSpec") if isinstance(rule.get("guiSpec"), dict) else {}
        rule_type = str(gui_spec.get("type") or "")
        if source_type == "focused_2d_mask" or rule_type == "focused_2d_mask":
            boxes.extend(focused_2d_rule_to_mask_boxes(rule, axes))
        elif source_type == "gui_conditional" or rule_type == "conditional_range":
            boxes.extend(conditional_rule_to_mask_boxes(rule, axes))
    return boxes


def _full_range_for_axis(axis_name: str, axis_counts: dict[str, int], ranges: dict[str, Any]) -> tuple[int, int]:
    raw_range = ranges.get(axis_name)
    if raw_range is None:
        return (0, int(axis_counts[axis_name]))
    start, end = raw_range
    return (max(0, min(int(axis_counts[axis_name]), int(start))), max(0, min(int(axis_counts[axis_name]), int(end))))


def _box_ranges_list(box: dict[str, Any], axis_names: list[str], axis_counts: dict[str, int]) -> list[tuple[int, int]]:
    ranges = box.get("ranges") if isinstance(box.get("ranges"), dict) else {}
    return [_full_range_for_axis(axis_name, axis_counts, ranges) for axis_name in axis_names]


def _normalize_boxes(boxes: list[dict[str, Any]], axes: list[dict[str, Any]]) -> list[list[tuple[int, int]]]:
    ordered_axes = canonical_axes(axes)
    axis_names = [str(axis["name"]) for axis in ordered_axes]
    counts = axis_bin_counts(ordered_axes)
    normalized: list[list[tuple[int, int]]] = []
    for box in boxes:
        ranges = _box_ranges_list(box, axis_names, counts)
        if all(end > start for start, end in ranges):
            normalized.append(ranges)
    return normalized


def union_volume(
    boxes: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    progress_callback: Callable[[float, str], None] | None = None,
) -> int:
    normalized = _normalize_boxes(boxes, axes)
    dimensions = len(canonical_axes(axes))
    if not normalized or dimensions <= 0:
        return 0

    def recurse(active_boxes: list[list[tuple[int, int]]], dimension: int) -> int:
        if not active_boxes:
            return 0
        if dimension >= dimensions:
            return 1
        cuts = sorted({point for box in active_boxes for point in box[dimension]})
        if len(cuts) < 2:
            return 0
        total = 0
        intervals = list(zip(cuts, cuts[1:]))
        for index, (start, end) in enumerate(intervals):
            if end <= start:
                continue
            slab_boxes = [
                box
                for box in active_boxes
                if box[dimension][0] <= start and box[dimension][1] >= end
            ]
            total += (end - start) * recurse(slab_boxes, dimension + 1)
            if progress_callback and dimension == 0 and intervals:
                progress_callback(25.0 + 70.0 * ((index + 1) / len(intervals)), "box union counting")
        return int(total)

    return int(recurse(normalized, 0))


def box_contains_tuple(
    box: dict[str, Any],
    axis_order: list[str],
    bin_tuple: list[int] | tuple[int, ...],
    axis_counts: dict[str, int] | None = None,
) -> bool:
    ranges = box.get("ranges") if isinstance(box.get("ranges"), dict) else {}
    for index, axis_name in enumerate(axis_order):
        if axis_name in ranges:
            start, end = ranges[axis_name]
        elif axis_counts is not None and axis_name in axis_counts:
            start, end = 0, int(axis_counts[axis_name])
        else:
            continue
        value = int(bin_tuple[index])
        if value < int(start) or value >= int(end):
            return False
    return True


def bin_tuple_masked_by_boxes(
    bin_tuple: list[int] | tuple[int, ...],
    axis_order: list[str],
    boxes: list[dict[str, Any]],
    axis_counts: dict[str, int] | None = None,
) -> bool:
    return any(box_contains_tuple(box, axis_order, bin_tuple, axis_counts) for box in boxes)


def mask_signature(axes: list[dict[str, Any]], boxes: list[dict[str, Any]]) -> str:
    payload = {
        "axes": [
            {
                "name": str(axis["name"]),
                "domainMin": float(axis["domainMin"]),
                "domainMax": float(axis["domainMax"]),
                "resolution": float(axis["resolution"]),
            }
            for axis in canonical_axes(axes)
        ],
        "boxes": boxes,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_a_valid_for_rules(
    axes: list[dict[str, Any]],
    rules: list[dict[str, Any]] | None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    ordered_axes = canonical_axes(axes)
    total_bins = compute_rectangular_total_bins(ordered_axes)
    if progress_callback:
        progress_callback(10.0, "rules normalize")
    boxes = compile_rules_to_mask_boxes(rules or [], ordered_axes)
    if progress_callback:
        progress_callback(25.0, "compile to boxes")
    masked_bins = union_volume(boxes, ordered_axes, progress_callback=progress_callback)
    masked_bins = max(0, min(int(total_bins), int(masked_bins)))
    a_valid = max(0, int(total_bins) - masked_bins)
    if progress_callback:
        progress_callback(100.0, "ready")
    return {
        "totalBins": int(total_bins),
        "rectangularTotalBins": int(total_bins),
        "validBins": int(a_valid),
        "aValid": int(a_valid),
        "maskedBins": int(masked_bins),
        "maskBoxes": boxes,
        "maskBoxCount": int(len(boxes)),
        "feasibleMaskEnabled": bool(boxes),
        "validDomainRatio": float(a_valid / total_bins) if total_bins else 0.0,
        "aValidStatus": "ready",
        "aValidProgressPercent": 100,
        "aValidMode": "exact_box_union",
        "coverageDenominator": int(a_valid),
        "maskSignature": mask_signature(ordered_axes, boxes),
        "feasibleMaskEvaluationSkipped": False,
        "feasibleMaskWarning": "",
        "coverageWarning": "",
    }
