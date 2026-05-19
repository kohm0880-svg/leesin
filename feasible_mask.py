from __future__ import annotations

import ast
import hashlib
import json
import math
from typing import Any

import numpy as np


MAX_MASK_EVAL_BINS = 2_000_000


def _safe_min(*args: Any) -> Any:
    if not args:
        raise ValueError("min() requires at least one argument.")
    arrays = np.broadcast_arrays(*[np.asarray(arg) for arg in args])
    return np.minimum.reduce(np.asarray(arrays))


def _safe_max(*args: Any) -> Any:
    if not args:
        raise ValueError("max() requires at least one argument.")
    arrays = np.broadcast_arrays(*[np.asarray(arg) for arg in args])
    return np.maximum.reduce(np.asarray(arrays))


SAFE_FUNCTIONS = {
    "abs": np.abs,
    "min": _safe_min,
    "max": _safe_max,
    "sqrt": np.sqrt,
    "log": np.log,
    "exp": np.exp,
}

BIN_OPS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
    ast.Mod: np.mod,
}

UNARY_OPS = {
    ast.UAdd: lambda value: value,
    ast.USub: np.negative,
    ast.Not: np.logical_not,
}

COMPARE_OPS = {
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}

GUI_RULE_OPERATORS = {">", ">=", "<", "<=", "==", "!="}


def normalize_feasible_expressions(expressions: Any) -> list[str]:
    if not isinstance(expressions, list):
        return []
    normalized: list[str] = []
    for raw_expression in expressions:
        expression = str(raw_expression or "").strip()
        if expression:
            normalized.append(expression)
    return normalized


def _format_number(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Numeric rule value is invalid: {value}") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"Numeric rule value must be finite: {value}")
    return format(numeric, ".12g")


def compile_gui_feasible_rule(rule: dict[str, Any], allowed_axes: list[str]) -> str:
    allowed = _allowed_axis_set(allowed_axes)
    gui_spec = rule.get("guiSpec") if isinstance(rule.get("guiSpec"), dict) else {}
    if str(gui_spec.get("type") or "conditional_range") != "conditional_range":
        raise ValueError("Only conditional_range feasible GUI rules are supported.")
    if_spec = gui_spec.get("if") if isinstance(gui_spec.get("if"), dict) else {}
    then_spec = gui_spec.get("then") if isinstance(gui_spec.get("then"), dict) else {}
    if_axis = str(if_spec.get("axis") or "").strip()
    then_axis = str(then_spec.get("axis") or "").strip()
    operator = str(if_spec.get("op") or "").strip()
    if if_axis not in allowed:
        raise ValueError(f"Unknown IF axis '{if_axis}' in feasible GUI rule.")
    if then_axis not in allowed:
        raise ValueError(f"Unknown THEN axis '{then_axis}' in feasible GUI rule.")
    if operator not in GUI_RULE_OPERATORS:
        raise ValueError(f"Unsupported IF operator '{operator}' in feasible GUI rule.")
    if_value = _format_number(if_spec.get("value"))
    min_value = _format_number(then_spec.get("min"))
    max_value = _format_number(then_spec.get("max"))
    if float(min_value) > float(max_value):
        raise ValueError("THEN minimum must be less than or equal to maximum.")
    expression = f"not ({if_axis} {operator} {if_value} and ({then_axis} < {min_value} or {then_axis} > {max_value}))"
    validate_feasible_expression(expression, allowed_axes)
    return expression


def _rule_id_for_expression(expression: str) -> str:
    digest = hashlib.sha1(expression.encode("utf-8")).hexdigest()[:12]
    return f"rule_{digest}"


def normalize_feasible_rules(rules: Any, allowed_axes: list[str]) -> list[dict[str, Any]]:
    if not isinstance(rules, list):
        return []
    normalized: list[dict[str, Any]] = []
    for raw_rule in rules:
        if not isinstance(raw_rule, dict):
            continue
        source_type = str(raw_rule.get("sourceType") or "gui_conditional")
        editable_mode = str(raw_rule.get("editableMode") or "gui")
        if source_type != "gui_conditional" or editable_mode != "gui":
            continue
        expression = compile_gui_feasible_rule(raw_rule, allowed_axes)
        gui_spec = raw_rule.get("guiSpec") if isinstance(raw_rule.get("guiSpec"), dict) else {}
        if_spec = gui_spec.get("if") if isinstance(gui_spec.get("if"), dict) else {}
        then_spec = gui_spec.get("then") if isinstance(gui_spec.get("then"), dict) else {}
        enabled = bool(raw_rule.get("enabled", True))
        normalized.append(
            {
                "id": str(raw_rule.get("id") or _rule_id_for_expression(expression)),
                "sourceType": "gui_conditional",
                "editableMode": "gui",
                "guiSpec": {
                    "type": "conditional_range",
                    "if": {
                        "axis": str(if_spec.get("axis") or "").strip(),
                        "op": str(if_spec.get("op") or "").strip(),
                        "value": float(if_spec.get("value")),
                    },
                    "then": {
                        "axis": str(then_spec.get("axis") or "").strip(),
                        "min": float(then_spec.get("min")),
                        "max": float(then_spec.get("max")),
                    },
                },
                "expression": expression,
                "enabled": enabled,
            }
        )
    return normalized


def _allowed_axis_set(allowed_axes: list[str]) -> set[str]:
    return {str(axis).strip() for axis in allowed_axes if str(axis).strip()}


def expression_axis_names(expression: str) -> set[str]:
    try:
        tree = ast.parse(str(expression), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid feasible expression syntax: {expression}") from exc
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in SAFE_FUNCTIONS
    }


def _validate_node(node: ast.AST, allowed_axes: set[str]) -> None:
    if isinstance(node, ast.Expression):
        _validate_node(node.body, allowed_axes)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or isinstance(node.value, (int, float)):
            return
        raise ValueError("Only numeric constants are allowed in feasible expressions.")
    if isinstance(node, ast.Name):
        if node.id.startswith("__"):
            raise ValueError(f"Unsafe name '{node.id}' is not allowed.")
        if node.id in allowed_axes:
            return
        raise ValueError(f"Unknown axis name '{node.id}' in feasible expression.")
    if isinstance(node, ast.BinOp):
        if type(node.op) not in BIN_OPS:
            raise ValueError("Unsupported arithmetic operator in feasible expression.")
        _validate_node(node.left, allowed_axes)
        _validate_node(node.right, allowed_axes)
        return
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in UNARY_OPS:
            raise ValueError("Unsupported unary operator in feasible expression.")
        _validate_node(node.operand, allowed_axes)
        return
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError("Unsupported boolean operator in feasible expression.")
        for value in node.values:
            _validate_node(value, allowed_axes)
        return
    if isinstance(node, ast.Compare):
        _validate_node(node.left, allowed_axes)
        for comparator in node.comparators:
            _validate_node(comparator, allowed_axes)
        for operator in node.ops:
            if type(operator) not in COMPARE_OPS:
                raise ValueError("Unsupported comparison operator in feasible expression.")
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
            raise ValueError("Only safe numeric helper functions are allowed in feasible expressions.")
        if node.keywords:
            raise ValueError("Function keyword arguments are not allowed in feasible expressions.")
        for argument in node.args:
            _validate_node(argument, allowed_axes)
        return
    raise ValueError(f"Unsupported syntax '{type(node).__name__}' in feasible expression.")


def validate_feasible_expression(expression: str, allowed_axes: list[str]) -> None:
    expression = str(expression or "").strip()
    if not expression:
        raise ValueError("Feasible expression cannot be empty.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid feasible expression syntax: {expression}") from exc
    _validate_node(tree, _allowed_axis_set(allowed_axes))


def _as_bool_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "b":
        raise ValueError("Feasible expressions must evaluate to boolean values.")
    if array.shape == ():
        return np.full(shape, bool(array), dtype=bool)
    try:
        return np.broadcast_to(array, shape).astype(bool, copy=False)
    except ValueError as exc:
        raise ValueError("Feasible expression result shape is not compatible with axis arrays.") from exc


def _eval_node(node: ast.AST, axis_arrays: dict[str, Any], shape: tuple[int, ...]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, axis_arrays, shape)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in axis_arrays:
            raise ValueError(f"Axis array '{node.id}' is missing for feasible expression evaluation.")
        return axis_arrays[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, axis_arrays, shape)
        right = _eval_node(node.right, axis_arrays, shape)
        with np.errstate(all="ignore"):
            return BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, axis_arrays, shape)
        return UNARY_OPS[type(node.op)](operand)
    if isinstance(node, ast.BoolOp):
        values = [_as_bool_array(_eval_node(value, axis_arrays, shape), shape) for value in node.values]
        if isinstance(node.op, ast.And):
            return np.logical_and.reduce(values)
        return np.logical_or.reduce(values)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, axis_arrays, shape)
        comparisons: list[np.ndarray] = []
        for operator, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, axis_arrays, shape)
            comparisons.append(_as_bool_array(COMPARE_OPS[type(operator)](left, right), shape))
            left = right
        return np.logical_and.reduce(comparisons)
    if isinstance(node, ast.Call):
        function = SAFE_FUNCTIONS[node.func.id]  # type: ignore[index, union-attr]
        args = [_eval_node(argument, axis_arrays, shape) for argument in node.args]
        with np.errstate(all="ignore"):
            return function(*args)
    raise ValueError(f"Unsupported syntax '{type(node).__name__}' in feasible expression.")


def evaluate_feasible_expression_on_arrays(
    expression: str,
    axis_arrays: dict[str, np.ndarray],
    allowed_axes: list[str],
) -> np.ndarray:
    validate_feasible_expression(expression, allowed_axes)
    arrays = {name: np.asarray(axis_arrays[name]) for name in allowed_axes if name in axis_arrays}
    if arrays:
        shape = np.broadcast_shapes(*[array.shape for array in arrays.values()])
    else:
        shape = ()
    tree = ast.parse(str(expression).strip(), mode="eval")
    result = _eval_node(tree, arrays, shape)
    return _as_bool_array(result, shape)


def evaluate_feasible_expressions_on_arrays(
    expressions: list[str],
    axis_arrays: dict[str, np.ndarray],
    allowed_axes: list[str],
) -> np.ndarray:
    normalized = normalize_feasible_expressions(expressions)
    arrays = {name: np.asarray(axis_arrays[name]) for name in allowed_axes if name in axis_arrays}
    if arrays:
        shape = np.broadcast_shapes(*[array.shape for array in arrays.values()])
    else:
        shape = ()
    if not normalized:
        return np.ones(shape, dtype=bool)
    masks = [
        evaluate_feasible_expression_on_arrays(expression, axis_arrays, allowed_axes)
        for expression in normalized
    ]
    return np.logical_and.reduce(masks)


def axis_total_bins(axis: dict[str, Any]) -> int:
    return max(1, int(math.ceil((float(axis["domainMax"]) - float(axis["domainMin"])) / float(axis["resolution"]))))


def canonical_axes(axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(axis) for axis in axes), key=lambda axis: str(axis.get("name") or "").strip().lower())


def compute_valid_bin_mask_for_axes(
    axes: list[dict[str, Any]],
    expressions: list[str] | None = None,
    max_bins: int = MAX_MASK_EVAL_BINS,
) -> dict[str, Any]:
    ordered_axes = canonical_axes(axes)
    axis_names = [str(axis["name"]) for axis in ordered_axes]
    normalized = normalize_feasible_expressions(expressions)
    axis_bin_counts = [axis_total_bins(axis) for axis in ordered_axes]
    total_bins = int(math.prod(axis_bin_counts)) if axis_bin_counts else 0
    if not normalized:
        return {
            "totalBins": total_bins,
            "validBins": total_bins,
            "maskedBins": 0,
            "feasibleExpressions": [],
            "feasibleMaskEnabled": False,
            "validDomainRatio": 1.0 if total_bins else 0.0,
        }
    if total_bins > max_bins:
        raise ValueError(
            f"Feasible domain mask evaluation too large: {total_bins} bins exceeds limit {max_bins}."
        )

    centers = []
    for axis, count in zip(ordered_axes, axis_bin_counts):
        domain_min = float(axis["domainMin"])
        domain_max = float(axis["domainMax"])
        resolution = float(axis["resolution"])
        center_values = domain_min + (np.arange(count, dtype=float) + 0.5) * resolution
        centers.append(np.minimum(center_values, domain_max))
    meshes = np.meshgrid(*centers, indexing="ij") if centers else []
    axis_arrays = {axis_name: mesh for axis_name, mesh in zip(axis_names, meshes)}
    mask = evaluate_feasible_expressions_on_arrays(normalized, axis_arrays, axis_names)
    valid_bins = int(np.count_nonzero(mask))
    return {
        "totalBins": total_bins,
        "validBins": valid_bins,
        "maskedBins": int(total_bins - valid_bins),
        "feasibleExpressions": normalized,
        "feasibleMaskEnabled": True,
        "validDomainRatio": float(valid_bins / total_bins) if total_bins else 0.0,
    }


def bin_key_to_axis_centers(bin_key: str, axes: list[dict[str, Any]]) -> dict[str, float]:
    ordered_axes = canonical_axes(axes)
    try:
        indices = json.loads(str(bin_key))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid bin key: {bin_key}") from exc
    if not isinstance(indices, list) or len(indices) != len(ordered_axes):
        raise ValueError(f"Bin key does not match axis count: {bin_key}")
    centers: dict[str, float] = {}
    for raw_index, axis in zip(indices, ordered_axes):
        index = int(raw_index)
        total = axis_total_bins(axis)
        if index < 0 or index >= total:
            raise ValueError(f"Bin index {index} is outside axis range.")
        domain_min = float(axis["domainMin"])
        domain_max = float(axis["domainMax"])
        resolution = float(axis["resolution"])
        centers[str(axis["name"])] = min(domain_min + (index + 0.5) * resolution, domain_max)
    return centers


def is_bin_key_feasible(
    bin_key: str,
    axes: list[dict[str, Any]],
    expressions: list[str] | None = None,
    cache: dict[str, bool] | None = None,
) -> bool:
    normalized = normalize_feasible_expressions(expressions)
    if not normalized:
        return True
    cache_key = str(bin_key)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    centers = bin_key_to_axis_centers(cache_key, axes)
    axis_names = [str(axis["name"]) for axis in canonical_axes(axes)]
    axis_arrays = {name: np.asarray(value) for name, value in centers.items()}
    mask = evaluate_feasible_expressions_on_arrays(normalized, axis_arrays, axis_names)
    result = bool(np.asarray(mask).item())
    if cache is not None:
        cache[cache_key] = result
    return result


def filter_bin_counts_by_feasible_domain(
    bin_counts: dict[str, Any],
    axes: list[dict[str, Any]],
    expressions: list[str] | None = None,
) -> dict[str, Any]:
    """Return only bin counts inside the feasible domain.

    Invalid bin keys are skipped because they cannot be safely interpreted as
    density-grid observations. Infeasible keys are counted separately so callers
    can report how many peer rows/bins were excluded by the mask.
    """
    normalized = normalize_feasible_expressions(expressions)
    filtered: dict[str, int] = {}
    infeasible_rows = 0
    infeasible_bins = 0
    cache: dict[str, bool] = {}

    for raw_key, raw_count in (bin_counts or {}).items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        key = str(raw_key)
        try:
            feasible = is_bin_key_feasible(key, axes, normalized, cache)
        except ValueError:
            continue
        if feasible:
            filtered[key] = filtered.get(key, 0) + count
        else:
            infeasible_rows += count
            infeasible_bins += 1

    return {
        "binCounts": filtered,
        "filteredBinCounts": filtered,
        "infeasibleRows": int(infeasible_rows),
        "infeasibleBins": int(infeasible_bins),
    }
