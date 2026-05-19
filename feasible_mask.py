from __future__ import annotations

import ast
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


def normalize_feasible_expressions(expressions: Any) -> list[str]:
    if not isinstance(expressions, list):
        return []
    normalized: list[str] = []
    for raw_expression in expressions:
        expression = str(raw_expression or "").strip()
        if expression:
            normalized.append(expression)
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
