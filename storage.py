from __future__ import annotations

import json
import os
import itertools
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from feasible_box_counter import compile_rules_to_mask_boxes, compute_rectangular_total_bins, mask_signature
from feasible_mask import normalize_feasible_expressions, normalize_feasible_rules
from models import K_M, ExperimentConfig

try:
    import psycopg2  # type: ignore
except ModuleNotFoundError:
    psycopg2 = None


STORE_DIR = Path(os.environ.get("LEESIN_STORE_DIR", Path(__file__).parent)).resolve()
GOAL_STORE_PATH = STORE_DIR / "goal_store.json"
CLUSTER_STORE_PATH = STORE_DIR / "data_cluster_store.json"

DB_TABLE_NAME = os.environ.get("LEESIN_DB_TABLE", "leesin_contents").strip() or "leesin_contents"
DB_KEY_GOALS = "goal_store"
DB_KEY_CLUSTERS = "cluster_store"


DEFAULT_GOALS = [
    {
        "id": "goal_thermal",
        "name": "고온 유량 품질 인증",
        "K_m": K_M,
        "axes": [
            {"name": "temperature", "unit": "C", "domainMin": 0.0, "domainMax": 200.0, "resolution": 10.0},
            {"name": "pressure", "unit": "bar", "domainMin": 0.0, "domainMax": 100.0, "resolution": 5.0},
            {"name": "flow_rate", "unit": "kg/h", "domainMin": 0.0, "domainMax": 50.0, "resolution": 2.5},
        ],
    },
    {
        "id": "goal_vacuum",
        "name": "진공 유지 품질 인증",
        "K_m": K_M,
        "axes": [
            {"name": "vacuum_level", "unit": "kPa", "domainMin": 0.0, "domainMax": 100.0, "resolution": 5.0},
            {"name": "hold_time", "unit": "s", "domainMin": 0.0, "domainMax": 300.0, "resolution": 15.0},
            {"name": "leak_rate", "unit": "sccm", "domainMin": 0.0, "domainMax": 20.0, "resolution": 1.0},
        ],
    },
]


ANALYSIS_AT_UPLOAD_KEYS = {
    "analysisTimestamp": None,
    "peerGroupSize": None,
    "peerRecordCount": None,
    "externalPeerRecordCount": None,
    "externalPeerObservationCount": None,
    "referenceObservationCount": None,
    "referenceOccupiedBins": None,
    "targetIncludedInReference": None,
    "internalDensityMode": None,
    "selfContainedReferenceWarning": None,
    "engine": None,
    "specificityMethod": None,
    "specificityInterpretation": None,
    "specificityScore": None,
    "meanBinSpecificity": None,
    "maxSpecificity": None,
    "extremeSpecificityRate": None,
    "meanRarity": None,
    "maxRarity": None,
    "unseenBinRate": None,
    "rareBinRate": None,
    "outOfDomainRate": None,
    "confidence": None,
    "observationSupportS": None,
    "coverageC": None,
    "equitabilityE": None,
    "peerObservationCount": None,
    "validTargetRows": None,
    "invalidTargetRows": None,
    "outOfDomainRows": None,
    "maskedOutTargetRows": None,
    "infeasibleTargetRows": None,
    "totalBins": None,
    "validBins": None,
    "maskedBins": None,
    "validDomainRatio": None,
    "occupiedBins": None,
    "feasibleMaskEnabled": None,
    "feasibleMaskEvaluationSkipped": None,
    "feasibleMaskWarning": None,
    "coverageWarning": None,
    "feasibleExpressions": [],
    "infeasiblePeerRows": None,
    "infeasiblePeerBins": None,
    "outOfDomainWarnings": [],
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_optional_float_list(values: Any, size: int, default: float | None = None) -> list[float | None]:
    if not isinstance(values, list) or len(values) != size:
        return [default for _ in range(size)]
    normalized: list[float | None] = []
    for value in values:
        if value is None:
            normalized.append(None)
        else:
            try:
                normalized.append(float(value))
            except (TypeError, ValueError):
                normalized.append(default)
    return normalized


def normalize_int_count_map(values: Any) -> dict[str, int]:
    if not isinstance(values, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in values.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        normalized[str(key)] = count
    return normalized


def normalize_axis_bin_occupancy(values: Any, axis_names: list[str]) -> dict[str, dict[str, int]]:
    if not isinstance(values, dict):
        return {}
    normalized: dict[str, dict[str, int]] = {}
    axis_lookup = {normalize_axis_name(name): str(name) for name in axis_names}
    for raw_axis, counts in values.items():
        axis_name = axis_lookup.get(normalize_axis_name(raw_axis), str(raw_axis))
        count_map = normalize_int_count_map(counts)
        if count_map:
            normalized[axis_name] = count_map
    return normalized


def normalize_row_level_vectors(values: Any, width: int) -> list[list[float]]:
    if not isinstance(values, list) or width <= 0:
        return []
    normalized: list[list[float]] = []
    for row in values:
        if not isinstance(row, list) or len(row) != width:
            continue
        vector: list[float] = []
        row_ok = True
        for value in row:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                row_ok = False
                break
            if not np.isfinite(numeric):
                row_ok = False
                break
            vector.append(numeric)
        if row_ok:
            normalized.append(vector)
    return normalized


def normalize_bin_occupancy_meta(meta: Any, row_count: int, bin_occupancy: dict[str, int]) -> dict[str, Any]:
    source = meta if isinstance(meta, dict) else {}
    valid_count = source.get("validMultidimensionalRowCount")
    if valid_count is None:
        valid_count = sum(bin_occupancy.values())
    return {
        "version": int(source.get("version") or 1),
        "basis": str(source.get("basis") or ("row_level" if bin_occupancy else "unavailable")),
        "validMultidimensionalRowCount": int(valid_count or 0),
        "feasibleValidRowCount": int(source.get("feasibleValidRowCount") or valid_count or 0),
        "invalidRowCount": int(source.get("invalidRowCount") or 0),
        "outOfDomainRowCount": int(source.get("outOfDomainRowCount") or 0),
        "maskedOutRowCount": int(source.get("maskedOutRowCount") or source.get("infeasibleRowCount") or 0),
        "infeasibleRowCount": int(source.get("infeasibleRowCount") or source.get("maskedOutRowCount") or 0),
        "maskedOutBinCount": int(source.get("maskedOutBinCount") or 0),
        "totalRows": int(source.get("totalRows") or row_count or 0),
    }


def bin_occupancy_hash(values: Any) -> str:
    count_map = normalize_int_count_map(values)
    canonical = {
        str(key): int(count)
        for key, count in sorted(count_map.items(), key=lambda item: str(item[0]))
    }
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def normalize_analysis_snapshot(snapshot: Any) -> dict[str, Any]:
    source = snapshot if isinstance(snapshot, dict) else {}
    normalized = dict(ANALYSIS_AT_UPLOAD_KEYS)
    normalized["outOfDomainWarnings"] = []
    aliases = {
        "specificity_score": "specificityScore",
        "specificity_method": "specificityMethod",
        "specificity_interpretation": "specificityInterpretation",
        "mean_bin_specificity": "meanBinSpecificity",
        "max_specificity": "maxSpecificity",
        "extreme_specificity_rate": "extremeSpecificityRate",
        "mean_rarity": "meanRarity",
        "max_rarity": "maxRarity",
        "unseen_bin_rate": "unseenBinRate",
        "rare_bin_rate": "rareBinRate",
        "out_of_domain_rate": "outOfDomainRate",
        "observation_support_S": "observationSupportS",
        "observation_support_s": "observationSupportS",
        "coverage_C": "coverageC",
        "coverage_c": "coverageC",
        "equitability_E": "equitabilityE",
        "equitability_e": "equitabilityE",
        "peer_observation_count": "peerObservationCount",
        "peer_record_count": "peerRecordCount",
        "external_peer_record_count": "externalPeerRecordCount",
        "external_peer_observation_count": "externalPeerObservationCount",
        "reference_observation_count": "referenceObservationCount",
        "reference_occupied_bins": "referenceOccupiedBins",
        "target_included_in_reference": "targetIncludedInReference",
        "internal_density_mode": "internalDensityMode",
        "self_contained_reference_warning": "selfContainedReferenceWarning",
        "valid_target_rows": "validTargetRows",
        "invalid_target_rows": "invalidTargetRows",
        "out_of_domain_rows": "outOfDomainRows",
        "masked_out_target_rows": "maskedOutTargetRows",
        "infeasible_target_rows": "infeasibleTargetRows",
        "total_bins": "totalBins",
        "valid_bins": "validBins",
        "masked_bins": "maskedBins",
        "valid_domain_ratio": "validDomainRatio",
        "occupied_bins": "occupiedBins",
        "feasible_mask_enabled": "feasibleMaskEnabled",
        "feasible_mask_evaluation_skipped": "feasibleMaskEvaluationSkipped",
        "feasible_mask_warning": "feasibleMaskWarning",
        "coverage_warning": "coverageWarning",
        "feasible_expressions": "feasibleExpressions",
        "infeasible_peer_rows": "infeasiblePeerRows",
        "infeasible_peer_bins": "infeasiblePeerBins",
    }
    for key, value in source.items():
        normalized[aliases.get(key, key)] = value
    if not isinstance(normalized.get("outOfDomainWarnings"), list):
        normalized["outOfDomainWarnings"] = []
    return normalized


def cluster_fingerprint_payload(record: dict[str, Any]) -> dict[str, Any]:
    axis_names = [str(name).strip() for name in record.get("axisNames", [])]
    axis_value_pairs = sorted(
        (
            normalize_axis_name(axis_name),
            round(float(value), 12),
        )
        for axis_name, value in zip(axis_names, record.get("values", []))
        if normalize_axis_name(axis_name)
    )
    goal_id = str(record.get("goalId", "")).strip()
    return {
        "goalId": goal_id,
        "peerGroupKey": peer_group_key(goal_id, axis_names),
        "axisNames": [axis_key for axis_key, _value in axis_value_pairs],
        "values": [value for _axis_key, value in axis_value_pairs],
        "rowCount": int(record.get("rowCount", 1) or 1),
        "summaryMethod": str(record.get("summaryMethod") or "mean"),
        "binOccupancyHash": str(record.get("binOccupancyHash") or bin_occupancy_hash(record.get("binOccupancy"))),
        "gridSignature": str(record.get("gridSignature") or ""),
    }


def cluster_fingerprint(record: dict[str, Any]) -> str:
    payload = cluster_fingerprint_payload(record)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _database_requested() -> bool:
    backend = os.environ.get("LEESIN_STORAGE_BACKEND", "").strip().lower()
    return backend == "database" or _env_truthy("LEESIN_USE_DATABASE")


def _database_url_exists() -> bool:
    return bool(os.environ.get("DATABASE_URL", "").strip())


def _db_enabled() -> bool:
    return _database_requested() and _database_url_exists()


def _db_connect() -> "psycopg2.extensions.connection":
    if psycopg2 is None:
        raise ModuleNotFoundError("psycopg2 is required when database storage is explicitly enabled.")
    return psycopg2.connect(os.environ["DATABASE_URL"], connect_timeout=5)


def init_database() -> None:
    if not _db_enabled():
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        return
    conn = _db_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {DB_TABLE_NAME} (
                        id SERIAL PRIMARY KEY,
                        content TEXT
                    )
                    """
                )
    finally:
        conn.close()


def _db_store_key_pattern(key: str) -> str:
    return f'%\"key\":\"{key}\"%'


def db_insert_store_payload(key: str, payload: Any) -> int:
    envelope = {"key": key, "payload": payload}
    content = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    conn = _db_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"INSERT INTO {DB_TABLE_NAME} (content) VALUES (%s) RETURNING id", (content,))
                row = cur.fetchone()
                return int(row[0]) if row else 0
    finally:
        conn.close()


def db_select_latest_store_payload(key: str) -> Any | None:
    if not _db_enabled():
        return None
    conn = _db_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT content FROM {DB_TABLE_NAME} WHERE content LIKE %s ORDER BY id DESC LIMIT 1",
                    (_db_store_key_pattern(key),),
                )
                row = cur.fetchone()
                if not row or row[0] is None:
                    return None
                try:
                    envelope = json.loads(str(row[0]))
                except json.JSONDecodeError:
                    return None
                if not isinstance(envelope, dict) or envelope.get("key") != key:
                    return None
                return envelope.get("payload")
    finally:
        conn.close()


def storage_label(path: Path) -> str:
    try:
        return str(path.relative_to(Path(__file__).parent))
    except ValueError:
        return str(path)


def storage_status() -> dict[str, Any]:
    db_requested = _database_requested()
    database_url_exists = _database_url_exists()
    db_enabled = _db_enabled()
    store_dir_configured = bool(os.environ.get("LEESIN_STORE_DIR", "").strip())
    return {
        "storageBackend": "database" if db_enabled else "json_file",
        "storeDir": storage_label(STORE_DIR),
        "leesinStoreDirExists": STORE_DIR.exists(),
        "leesinStoreDirConfigured": store_dir_configured,
        "databaseUrlExists": database_url_exists,
        "databaseAutoUseDisabled": database_url_exists and not db_requested,
        "databaseRequested": db_requested,
        "dbEnabled": db_enabled,
        "goalStorePath": storage_label(GOAL_STORE_PATH),
        "clusterStorePath": storage_label(CLUSTER_STORE_PATH),
        "isPersistentDiskJson": (not db_enabled) and store_dir_configured,
        "isLocalJsonFallback": (not db_enabled) and (not store_dir_configured),
    }


def normalize_axis_name(name: Any) -> str:
    return str(name or "").strip().lower()


def axis_signature(axis_names: list[str]) -> tuple[str, ...]:
    return tuple(sorted(normalize_axis_name(name) for name in axis_names if normalize_axis_name(name)))


def axis_subset_key(axis_names: list[str]) -> str:
    return "|".join(axis_signature(axis_names))


def axis_role(axis: dict[str, Any]) -> str:
    return "output" if str(axis.get("role") or "").strip().lower() == "output" else "input"


def input_axes(axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [axis for axis in axes if axis_role(axis) == "input"]


def feasible_rule_axis_names(rule: dict[str, Any]) -> set[str]:
    gui_spec = rule.get("guiSpec") if isinstance(rule.get("guiSpec"), dict) else {}
    rule_type = str(gui_spec.get("type") or rule.get("sourceType") or "")
    names: set[str] = set()
    if rule_type == "focused_2d_mask" or rule.get("sourceType") == "focused_2d_mask":
        for axis_name in (gui_spec.get("xAxis"), gui_spec.get("yAxis")):
            if str(axis_name or "").strip():
                names.add(str(axis_name).strip())
        scope = gui_spec.get("scope") if isinstance(gui_spec.get("scope"), dict) else {}
        for axis_name, scope_spec in scope.items():
            if isinstance(scope_spec, dict) and str(scope_spec.get("mode") or "all").lower() == "range":
                names.add(str(axis_name).strip())
        return names
    if_spec = gui_spec.get("if") if isinstance(gui_spec.get("if"), dict) else {}
    then_spec = gui_spec.get("then") if isinstance(gui_spec.get("then"), dict) else {}
    for axis_name in (if_spec.get("axis"), then_spec.get("axis")):
        if str(axis_name or "").strip():
            names.add(str(axis_name).strip())
    return names


def canonical_grid_axes(axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for axis in axes:
        normalized_name = normalize_axis_name(axis.get("name"))
        if not normalized_name:
            continue
        canonical.append(
            {
                "name": normalized_name,
                "domainMin": float(axis.get("domainMin")),
                "domainMax": float(axis.get("domainMax")),
                "resolution": float(axis.get("resolution")),
            }
        )
    return sorted(canonical, key=lambda item: item["name"])


def grid_signature_from_axes(axes: list[dict[str, Any]]) -> str:
    payload = {"axes": canonical_grid_axes(axes)}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def peer_group_key(goal_id: str, axis_names: list[str]) -> str:
    return f"{str(goal_id).strip()}|{axis_subset_key(axis_names)}"


def _default_goal_payload() -> list[dict[str, Any]]:
    return json.loads(json.dumps(DEFAULT_GOALS, ensure_ascii=False))


def validate_goal(goal: dict[str, Any]) -> dict[str, Any]:
    name = str(goal.get("name", "")).strip()
    if not name:
        raise ValueError("Experiment Goal name is required.")
    axes = goal.get("axes", [])
    if not isinstance(axes, list) or not axes:
        raise ValueError("At least one axis is required.")
    k_m = float(goal.get("K_m", goal.get("km", K_M)))
    if k_m <= 0:
        raise ValueError("K_m must be greater than 0.")

    normalized_axes = []
    seen: set[str] = set()
    for axis in axes:
        axis_name = str(axis.get("name", "")).strip()
        if not axis_name:
            raise ValueError("Axis name is required.")
        key = normalize_axis_name(axis_name)
        if key in seen:
            raise ValueError(f"Axis '{axis_name}' is duplicated.")
        seen.add(key)
        normalized_axes.append(
            {
                "name": axis_name,
                "unit": str(axis.get("unit", "")).strip(),
                "domainMin": float(axis.get("domainMin")),
                "domainMax": float(axis.get("domainMax")),
                "resolution": float(axis.get("resolution")),
                "role": axis_role(axis),
            }
        )
    design_axes = input_axes(normalized_axes)
    if not design_axes:
        raise ValueError("At least one input axis is required.")

    ExperimentConfig(
        axis_names=[axis["name"] for axis in normalized_axes],
        domain_range=[(axis["domainMin"], axis["domainMax"]) for axis in normalized_axes],
        resolution=[axis["resolution"] for axis in normalized_axes],
        K_m=k_m,
    )
    allowed_axes = [axis["name"] for axis in design_axes]
    raw_rules = goal.get("feasibleDomainRules") if isinstance(goal.get("feasibleDomainRules"), list) else []
    raw_input_rules = []
    role_excluded_rules = []
    allowed_axis_set = set(allowed_axes)
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue
        if feasible_rule_axis_names(raw_rule).issubset(allowed_axis_set):
            raw_input_rules.append(raw_rule)
        else:
            excluded = dict(raw_rule)
            excluded["enabled"] = False
            excluded["roleExcluded"] = True
            excluded["roleExcludedReason"] = "Output axis rules are excluded from certified design masks."
            role_excluded_rules.append(excluded)
    feasible_rules = normalize_feasible_rules(raw_input_rules, allowed_axes) + role_excluded_rules
    rule_expressions = [str(rule.get("expression") or "") for rule in feasible_rules if rule.get("enabled", True) and str(rule.get("expression") or "")]
    source_advanced = []
    source_advanced.extend(normalize_feasible_expressions(goal.get("legacyAdvancedExpressions")))
    source_advanced.extend(normalize_feasible_expressions(goal.get("feasibleDomainAdvancedExpressions")))
    source_expressions = normalize_feasible_expressions(goal.get("feasibleDomainExpressions"))
    rule_expression_set = set(rule_expressions)
    source_advanced.extend(expression for expression in source_expressions if expression not in rule_expression_set)
    legacy_advanced_expressions = []
    seen_legacy: set[str] = set()
    for expression in source_advanced:
        if expression and expression not in seen_legacy:
            legacy_advanced_expressions.append(expression)
            seen_legacy.add(expression)
    feasible_expressions = rule_expressions
    active_feasible_rules = [rule for rule in feasible_rules if rule.get("enabled", True)]
    mask_boxes = compile_rules_to_mask_boxes(active_feasible_rules, design_axes)
    rectangular_total_bins = compute_rectangular_total_bins(design_axes)
    current_mask_signature = mask_signature(design_axes, mask_boxes)
    cache = goal.get("aValidCache") if isinstance(goal.get("aValidCache"), dict) else {}
    cache_ready = (
        cache.get("maskSignature") == current_mask_signature
        and cache.get("aValid") is not None
        and cache.get("maskedBins") is not None
        and cache.get("rectangularTotalBins") == rectangular_total_bins
    )
    if not mask_boxes:
        cache_ready = True
        cache = {
            "maskSignature": current_mask_signature,
            "rectangularTotalBins": rectangular_total_bins,
            "maskedBins": 0,
            "aValid": rectangular_total_bins,
            "aValidMode": "exact_box_union",
            "computedAt": goal.get("aValidComputedAt"),
            "durationMs": 0,
        }
    a_valid_status = "ready" if cache_ready else "stale"
    return {
        "id": str(goal.get("id") or f"goal_{abs(hash(name))}"),
        "name": name,
        "K_m": k_m,
        "axes": normalized_axes,
        "inputAxisNames": [axis["name"] for axis in design_axes],
        "outputAxisNames": [axis["name"] for axis in normalized_axes if axis_role(axis) == "output"],
        "feasibleDomainRules": feasible_rules,
        "legacyAdvancedExpressions": legacy_advanced_expressions,
        "feasibleDomainAdvancedExpressions": [],
        "feasibleDomainExpressions": feasible_expressions,
        "generatedFeasibleExpressions": feasible_expressions,
        "feasibleMaskBoxes": mask_boxes,
        "maskBoxCount": len(mask_boxes),
        "rectangularTotalBins": rectangular_total_bins,
        "designTotalBins": rectangular_total_bins,
        "aValid": cache.get("aValid") if cache_ready else None,
        "designAValid": cache.get("aValid") if cache_ready else None,
        "maskedBins": cache.get("maskedBins") if cache_ready else None,
        "designMaskedBins": cache.get("maskedBins") if cache_ready else None,
        "designAxes": [axis["name"] for axis in design_axes],
        "aValidStatus": a_valid_status,
        "aValidProgressPercent": 100 if cache_ready else 0,
        "aValidMode": "exact_box_union",
        "maskSignature": current_mask_signature,
        "aValidComputedAt": cache.get("computedAt") if cache_ready else None,
        "aValidCache": dict(cache) if cache else {},
    }


def normalize_goal_for_display(goal: dict[str, Any]) -> dict[str, Any]:
    return validate_goal(goal)


def normalize_goals_for_display(goals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_goal_for_display(goal) for goal in goals]


def load_goal_store() -> list[dict[str, Any]]:
    raw_goals: Any | None = None
    if _db_enabled():
        raw_goals = db_select_latest_store_payload(DB_KEY_GOALS)

    if raw_goals is None:
        if GOAL_STORE_PATH.exists():
            raw_goals = json.loads(GOAL_STORE_PATH.read_text(encoding="utf-8"))
        else:
            raw_goals = _default_goal_payload()
        if _db_enabled():
            db_insert_store_payload(DB_KEY_GOALS, raw_goals)

    normalized: list[dict[str, Any]] = []
    for goal in raw_goals:
        try:
            normalized.append(validate_goal(goal))
        except (TypeError, ValueError, KeyError):
            continue

    if not normalized:
        normalized = [validate_goal(goal) for goal in _default_goal_payload()]
    return normalized


def save_goal_store(goals: list[dict[str, Any]]) -> None:
    normalized = [validate_goal(goal) for goal in goals]
    if _db_enabled():
        db_insert_store_payload(DB_KEY_GOALS, normalized)
        return
    GOAL_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOAL_STORE_PATH.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_cluster(item: dict[str, Any]) -> dict[str, Any] | None:
    try:
        axis_names = [str(name).strip() for name in item["axisNames"]]
        values = [float(value) for value in item["values"]]
        if len(axis_names) != len(values):
            return None
        goal_id = str(item["goalId"]).strip()
        signature = axis_subset_key(axis_names)
        row_count = int(item.get("rowCount", 1) or 1)
        summary_method = str(item.get("summaryMethod") or item.get("summary_method") or "mean")
        created_at = str(item.get("createdAt") or item.get("uploadedAt") or utc_now_iso())
        uploaded_at = str(item.get("uploadedAt") or created_at)
        analysis = normalize_analysis_snapshot(item.get("analysisAtUpload"))
        bin_occupancy = normalize_int_count_map(item.get("binOccupancy"))
        axis_bin_occupancy = normalize_axis_bin_occupancy(item.get("axisBinOccupancy"), axis_names)
        bin_occupancy_meta = normalize_bin_occupancy_meta(item.get("binOccupancyMeta"), row_count, bin_occupancy)
        cluster_vector_row_count = int(item.get("clusterVectorRowCount") or item.get("validMultidimensionalNumericRowCount") or row_count or 0)
        axis_numeric_counts = normalize_int_count_map(item.get("axisNumericCounts"))
        row_level_vector_axis_order = [str(name).strip() for name in (item.get("rowLevelVectorAxisOrder") or axis_names)]
        row_level_vectors = normalize_row_level_vectors(item.get("rowLevelVectors"), len(row_level_vector_axis_order))
        row_level_vector_count = len(row_level_vectors)
        has_row_level_vectors = bool(row_level_vectors)
        occupancy_hash = str(item.get("binOccupancyHash") or bin_occupancy_hash(bin_occupancy))
        grid_signature = str(item.get("gridSignature") or "").strip()
        normalized = {
            "id": str(item.get("id") or f"cluster_{os.urandom(8).hex()}"),
            "goalId": goal_id,
            "goalName": str(item.get("goalName", "")),
            "axisNames": axis_names,
            "axisSignature": signature,
            "gridSignature": grid_signature,
            "peerGroupKey": peer_group_key(goal_id, axis_names),
            "values": values,
            "valuesMean": normalize_optional_float_list(item.get("valuesMean", values), len(values), None),
            "valuesVariance": normalize_optional_float_list(item.get("valuesVariance"), len(values), None),
            "valuesStd": normalize_optional_float_list(item.get("valuesStd"), len(values), None),
            "rowCount": row_count,
            "binOccupancy": bin_occupancy,
            "axisBinOccupancy": axis_bin_occupancy,
            "binOccupancyMeta": bin_occupancy_meta,
            "binOccupancyHash": occupancy_hash,
            "clusterVectorRowCount": cluster_vector_row_count,
            "clusterVectorBasis": str(item.get("clusterVectorBasis") or "valid_multidimensional_numeric_rows"),
            "axisNumericCounts": axis_numeric_counts,
            "rowLevelVectors": row_level_vectors,
            "rowLevelVectorAxisOrder": row_level_vector_axis_order,
            "rowLevelVectorCount": row_level_vector_count,
            "rowLevelVectorBasis": str(item.get("rowLevelVectorBasis") or "valid_multidimensional_numeric_rows"),
            "hasRowLevelVectors": has_row_level_vectors,
            "createdAt": created_at,
            "uploadedAt": uploaded_at,
            "sourceBatchId": item.get("sourceBatchId") or item.get("source_batch_id"),
            "summaryMethod": summary_method,
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
        normalized["fingerprint"] = cluster_fingerprint(normalized)
        return normalized
    except (TypeError, ValueError, KeyError):
        return None


def load_cluster_store() -> list[dict[str, Any]]:
    raw_clusters: Any | None = None
    if _db_enabled():
        raw_clusters = db_select_latest_store_payload(DB_KEY_CLUSTERS)
    if raw_clusters is None:
        if not CLUSTER_STORE_PATH.exists():
            return []
        raw = json.loads(CLUSTER_STORE_PATH.read_text(encoding="utf-8"))
        raw_clusters = raw.get("clusters", raw if isinstance(raw, list) else [])

    clusters = []
    for item in raw_clusters:
        normalized = _normalize_cluster(item)
        if normalized:
            clusters.append(normalized)
    return clusters


def save_cluster_store(clusters: list[dict[str, Any]]) -> None:
    normalized = [cluster for cluster in (_normalize_cluster(item) for item in clusters) if cluster]
    payload = {
        "version": 1,
        "privacy": "Stores only selected-axis sanitized numeric row vectors, compatibility mean summaries, row-level bin count summaries, grid signatures, and goal metadata. Raw uploaded rows, filenames, unmapped columns, and personal data columns are not stored.",
        "clusterDefinition": "One CSV file is one saved record. Density scoring re-bins row-level sanitized axis vectors; the mean vector is compatibility/display metadata only.",
        "clusters": normalized,
    }
    if _db_enabled():
        db_insert_store_payload(DB_KEY_CLUSTERS, normalized)
        return
    CLUSTER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLUSTER_STORE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def should_save_data_clusters() -> bool:
    return os.environ.get("SAVE_DATA_CLUSTERS", "true").lower() in {"1", "true", "yes", "on"}


def _extract_selected_values(cluster: dict[str, Any], selected_axis_names: list[str]) -> list[float] | None:
    values, _reason = extract_selected_values_with_reason(cluster, selected_axis_names)
    return values


def extract_selected_values_with_reason(
    cluster: dict[str, Any],
    selected_axis_names: list[str],
) -> tuple[list[float] | None, dict[str, Any]]:
    cluster_axes = [str(name) for name in cluster.get("axisNames", [])]
    cluster_values = cluster.get("values", [])
    value_by_axis: dict[str, float] = {}
    duplicate_axes: list[str] = []
    for axis_name, value in zip(cluster_axes, cluster_values):
        axis_key = normalize_axis_name(axis_name)
        if not axis_key:
            continue
        if axis_key in value_by_axis:
            duplicate_axes.append(str(axis_name))
            continue
        value_by_axis[axis_key] = float(value)

    selected_keys = [normalize_axis_name(axis_name) for axis_name in selected_axis_names]
    missing_axes = [
        str(axis_name)
        for axis_name, axis_key in zip(selected_axis_names, selected_keys)
        if not axis_key or axis_key not in value_by_axis
    ]
    reason = {
        "clusterId": str(cluster.get("id", "")),
        "clusterAxisNames": cluster_axes,
        "clusterAxisKeys": [normalize_axis_name(axis_name) for axis_name in cluster_axes],
        "selectedAxisNames": [str(axis_name) for axis_name in selected_axis_names],
        "selectedAxisKeys": selected_keys,
        "missingAxes": missing_axes,
        "duplicateAxes": duplicate_axes,
    }
    if missing_axes:
        reason["reason"] = "missing_axis"
        cluster["_peerFilterDebug"] = reason
        return None
    if duplicate_axes:
        reason["reason"] = "duplicate_axis"
    else:
        reason["reason"] = "compatible"
    return [float(value_by_axis[axis_key]) for axis_key in selected_keys], reason


def explain_peer_filter(
    goal_id: str,
    selected_axis_names: list[str],
    exclude_cluster_id: str | None = None,
    example_limit: int = 3,
) -> dict[str, Any]:
    clusters = load_cluster_store()
    wanted_goal_id = str(goal_id).strip()
    selected_names = [str(name) for name in selected_axis_names]
    selected_keys = [normalize_axis_name(name) for name in selected_names]
    diagnostics: dict[str, Any] = {
        "totalClusters": len(clusters),
        "sameGoalCount": 0,
        "compatibleAxisCount": 0,
        "excludedByGoal": 0,
        "excludedByAxis": 0,
        "excludedBySelf": 0,
        "selectedAxisNames": selected_names,
        "selectedAxisKeys": selected_keys,
        "peerGroupKey": peer_group_key(wanted_goal_id, selected_names),
        "examplesExcluded": [],
    }

    for cluster in clusters:
        cluster_id = str(cluster.get("id", ""))
        if exclude_cluster_id and cluster_id == str(exclude_cluster_id):
            diagnostics["excludedBySelf"] += 1
            if len(diagnostics["examplesExcluded"]) < example_limit:
                diagnostics["examplesExcluded"].append(
                    {
                        "id": cluster_id,
                        "goalId": str(cluster.get("goalId", "")),
                        "axisNames": list(cluster.get("axisNames", [])),
                        "reason": "excluded_self",
                    }
                )
            continue
        if str(cluster.get("goalId", "")).strip() != wanted_goal_id:
            diagnostics["excludedByGoal"] += 1
            if len(diagnostics["examplesExcluded"]) < example_limit:
                diagnostics["examplesExcluded"].append(
                    {
                        "id": cluster_id,
                        "goalId": str(cluster.get("goalId", "")),
                        "axisNames": list(cluster.get("axisNames", [])),
                        "reason": "different_goal",
                    }
                )
            continue
        diagnostics["sameGoalCount"] += 1
        values, reason = extract_selected_values_with_reason(cluster, selected_names)
        if values is None:
            diagnostics["excludedByAxis"] += 1
            if len(diagnostics["examplesExcluded"]) < example_limit:
                diagnostics["examplesExcluded"].append(
                    {
                        "id": cluster_id,
                        "goalId": str(cluster.get("goalId", "")),
                        "axisNames": list(cluster.get("axisNames", [])),
                        "reason": reason.get("reason", "axis_mismatch"),
                        "missingAxes": reason.get("missingAxes", []),
                        "clusterAxisKeys": reason.get("clusterAxisKeys", []),
                    }
                )
            continue
        diagnostics["compatibleAxisCount"] += 1
    return diagnostics


def load_peer_clusters(
    goal_id: str,
    selected_axis_names: list[str],
    exclude_cluster_id: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    wanted_goal_id = str(goal_id).strip()
    selected_names = [str(name).strip() for name in selected_axis_names]
    selected_key = axis_subset_key(selected_axis_names)
    for cluster in load_cluster_store():
        if str(cluster.get("goalId", "")).strip() != wanted_goal_id:
            continue
        if exclude_cluster_id and str(cluster.get("id")) == str(exclude_cluster_id):
            continue
        values = _extract_selected_values(cluster, selected_names)
        if values is None:
            continue
        item = dict(cluster)
        item["storedAxisNames"] = list(cluster.get("axisNames", []))
        item["storedAxisSignature"] = str(cluster.get("axisSignature") or axis_subset_key(list(cluster.get("axisNames", []))))
        item["axisNames"] = list(selected_names)
        item["axisSignature"] = selected_key
        item["peerGroupKey"] = peer_group_key(wanted_goal_id, selected_names)
        item["values"] = values
        rows.append(item)
    return rows


def get_peer_group(
    goal: dict[str, Any],
    selected_axis_names: list[str],
    exclude_cluster_id: str | None = None,
) -> np.ndarray:
    rows = [
        [float(value) for value in cluster["values"]]
        for cluster in load_peer_clusters(str(goal["id"]), selected_axis_names, exclude_cluster_id)
    ]
    if not rows:
        raise ValueError(
            "No saved records match the selected Experiment Goal and Axis configuration."
        )
    return np.asarray(rows, dtype=float)


def pick_peer_group(
    goal: dict[str, Any],
    selected_axis_names: list[str] | None = None,
    exclude_cluster_id: str | None = None,
) -> np.ndarray:
    selected = selected_axis_names or [axis["name"] for axis in goal["axes"]]
    return get_peer_group(goal, selected, exclude_cluster_id)


def peer_group_subset_counts(goal: dict[str, Any]) -> dict[str, int]:
    axis_names = [axis["name"] for axis in goal["axes"]]
    counts: dict[str, int] = {}
    for size in range(1, len(axis_names) + 1):
        for subset in itertools.combinations(axis_names, size):
            names = list(subset)
            stored_count = len(load_peer_clusters(str(goal["id"]), names))
            counts[axis_subset_key(names)] = stored_count
    return counts


def save_peer_cluster(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    clusters = load_cluster_store()
    normalized = _normalize_cluster(record)
    if normalized is None:
        raise ValueError("Cluster record is invalid.")
    wanted_payload = cluster_fingerprint_payload(normalized)
    for cluster in clusters:
        if cluster.get("fingerprint") == normalized["fingerprint"] or cluster_fingerprint_payload(cluster) == wanted_payload:
            return cluster, False
    clusters.append(normalized)
    save_cluster_store(clusters)
    return normalized, True


def save_data_cluster(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    return save_peer_cluster(record)


def delete_peer_cluster(cluster_id: str) -> bool:
    wanted = str(cluster_id or "").strip()
    clusters = load_cluster_store()
    kept = [cluster for cluster in clusters if str(cluster.get("id")) != wanted]
    if len(kept) == len(clusters):
        return False
    save_cluster_store(kept)
    return True


def list_cluster_summaries(include_density: bool = False) -> list[dict[str, Any]]:
    return [cluster_summary(cluster, include_density=include_density) for cluster in load_cluster_store()]


def cluster_summary(cluster: dict[str, Any], include_density: bool = False) -> dict[str, Any]:
    bin_occupancy = normalize_int_count_map(cluster.get("binOccupancy"))
    axis_bin_occupancy = normalize_axis_bin_occupancy(cluster.get("axisBinOccupancy"), list(cluster.get("axisNames") or []))
    bin_meta = normalize_bin_occupancy_meta(cluster.get("binOccupancyMeta"), int(cluster.get("rowCount", 0) or 0), bin_occupancy)
    summary = {
        "id": str(cluster.get("id", "")),
        "goalId": str(cluster.get("goalId", "")),
        "goalName": str(cluster.get("goalName", "")),
        "peerGroupKey": str(cluster.get("peerGroupKey", "")),
        "axisNames": list(cluster.get("axisNames") or []),
        "axisSignature": str(cluster.get("axisSignature", "")),
        "gridSignature": str(cluster.get("gridSignature", "")),
        "values": [round(float(value), 6) for value in cluster.get("values", [])],
        "valuesMean": [None if value is None else round(float(value), 6) for value in cluster.get("valuesMean", [])],
        "valuesVariance": [None if value is None else round(float(value), 6) for value in cluster.get("valuesVariance", [])],
        "valuesStd": [None if value is None else round(float(value), 6) for value in cluster.get("valuesStd", [])],
        "rowCount": int(cluster.get("rowCount", 0) or 0),
        "binOccupancyHash": str(cluster.get("binOccupancyHash") or bin_occupancy_hash(bin_occupancy)),
        "hasRowLevelBinOccupancy": bool(bin_occupancy),
        "rowLevelValidCount": int(bin_meta.get("validMultidimensionalRowCount") or 0),
        "rowLevelMaskedOutCount": int(bin_meta.get("maskedOutRowCount") or bin_meta.get("infeasibleRowCount") or 0),
        "rowLevelOccupiedBinCount": len(bin_occupancy),
        "rowLevelVectorCount": int(cluster.get("rowLevelVectorCount") or len(cluster.get("rowLevelVectors") or []) or 0),
        "rowLevelVectorAxisOrder": list(cluster.get("rowLevelVectorAxisOrder") or []),
        "rowLevelVectorBasis": str(cluster.get("rowLevelVectorBasis") or "valid_multidimensional_numeric_rows"),
        "hasRowLevelVectors": bool(cluster.get("hasRowLevelVectors") or cluster.get("rowLevelVectors")),
        "clusterVectorRowCount": int(cluster.get("clusterVectorRowCount") or cluster.get("validMultidimensionalNumericRowCount") or 0),
        "clusterVectorBasis": str(cluster.get("clusterVectorBasis") or "valid_multidimensional_numeric_rows"),
        "axisNumericCounts": normalize_int_count_map(cluster.get("axisNumericCounts")),
        "createdAt": str(cluster.get("createdAt", "")),
        "uploadedAt": str(cluster.get("uploadedAt", cluster.get("createdAt", ""))),
        "sourceBatchId": cluster.get("sourceBatchId"),
        "summaryMethod": str(cluster.get("summaryMethod", "mean")),
        "fingerprint": str(cluster.get("fingerprint", "")),
        "peerGroupSizeAtUpload": cluster.get("peerGroupSizeAtUpload"),
        "engineAtUpload": cluster.get("engineAtUpload"),
        "specificityScoreAtUpload": cluster.get("specificityScoreAtUpload"),
        "specificityMethodAtUpload": cluster.get("specificityMethodAtUpload"),
        "meanBinSpecificityAtUpload": cluster.get("meanBinSpecificityAtUpload"),
        "maxSpecificityAtUpload": cluster.get("maxSpecificityAtUpload"),
        "extremeSpecificityRateAtUpload": cluster.get("extremeSpecificityRateAtUpload"),
        "meanRarityAtUpload": cluster.get("meanRarityAtUpload"),
        "maxRarityAtUpload": cluster.get("maxRarityAtUpload"),
        "unseenBinRateAtUpload": cluster.get("unseenBinRateAtUpload"),
        "rareBinRateAtUpload": cluster.get("rareBinRateAtUpload"),
        "outOfDomainRateAtUpload": cluster.get("outOfDomainRateAtUpload"),
        "confidenceAtUpload": cluster.get("confidenceAtUpload"),
        "observationSupportSAtUpload": cluster.get("observationSupportSAtUpload"),
        "peerObservationCountAtUpload": cluster.get("peerObservationCountAtUpload"),
        "validTargetRowsAtUpload": cluster.get("validTargetRowsAtUpload"),
        "maskedOutTargetRowsAtUpload": cluster.get("maskedOutTargetRowsAtUpload"),
        "coverageCAtUpload": cluster.get("coverageCAtUpload"),
        "equitabilityEAtUpload": cluster.get("equitabilityEAtUpload"),
        "totalBinsAtUpload": cluster.get("totalBinsAtUpload"),
        "validBinsAtUpload": cluster.get("validBinsAtUpload"),
        "maskedBinsAtUpload": cluster.get("maskedBinsAtUpload"),
        "occupiedBinsAtUpload": cluster.get("occupiedBinsAtUpload"),
    }
    analysis = normalize_analysis_snapshot(cluster.get("analysisAtUpload"))
    summary["feasibleMaskEnabledAtUpload"] = bool(analysis.get("feasibleMaskEnabled"))
    if include_density:
        summary.update(
            {
                "binOccupancy": bin_occupancy,
                "axisBinOccupancy": axis_bin_occupancy,
                "binOccupancyMeta": bin_meta,
                "analysisAtUpload": analysis,
            }
        )
    return summary
