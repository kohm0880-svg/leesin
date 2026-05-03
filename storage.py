from __future__ import annotations

import json
import os
import itertools
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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


SAMPLE_PEER_LIBRARY = {
    "goal_thermal|temperature|pressure|flow_rate": [
        [101.2, 49.8, 24.5],
        [98.7, 51.1, 26.0],
        [103.5, 53.2, 24.9],
        [96.8, 47.5, 25.4],
        [100.9, 50.7, 23.8],
        [99.5, 48.9, 24.1],
        [104.1, 54.5, 25.7],
        [97.9, 46.8, 24.9],
        [102.7, 52.2, 26.3],
        [101.8, 49.5, 24.7],
        [98.9, 48.2, 23.5],
        [95.6, 45.1, 22.8],
        [105.0, 55.4, 27.2],
        [103.1, 51.8, 25.5],
        [99.7, 50.0, 24.0],
    ],
    "goal_vacuum|vacuum_level|hold_time|leak_rate": [
        [52.0, 120.0, 2.1],
        [55.5, 135.0, 1.8],
        [49.8, 110.0, 2.4],
        [57.2, 140.0, 1.5],
        [53.9, 128.0, 1.9],
        [51.0, 118.0, 2.2],
        [56.1, 144.0, 1.6],
        [54.2, 132.0, 1.8],
        [50.6, 115.0, 2.0],
        [58.0, 150.0, 1.4],
    ],
}


ANALYSIS_AT_UPLOAD_KEYS = {
    "analysisTimestamp": None,
    "peerGroupSize": None,
    "engine": None,
    "isNormal": None,
    "center": None,
    "D2": None,
    "pValue": None,
    "heterogeneity": None,
    "confidence": None,
    "sampleSizeZ": None,
    "coverageC": None,
    "equitabilityE": None,
    "wEff": None,
    "totalBins": None,
    "occupiedBins": None,
    "contributions": None,
    "mardiaSkewStat": None,
    "mardiaSkewPval": None,
    "mardiaKurtStat": None,
    "mardiaKurtPval": None,
    "b2p": None,
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


def normalize_analysis_snapshot(snapshot: Any) -> dict[str, Any]:
    source = snapshot if isinstance(snapshot, dict) else {}
    normalized = dict(ANALYSIS_AT_UPLOAD_KEYS)
    normalized["outOfDomainWarnings"] = []
    aliases = {
        "p_value": "pValue",
        "sample_size_Z": "sampleSizeZ",
        "coverage_C": "coverageC",
        "equitability_E": "equitabilityE",
        "w_eff": "wEff",
        "mardia_skew_stat": "mardiaSkewStat",
        "mardia_skew_pval": "mardiaSkewPval",
        "mardia_kurt_stat": "mardiaKurtStat",
        "mardia_kurt_pval": "mardiaKurtPval",
        "is_normal": "isNormal",
    }
    for key, value in source.items():
        normalized[aliases.get(key, key)] = value
    if not isinstance(normalized.get("outOfDomainWarnings"), list):
        normalized["outOfDomainWarnings"] = []
    return normalized


def cluster_fingerprint_payload(record: dict[str, Any]) -> dict[str, Any]:
    axis_names = [str(name) for name in record.get("axisNames", [])]
    values = [round(float(value), 12) for value in record.get("values", [])]
    goal_id = str(record.get("goalId", ""))
    return {
        "goalId": goal_id,
        "peerGroupKey": str(record.get("peerGroupKey") or peer_group_key(goal_id, axis_names)),
        "axisNames": axis_names,
        "values": values,
        "rowCount": int(record.get("rowCount", 1) or 1),
        "summaryMethod": str(record.get("summaryMethod") or "mean"),
    }


def cluster_fingerprint(record: dict[str, Any]) -> str:
    payload = cluster_fingerprint_payload(record)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _db_enabled() -> bool:
    url = os.environ.get("DATABASE_URL")
    return bool(url and url.strip())


def _db_connect() -> "psycopg2.extensions.connection":
    if psycopg2 is None:
        raise ModuleNotFoundError("psycopg2 is required when DATABASE_URL is set.")
    return psycopg2.connect(os.environ["DATABASE_URL"], connect_timeout=5)


def init_database() -> None:
    if not _db_enabled():
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


def axis_signature(axis_names: list[str]) -> tuple[str, ...]:
    return tuple(str(name).strip().lower() for name in axis_names)


def axis_subset_key(axis_names: list[str]) -> str:
    return "|".join(axis_signature(axis_names))


def peer_group_key(goal_id: str, axis_names: list[str]) -> str:
    return f"{goal_id}|{axis_subset_key(axis_names)}"


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
        key = axis_name.lower()
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
            }
        )

    ExperimentConfig(
        axis_names=[axis["name"] for axis in normalized_axes],
        domain_range=[(axis["domainMin"], axis["domainMax"]) for axis in normalized_axes],
        resolution=[axis["resolution"] for axis in normalized_axes],
        K_m=k_m,
    )
    return {
        "id": str(goal.get("id") or f"goal_{abs(hash(name))}"),
        "name": name,
        "K_m": k_m,
        "axes": normalized_axes,
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
        axis_names = [str(name) for name in item["axisNames"]]
        values = [float(value) for value in item["values"]]
        if len(axis_names) != len(values):
            return None
        goal_id = str(item["goalId"])
        signature = axis_subset_key(axis_names)
        row_count = int(item.get("rowCount", 1) or 1)
        summary_method = str(item.get("summaryMethod") or item.get("summary_method") or "mean")
        created_at = str(item.get("createdAt") or item.get("uploadedAt") or utc_now_iso())
        uploaded_at = str(item.get("uploadedAt") or created_at)
        analysis = normalize_analysis_snapshot(item.get("analysisAtUpload"))
        normalized = {
            "id": str(item.get("id") or f"cluster_{os.urandom(8).hex()}"),
            "goalId": goal_id,
            "goalName": str(item.get("goalName", "")),
            "axisNames": axis_names,
            "axisSignature": signature,
            "peerGroupKey": str(item.get("peerGroupKey") or peer_group_key(goal_id, axis_names)),
            "values": values,
            "valuesMean": normalize_optional_float_list(item.get("valuesMean", values), len(values), None),
            "valuesVariance": normalize_optional_float_list(item.get("valuesVariance"), len(values), None),
            "valuesStd": normalize_optional_float_list(item.get("valuesStd"), len(values), None),
            "rowCount": row_count,
            "createdAt": created_at,
            "uploadedAt": uploaded_at,
            "sourceBatchId": item.get("sourceBatchId") or item.get("source_batch_id"),
            "summaryMethod": summary_method,
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
        "privacy": "Stores only mapped numeric axis vectors, row counts, and goal metadata. Raw uploaded rows, filenames, and unmapped columns are not stored.",
        "clusterDefinition": "One CSV file is one data cluster. CSV rows are repeated observations summarized into one cluster vector.",
        "clusters": normalized,
    }
    if _db_enabled():
        db_insert_store_payload(DB_KEY_CLUSTERS, normalized)
        return
    CLUSTER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLUSTER_STORE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def should_save_data_clusters() -> bool:
    return os.environ.get("SAVE_DATA_CLUSTERS", "true").lower() in {"1", "true", "yes", "on"}


def use_demo_peer_group() -> bool:
    return os.environ.get("USE_DEMO_PEER_GROUP", "false").lower() in {"1", "true", "yes", "on"}


def _extract_selected_values(cluster: dict[str, Any], selected_axis_names: list[str]) -> list[float] | None:
    cluster_axes = [str(name) for name in cluster.get("axisNames", [])]
    value_by_axis = dict(zip(cluster_axes, cluster.get("values", [])))
    if not all(axis_name in value_by_axis for axis_name in selected_axis_names):
        return None
    return [float(value_by_axis[axis_name]) for axis_name in selected_axis_names]


def load_peer_clusters(
    goal_id: str,
    selected_axis_names: list[str],
    exclude_cluster_id: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    selected_key = axis_subset_key(selected_axis_names)
    for cluster in load_cluster_store():
        if str(cluster.get("goalId")) != str(goal_id):
            continue
        if exclude_cluster_id and str(cluster.get("id")) == str(exclude_cluster_id):
            continue
        values = _extract_selected_values(cluster, selected_axis_names)
        if values is None:
            continue
        item = dict(cluster)
        item["axisNames"] = list(selected_axis_names)
        item["axisSignature"] = selected_key
        item["peerGroupKey"] = peer_group_key(goal_id, selected_axis_names)
        item["values"] = values
        rows.append(item)
    return rows


def demo_peer_rows(goal: dict[str, Any], selected_axis_names: list[str]) -> list[list[float]]:
    key = peer_group_key(str(goal["id"]), [axis["name"] for axis in goal["axes"]])
    rows = SAMPLE_PEER_LIBRARY.get(key, [])
    if not rows:
        return []
    name_to_index = {axis["name"]: index for index, axis in enumerate(goal["axes"])}
    try:
        indices = [name_to_index[name] for name in selected_axis_names]
    except KeyError as exc:
        raise ValueError(f"Selected axis '{exc.args[0]}' does not belong to this Goal.") from exc
    return [[float(row[index]) for index in indices] for row in rows]


def get_peer_group(
    goal: dict[str, Any],
    selected_axis_names: list[str],
    exclude_cluster_id: str | None = None,
) -> np.ndarray:
    stored = [
        [float(value) for value in cluster["values"]]
        for cluster in load_peer_clusters(str(goal["id"]), selected_axis_names, exclude_cluster_id)
    ]
    rows = list(stored)
    if use_demo_peer_group():
        rows = demo_peer_rows(goal, selected_axis_names) + rows
    if not rows:
        raise ValueError(
            "이 Experiment Goal과 Axis 구성이 정확히 같은 누적 군집만 Peer Group으로 사용됩니다. "
            "새 Goal은 충분한 과거 군집이 쌓이기 전까지 분석 신뢰도가 낮거나 분석이 제한될 수 있습니다."
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
            demo_count = len(demo_peer_rows(goal, names)) if use_demo_peer_group() else 0
            counts[axis_subset_key(names)] = stored_count + demo_count
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


def list_cluster_summaries() -> list[dict[str, Any]]:
    return [cluster_summary(cluster) for cluster in load_cluster_store()]


def cluster_summary(cluster: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(cluster.get("id", "")),
        "goalId": str(cluster.get("goalId", "")),
        "goalName": str(cluster.get("goalName", "")),
        "peerGroupKey": str(cluster.get("peerGroupKey", "")),
        "axisNames": list(cluster.get("axisNames") or []),
        "values": [round(float(value), 6) for value in cluster.get("values", [])],
        "valuesMean": [None if value is None else round(float(value), 6) for value in cluster.get("valuesMean", [])],
        "valuesVariance": [None if value is None else round(float(value), 6) for value in cluster.get("valuesVariance", [])],
        "valuesStd": [None if value is None else round(float(value), 6) for value in cluster.get("valuesStd", [])],
        "rowCount": int(cluster.get("rowCount", 0) or 0),
        "createdAt": str(cluster.get("createdAt", "")),
        "uploadedAt": str(cluster.get("uploadedAt", cluster.get("createdAt", ""))),
        "sourceBatchId": cluster.get("sourceBatchId"),
        "summaryMethod": str(cluster.get("summaryMethod", "mean")),
        "fingerprint": str(cluster.get("fingerprint", "")),
        "analysisAtUpload": normalize_analysis_snapshot(cluster.get("analysisAtUpload")),
        "peerGroupSizeAtUpload": cluster.get("peerGroupSizeAtUpload"),
        "engineAtUpload": cluster.get("engineAtUpload"),
        "heterogeneityAtUpload": cluster.get("heterogeneityAtUpload"),
        "confidenceAtUpload": cluster.get("confidenceAtUpload"),
        "D2AtUpload": cluster.get("D2AtUpload"),
        "pValueAtUpload": cluster.get("pValueAtUpload"),
    }
