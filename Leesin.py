from __future__ import annotations

import argparse
import ipaddress
import itertools
import json
import math
import os
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import numpy as np
    from numpy.linalg import LinAlgError, pinv
    from scipy import stats
    from sklearn.covariance import LedoitWolf
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency. Run this app with your project's virtual environment interpreter, for example:\n"
        r"  .\Lee_sin.venv\Scripts\python.exe .\Leesin.py"
        "\nOr install requirements in the active environment with:\n"
        r"  python -m pip install -r requirements.txt"
    ) from exc


K_M = 10.0
GOAL_STORE_PATH = Path(__file__).with_name("goal_store.json")


@dataclass
class ExperimentConfig:
    axis_names: list[str]
    domain_range: list[tuple[float, float]]
    resolution: list[float]
    K_m: float = K_M

    def __post_init__(self) -> None:
        if not self.axis_names:
            raise ValueError("At least one axis is required.")
        if len(self.axis_names) != len(self.domain_range) or len(self.axis_names) != len(self.resolution):
            raise ValueError("axis_names, domain_range, and resolution must have the same length.")
        for index, ((lo, hi), step) in enumerate(zip(self.domain_range, self.resolution)):
            if hi <= lo:
                raise ValueError(f"Axis {index} has an invalid domain range.")
            if step <= 0:
                raise ValueError(f"Axis {index} must have a positive resolution.")


@dataclass
class DiagnosisResult:
    engine: str
    is_normal: bool | None
    center: np.ndarray
    D2: float
    p_value: float
    heterogeneity: float
    contributions: np.ndarray
    sample_size_Z: float
    coverage_C: float
    equitability_E: float
    w_eff: float
    confidence: float
    total_bins: int
    occupied_bins: int
    mardia_skew_stat: float | None = None
    mardia_skew_pval: float | None = None
    mardia_kurt_stat: float | None = None
    mardia_kurt_pval: float | None = None
    b2p: float | None = None

    def to_payload(self, axis_names: list[str]) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "is_normal": self.is_normal,
            "center": [round(float(value), 6) for value in self.center],
            "D2": round(float(self.D2), 6),
            "p_value": round(float(self.p_value), 6),
            "heterogeneity": round(float(self.heterogeneity), 6),
            "contributions": [
                {"axis": axis_name, "percent": round(float(percent), 4)}
                for axis_name, percent in zip(axis_names, self.contributions)
            ],
            "sample_size_Z": round(float(self.sample_size_Z), 6),
            "coverage_C": round(float(self.coverage_C), 6),
            "equitability_E": round(float(self.equitability_E), 6),
            "w_eff": round(float(self.w_eff), 6),
            "confidence": round(float(self.confidence), 6),
            "total_bins": self.total_bins,
            "occupied_bins": self.occupied_bins,
            "mardia_skew_stat": None if self.mardia_skew_stat is None else round(float(self.mardia_skew_stat), 6),
            "mardia_skew_pval": None if self.mardia_skew_pval is None else round(float(self.mardia_skew_pval), 6),
            "mardia_kurt_stat": None if self.mardia_kurt_stat is None else round(float(self.mardia_kurt_stat), 6),
            "mardia_kurt_pval": None if self.mardia_kurt_pval is None else round(float(self.mardia_kurt_pval), 6),
            "b2p": None if self.b2p is None else round(float(self.b2p), 6),
        }


class BinGridTracker:
    def __init__(self, domain_range: list[tuple[float, float]], resolution: list[float]):
        self.domain_range = domain_range
        self.resolution = resolution
        self._bins: dict[str, int] = {}

    def _bin_index(self, value: float, lo: float, hi: float, step: float) -> int:
        clipped = min(max(value, lo), hi - np.finfo(float).eps)
        return int(np.floor((clipped - lo) / step))

    def add(self, row: np.ndarray) -> None:
        indices = []
        for value, (lo, hi), step in zip(row, self.domain_range, self.resolution):
            indices.append(self._bin_index(float(value), lo, hi, step))
        key = json.dumps(indices)
        self._bins[key] = self._bins.get(key, 0) + 1

    @property
    def total_bins(self) -> int:
        total = 1
        for (lo, hi), step in zip(self.domain_range, self.resolution):
            total *= max(1, int(math.ceil((hi - lo) / step)))
        return total

    @property
    def occupied_bins(self) -> int:
        return len(self._bins)

    @property
    def coverage(self) -> float:
        return self.occupied_bins / self.total_bins if self.total_bins else 0.0

    @property
    def equitability(self) -> float:
        occupied = self.occupied_bins
        if occupied <= 1:
            return 0.0
        counts = np.array(list(self._bins.values()), dtype=float)
        proportions = counts / counts.sum()
        entropy = -np.sum(proportions * np.log(proportions + 1e-12))
        return float(entropy / np.log(occupied))


def spatial_median(X: np.ndarray, max_iter: int = 300, tol: float = 1e-7) -> np.ndarray:
    center = np.median(X, axis=0).astype(float)
    for _ in range(max_iter):
        diff = X - center
        distances = np.linalg.norm(diff, axis=1)
        zero_mask = distances < 1e-12
        if np.any(zero_mask):
            return X[zero_mask][0].astype(float)
        weights = 1.0 / distances
        next_center = (weights[:, None] * X).sum(axis=0) / weights.sum()
        if np.linalg.norm(next_center - center) < tol:
            return next_center
        center = next_center
    return center


def sscm(X: np.ndarray, center: np.ndarray) -> np.ndarray:
    shifted = X - center
    norms = np.linalg.norm(shifted, axis=1, keepdims=True)
    unit_vectors = np.divide(shifted, np.where(norms < 1e-12, 1.0, norms))
    return (unit_vectors.T @ unit_vectors) / len(X)


def mardia_test(X: np.ndarray) -> dict[str, float | bool]:
    n, p = X.shape
    center = X.mean(axis=0)
    covariance = np.cov(X, rowvar=False)
    try:
        covariance_inv = np.linalg.inv(covariance)
    except LinAlgError:
        covariance_inv = pinv(covariance)

    Xc = X - center
    gram = Xc @ covariance_inv @ Xc.T

    b1p = float(np.mean(gram**3))
    skew_stat = n * b1p / 6.0
    df_skew = p * (p + 1) * (p + 2) / 6
    skew_pval = float(1.0 - stats.chi2.cdf(skew_stat, df=df_skew))

    mahal_sq = np.sum((Xc @ covariance_inv) * Xc, axis=1)
    b2p = float(np.mean(mahal_sq**2))
    kurt_mean = p * (p + 2)
    kurt_var = 8 * p * (p + 2) / n
    kurt_stat = float((b2p - kurt_mean) / np.sqrt(kurt_var))
    kurt_pval = float(2.0 * (1.0 - stats.norm.cdf(abs(kurt_stat))))

    return {
        "skew_stat": skew_stat,
        "skew_pval": skew_pval,
        "kurt_stat": kurt_stat,
        "kurt_pval": kurt_pval,
        "b2p": b2p,
        "is_normal": bool(skew_pval > 0.05 and kurt_pval > 0.05),
    }


class DataQualityAnalyzer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.p = len(config.axis_names)
        self._peers: list[np.ndarray] = []
        self._grid = BinGridTracker(config.domain_range, config.resolution)

    def add_peers(self, X: np.ndarray | list[list[float]]) -> None:
        rows = np.asarray(X, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != self.p:
            raise ValueError(f"Peer data must have shape (n, {self.p}).")
        for row in rows:
            vector = np.asarray(row, dtype=float)
            self._peers.append(vector)
            self._grid.add(vector)

    def _guard_dimensions(self, count: int) -> None:
        if count <= self.p + 1:
            raise ValueError(
                f"N={count} is too small for a stable multivariate distance with p={self.p}. "
                "Add more peer clusters or lower the resolution."
            )

    def _select_engine(self, X: np.ndarray) -> tuple[str, bool | None, dict[str, float | bool]]:
        if len(X) < 10:
            return "spatial_rank", None, {}
        mardia = mardia_test(X)
        if bool(mardia["is_normal"]):
            return "mahalanobis", True, mardia
        return "sscm", False, mardia

    def _w_eff(self, engine: str, b2p: float | None) -> float:
        if engine == "spatial_rank":
            return 0.5
        if b2p is None or b2p <= 0:
            return 1.0
        return float(min(1.0, self.p * (self.p + 2) / b2p))

    def _compute_heterogeneity(
        self,
        x_target: np.ndarray,
        X: np.ndarray,
        engine: str,
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        count, p = X.shape

        if engine == "spatial_rank":
            center = spatial_median(X)
            peer_distances = np.linalg.norm(X - center, axis=1)
            target_distance = float(np.linalg.norm(x_target - center))
            rank_fraction = float(np.sum(peer_distances < target_distance) / count)
            D2 = float(rank_fraction * stats.chi2.ppf(0.99, df=p))
            p_value = float(max(0.0, 1.0 - rank_fraction))
            diff = np.abs(x_target - center)
            contributions = (diff / (diff.sum() + 1e-12)) * 100
            return center, D2, p_value, contributions

        if engine == "mahalanobis":
            model = LedoitWolf().fit(X)
            center = model.location_
            covariance = model.covariance_
        else:
            center = spatial_median(X)
            covariance = sscm(X, center)

        try:
            covariance_inv = np.linalg.inv(covariance)
        except LinAlgError:
            covariance_inv = pinv(covariance)

        diff = x_target - center
        D2 = float(diff @ covariance_inv @ diff)
        p_value = float(1.0 - stats.chi2.cdf(D2, df=p))
        contribution_raw = diff * (covariance_inv @ diff)
        contributions = (np.abs(contribution_raw) / (np.abs(contribution_raw).sum() + 1e-12)) * 100
        return center, D2, p_value, contributions

    def diagnose(self, x_target: np.ndarray | list[float]) -> DiagnosisResult:
        if not self._peers:
            raise ValueError("Peer group is empty.")

        target = np.asarray(x_target, dtype=float)
        if target.shape != (self.p,):
            raise ValueError(f"Target row must have shape ({self.p},).")

        X = np.asarray(self._peers, dtype=float)
        count = len(X)

        engine_preview, _, _ = self._select_engine(X)
        if engine_preview != "spatial_rank":
            self._guard_dimensions(count)

        engine, is_normal, mardia = self._select_engine(X)
        b2p = float(mardia["b2p"]) if "b2p" in mardia else None
        w_eff = self._w_eff(engine, b2p)
        center, D2, p_value, contributions = self._compute_heterogeneity(target, X, engine)

        sample_size_Z = count / (count + self.config.K_m)
        coverage_C = self._grid.coverage
        equitability_E = self._grid.equitability
        confidence = float((sample_size_Z * coverage_C * equitability_E) ** (1.0 / 3.0) * w_eff)

        return DiagnosisResult(
            engine=engine,
            is_normal=is_normal,
            center=center,
            D2=D2,
            p_value=p_value,
            heterogeneity=1.0 - p_value,
            contributions=contributions,
            sample_size_Z=sample_size_Z,
            coverage_C=coverage_C,
            equitability_E=equitability_E,
            w_eff=w_eff,
            confidence=confidence,
            total_bins=self._grid.total_bins,
            occupied_bins=self._grid.occupied_bins,
            mardia_skew_stat=float(mardia["skew_stat"]) if "skew_stat" in mardia else None,
            mardia_skew_pval=float(mardia["skew_pval"]) if "skew_pval" in mardia else None,
            mardia_kurt_stat=float(mardia["kurt_stat"]) if "kurt_stat" in mardia else None,
            mardia_kurt_pval=float(mardia["kurt_pval"]) if "kurt_pval" in mardia else None,
            b2p=b2p,
        )


DEFAULT_GOALS = [
    {
        "id": "goal_thermal",
        "K_m": K_M,
        "name": "고온 유량 품질 인증",
        "axes": [
            {"name": "temperature", "unit": "℃", "domainMin": 0.0, "domainMax": 200.0, "resolution": 10.0},
            {"name": "pressure", "unit": "bar", "domainMin": 0.0, "domainMax": 100.0, "resolution": 5.0},
            {"name": "flow_rate", "unit": "kg/h", "domainMin": 0.0, "domainMax": 50.0, "resolution": 2.5},
        ],
    },
    {
        "id": "goal_vacuum",
        "K_m": K_M,
        "name": "진공 유지 품질 인증",
        "axes": [
            {"name": "vacuum_level", "unit": "kPa", "domainMin": 0.0, "domainMax": 100.0, "resolution": 5.0},
            {"name": "hold_time", "unit": "s", "domainMin": 0.0, "domainMax": 300.0, "resolution": 15.0},
            {"name": "leak_rate", "unit": "sccm", "domainMin": 0.0, "domainMax": 20.0, "resolution": 1.0},
        ],
    },
]


PEER_GROUP_LIBRARY = {
    ("temperature", "pressure", "flow_rate"): [
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
    ("vacuum_level", "hold_time", "leak_rate"): [
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


def _default_goal_payload() -> list[dict[str, Any]]:
    return json.loads(json.dumps(DEFAULT_GOALS, ensure_ascii=False))


def load_goal_store() -> list[dict[str, Any]]:
    if GOAL_STORE_PATH.exists():
        raw_goals = json.loads(GOAL_STORE_PATH.read_text(encoding="utf-8"))
    else:
        raw_goals = _default_goal_payload()

    normalized_goals: list[dict[str, Any]] = []
    for goal in raw_goals:
        try:
            normalized_goals.append(validate_goal(goal))
        except (TypeError, ValueError, KeyError):
            continue

    if not normalized_goals:
        normalized_goals = [validate_goal(goal) for goal in _default_goal_payload()]

    if normalized_goals != raw_goals:
        save_goal_store(normalized_goals)
    return normalized_goals


def save_goal_store(goals: list[dict[str, Any]]) -> None:
    GOAL_STORE_PATH.write_text(json.dumps(goals, ensure_ascii=False, indent=2), encoding="utf-8")


def axis_signature(axis_names: list[str]) -> tuple[str, ...]:
    return tuple(name.strip().lower() for name in axis_names)


def axis_subset_key(axis_names: list[str]) -> str:
    return "|".join(axis_signature(axis_names))


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
    for axis in axes:
        axis_name = str(axis.get("name", "")).strip()
        if not axis_name:
            raise ValueError("Axis name is required.")
        unit = str(axis.get("unit", "")).strip()
        domain_min = float(axis.get("domainMin"))
        domain_max = float(axis.get("domainMax"))
        resolution = float(axis.get("resolution"))
        normalized_axes.append(
            {
                "name": axis_name,
                "unit": unit,
                "domainMin": domain_min,
                "domainMax": domain_max,
                "resolution": resolution,
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


def build_bootstrap_payload(admin_allowed: bool) -> dict[str, Any]:
    goals = load_goal_store()
    peer_counts = {}
    peer_subset_counts = {}
    for goal in goals:
        try:
            peer_counts[goal["id"]] = int(len(pick_peer_group(goal)))
        except ValueError:
            peer_counts[goal["id"]] = 0
        peer_subset_counts[goal["id"]] = peer_group_subset_counts(goal)
    return {
        "adminAllowed": admin_allowed,
        "goals": goals,
        "peerCounts": peer_counts,
        "peerSubsetCounts": peer_subset_counts,
        "acceptedUploadTypes": [".csv", ".tsv", ".txt"],
        "stateShape": {
            "admin": ["selectedGoalId", "draftGoal"],
            "user": ["selectedGoalId", "fileName", "headers", "rows", "selectedAxes", "axisMapping", "primaryKey"],
            "report": ["status", "result", "meta", "summary", "confidenceReasons"],
        },
        "componentStructure": {
            "DashboardShell": ["TopActionBar", "AdminSettingsModal", "ColumnMapper", "CertificateReport"],
            "AdminSettingsModal": ["GoalSelector", "GoalEditorForm", "AxisTableEditor"],
            "TopActionBar": ["GoalDropdown", "UploadPanel", "AnalyzeButton"],
            "CertificateReport": ["ScoreCards", "ContributionChart", "SampleSizeList", "CoverageLines", "EquitabilityChart"],
        },
    }


def goal_subset(goal: dict[str, Any], selected_axis_names: list[str] | None = None) -> dict[str, Any]:
    if not selected_axis_names:
        return {
            "id": goal["id"],
            "name": goal["name"],
            "K_m": float(goal.get("K_m", K_M)),
            "axes": [dict(axis) for axis in goal["axes"]],
        }
    requested = {name.strip() for name in selected_axis_names if str(name).strip()}
    axes = [dict(axis) for axis in goal["axes"] if axis["name"] in requested]
    if not axes:
        raise ValueError("선택된 축이 없습니다. 분석에 포함할 Axis를 하나 이상 체크하세요.")
    return {
        "id": goal["id"],
        "name": goal["name"],
        "K_m": float(goal.get("K_m", K_M)),
        "axes": axes,
    }


def pick_peer_group(goal: dict[str, Any], selected_axis_names: list[str] | None = None) -> np.ndarray:
    signature = axis_signature([axis["name"] for axis in goal["axes"]])
    peer_rows = PEER_GROUP_LIBRARY.get(signature)
    if peer_rows is None:
        raise ValueError("No automatic peer group could be matched for this Experiment Goal.")
    peer_group = np.asarray(peer_rows, dtype=float)
    if not selected_axis_names:
        return peer_group
    name_to_index = {axis["name"]: index for index, axis in enumerate(goal["axes"])}
    try:
        selected_indices = [name_to_index[name] for name in selected_axis_names]
    except KeyError as exc:
        raise ValueError(f"Selected axis '{exc.args[0]}' does not belong to this Goal.") from exc
    return peer_group[:, selected_indices]


def peer_group_subset_counts(goal: dict[str, Any]) -> dict[str, int]:
    try:
        peer_group = pick_peer_group(goal)
    except ValueError:
        return {}

    axis_names = [axis["name"] for axis in goal["axes"]]
    axis_index = {axis_name: index for index, axis_name in enumerate(axis_names)}
    counts: dict[str, int] = {}

    for size in range(1, len(axis_names) + 1):
        for subset in itertools.combinations(axis_names, size):
            subset_matrix = peer_group[:, [axis_index[name] for name in subset]]
            counts[axis_subset_key(list(subset))] = int(np.sum(np.all(np.isfinite(subset_matrix), axis=1)))
    return counts


def build_target_vector(
    rows: list[list[Any]],
    axis_mapping: dict[str, str],
    goal: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    axis_names = [axis["name"] for axis in goal["axes"]]
    headers = list(rows[0].keys()) if rows else []
    matrix = []
    for row in rows:
        vector = []
        for axis_name in axis_names:
            column_name = axis_mapping.get(axis_name)
            if column_name not in row:
                raise ValueError(f"Axis '{axis_name}' is not mapped to a valid column.")
            value = row[column_name]
            vector.append(float(value))
        matrix.append(vector)
    values = np.asarray(matrix, dtype=float)
    return values.mean(axis=0), {"row_count": int(len(values)), "columns": headers}


def confidence_reasons(result: DiagnosisResult) -> list[dict[str, Any]]:
    reasons = [
        {
            "label": "Sample Size",
            "score": round(float(result.sample_size_Z), 4),
            "impact": "down" if result.sample_size_Z < 0.6 else "stable",
            "message": "같은 실험 목적의 비교 군집 수가 충분한지 반영합니다.",
        },
        {
            "label": "Coverage",
            "score": round(float(result.coverage_C), 4),
            "impact": "down" if result.coverage_C < 0.3 else "stable",
            "message": "도메인 전체 대비 실제 점유된 영역의 폭을 반영합니다.",
        },
        {
            "label": "Equitability",
            "score": round(float(result.equitability_E), 4),
            "impact": "down" if result.equitability_E < 0.5 else "stable",
            "message": "점유된 영역 안에서 샘플이 균등하게 분포하는지 반영합니다.",
        },
    ]
    if result.w_eff < 0.7:
        reasons.append(
            {
                "label": "Engine Robustness",
                "score": round(float(result.w_eff), 4),
                "impact": "down",
                "message": "Mardia 첨도 기반 효율 가중치가 신뢰도를 추가로 낮췄습니다.",
            }
        )
    return reasons


def build_axis_distribution(
    values: np.ndarray,
    domain_min: float,
    domain_max: float,
    resolution: float,
) -> dict[str, Any]:
    total_bins = max(1, int(math.ceil((domain_max - domain_min) / resolution)))
    counts = [0 for _ in range(total_bins)]
    for raw_value in values:
        clipped = min(max(float(raw_value), domain_min), domain_max - np.finfo(float).eps)
        index = int(np.floor((clipped - domain_min) / resolution))
        index = max(0, min(total_bins - 1, index))
        counts[index] += 1
    bins = [
        {
            "index": index,
            "start": round(float(domain_min + index * resolution), 6),
            "end": round(float(min(domain_min + (index + 1) * resolution, domain_max)), 6),
            "count": int(count),
        }
        for index, count in enumerate(counts)
    ]
    occupied_bins = [item for item in bins if item["count"] > 0]
    return {
        "totalBins": total_bins,
        "occupiedBins": len(occupied_bins),
        "bins": bins,
        "occupiedBinsDetail": occupied_bins,
    }


def axis_display_label(axis: dict[str, Any]) -> str:
    unit = str(axis.get("unit", "")).strip()
    return f"{axis['name']} ({unit})" if unit else str(axis["name"])


def build_report_visualizations(
    goal: dict[str, Any],
    peer_group: np.ndarray,
    target_vector: np.ndarray,
    result: DiagnosisResult,
) -> dict[str, Any]:
    axes = goal["axes"]
    goal_k_m = float(goal.get("K_m", K_M))
    sample_size_items = []
    coverage_axes = []
    equitability_axes = []

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
        "sampleSize": {
            "peerGroupCount": int(len(peer_group)),
            "z": round(float(result.sample_size_Z), 6),
            "items": sample_size_items,
        },
        "coverage": {
            "score": round(float(result.coverage_C), 6),
            "axes": coverage_axes,
        },
        "equitability": {
            "score": round(float(result.equitability_E), 6),
            "status": "balanced" if result.equitability_E >= 0.5 else "imbalanced",
            "axes": equitability_axes,
        },
    }


def analyze_request(payload: dict[str, Any]) -> dict[str, Any]:
    goals = load_goal_store()
    goal_id = payload.get("goalId")
    goal = next((item for item in goals if item["id"] == goal_id), None)
    if goal is None:
        raise ValueError("Selected Experiment Goal does not exist.")

    rows = payload.get("rows", [])
    if not rows:
        raise ValueError("Uploaded dataset is empty.")

    axis_mapping = payload.get("axisMapping", {})
    if not isinstance(axis_mapping, dict):
        raise ValueError("Axis mapping is missing.")

    selected_axis_names = payload.get("selectedAxes")
    if selected_axis_names is None:
        selected_axis_names = [axis["name"] for axis in goal["axes"]]
    if not isinstance(selected_axis_names, list):
        raise ValueError("Selected axes must be provided as a list.")
    selected_goal = goal_subset(goal, [str(name) for name in selected_axis_names])

    primary_key = str(payload.get("primaryKey", "")).strip()
    if primary_key not in [axis["name"] for axis in selected_goal["axes"]]:
        raise ValueError("Primary Key는 선택된 축 중 하나여야 합니다.")

    config = ExperimentConfig(
        axis_names=[axis["name"] for axis in selected_goal["axes"]],
        domain_range=[(axis["domainMin"], axis["domainMax"]) for axis in selected_goal["axes"]],
        resolution=[axis["resolution"] for axis in selected_goal["axes"]],
        K_m=float(selected_goal.get("K_m", K_M)),
    )

    target_vector, dataset_meta = build_target_vector(rows, axis_mapping, selected_goal)
    if dataset_meta["row_count"] < len(selected_goal["axes"]) + 1:
        raise ValueError("데이터가 너무 적어 공분산 연산이 불가능합니다. 데이터를 추가하거나 축 선택을 줄이세요.")
    peer_group = pick_peer_group(goal, [axis["name"] for axis in selected_goal["axes"]])

    analyzer = DataQualityAnalyzer(config)
    analyzer.add_peers(peer_group)
    result = analyzer.diagnose(target_vector)

    summary = build_summary(result)
    summary.append(f"Primary Key는 '{primary_key}' 기준으로 기록되었고 peer group은 백엔드에서 자동 매칭되었습니다.")

    return {
        "meta": {
            "experiment_goal": goal["name"],
            "primary_key": primary_key,
            "target_rows": dataset_meta["row_count"],
            "uploaded_columns": dataset_meta["columns"],
            "peer_group_size": int(len(peer_group)),
            "axis_names": config.axis_names,
            "axes": selected_goal["axes"],
            "available_axes": goal["axes"],
            "config": asdict(config),
        },
        "result": result.to_payload(config.axis_names),
        "summary": summary,
        "confidenceReasons": confidence_reasons(result),
        "visualizations": build_report_visualizations(selected_goal, peer_group, target_vector, result),
    }


def build_summary(result: DiagnosisResult) -> list[str]:
    messages: list[str] = []
    if result.heterogeneity > 0.95 and result.confidence > 0.7:
        messages.append("이질성과 신뢰도가 모두 높습니다. 새로운 물리적 발견 가능성을 우선 검토할 수 있습니다.")
    elif result.heterogeneity > 0.95 and result.confidence <= 0.4:
        messages.append("타겟은 매우 특이하지만 현재 비교 군집의 신뢰도가 낮아 설계 오류나 데이터 부족 가능성도 큽니다.")
    elif result.heterogeneity <= 0.5:
        messages.append("타겟은 현재 비교 군집과 통계적으로 크게 다르지 않습니다.")
    else:
        messages.append("이질성은 존재하지만 추가 샘플 확보와 분포 검증이 함께 필요합니다.")

    if result.sample_size_Z < 0.5:
        messages.append("Sample Size가 부족합니다. 같은 실험 목적의 비교 군집을 더 확보하세요.")
    if result.coverage_C < 0.3:
        messages.append("Coverage가 낮습니다. 설정한 도메인 전체 대비 점유 범위가 좁습니다.")
    if result.equitability_E < 0.5:
        messages.append("Equitability가 낮습니다. 일부 구간에 데이터가 편중되어 있습니다.")
    if result.w_eff < 0.7:
        messages.append("비정규 특성이 강해 엔진 효율 가중치가 낮아졌습니다.")
    return messages


def build_summary(result: DiagnosisResult) -> list[str]:
    messages: list[str] = []
    if result.heterogeneity > 0.95 and result.confidence > 0.7:
        messages.append("이질성과 신뢰도가 모두 높습니다. 새로운 물리적 발견 가능성을 우선 검토할 수 있습니다.")
    elif result.heterogeneity > 0.95 and result.confidence <= 0.4:
        messages.append("타겟은 매우 특이하지만 현재 비교 군집의 신뢰도가 낮아 설계 오류나 데이터 부족 가능성도 큽니다.")
    elif result.heterogeneity <= 0.5:
        messages.append("타겟은 현재 비교 군집과 통계적으로 크게 다르지 않습니다.")
    else:
        messages.append("이질성은 존재하지만 추가 샘플 확보와 분포 검증이 함께 필요합니다.")

    if result.sample_size_Z < 0.5:
        messages.append("Sample Size가 부족합니다. 같은 실험 목적의 비교 군집을 더 확보하세요.")
    if result.coverage_C < 0.3:
        messages.append("Coverage가 낮습니다. 설정한 도메인 전체 대비 점유 범위가 좁습니다.")
    if result.equitability_E < 0.5:
        messages.append("Equitability가 낮습니다. 일부 구간에 데이터가 편중되어 있습니다.")
    if result.w_eff < 0.7:
        messages.append("비정규 특성이 강해 엔진 효율 가중치가 낮아졌습니다.")
    return messages


PAGE_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Data Quality Certification System</title>
  <style>
    :root {
      --bg: #f4ede2;
      --panel: rgba(255,255,255,0.86);
      --panel-strong: rgba(255,255,255,0.94);
      --ink: #14202b;
      --muted: #516171;
      --line: rgba(20,32,43,0.12);
      --accent: #b9471f;
      --accent-2: #15523b;
      --warn: #8c3f18;
      --shadow: 0 20px 48px rgba(62, 40, 24, 0.12);
      --radius: 22px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Segoe UI", "Pretendard", sans-serif;
      background:
        radial-gradient(circle at 0% 0%, rgba(185,71,31,0.14), transparent 26%),
        radial-gradient(circle at 100% 18%, rgba(21,82,59,0.14), transparent 24%),
        linear-gradient(180deg, #fbf7f1 0%, #eee2d2 100%);
      min-height: 100vh;
    }
    .wrap {
      width: min(1240px, calc(100% - 28px));
      margin: 24px auto 40px;
    }
    .hero {
      padding: 28px;
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,248,240,0.92));
      border: 1px solid rgba(185,71,31,0.12);
      box-shadow: var(--shadow);
    }
    .eyebrow {
      display: inline-block;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(185,71,31,0.12);
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    h1 {
      margin: 14px 0 8px;
      font-size: clamp(34px, 5.7vw, 64px);
      line-height: 0.94;
      letter-spacing: -0.05em;
    }
    .hero p {
      margin: 0;
      max-width: 820px;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.65;
    }
    .shell {
      display: grid;
      gap: 18px;
      margin-top: 20px;
    }
    .tabs {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .tab {
      width: auto;
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      background: rgba(21,82,59,0.1);
      color: var(--accent-2);
      font-weight: 800;
      cursor: pointer;
    }
    .tab.active {
      background: linear-gradient(135deg, #b9471f, #dd6b20);
      color: white;
      box-shadow: 0 12px 26px rgba(185,71,31,0.26);
    }
    .view {
      display: none;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .view.active { display: block; }
    .view-header {
      padding: 22px 24px 0;
    }
    .view-header h2 {
      margin: 0;
      font-size: 24px;
      letter-spacing: -0.03em;
    }
    .view-header p {
      margin: 8px 0 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 14px;
      max-width: 820px;
    }
    .view-body {
      padding: 20px 24px 26px;
      display: grid;
      gap: 18px;
    }
    .card {
      padding: 18px;
      border-radius: 18px;
      background: rgba(255,255,255,0.8);
      border: 1px solid rgba(20,32,43,0.08);
    }
    .card h3 {
      margin: 0 0 10px;
      font-size: 16px;
    }
    .grid-2 {
      display: grid;
      gap: 16px;
      grid-template-columns: 1fr 1fr;
    }
    .grid-3 {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(3, 1fr);
    }
    label {
      display: grid;
      gap: 8px;
      font-size: 12px;
      font-weight: 800;
      color: var(--muted);
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    input, select, button, textarea {
      width: 100%;
      border: 1px solid rgba(20,32,43,0.12);
      border-radius: 14px;
      padding: 12px 14px;
      background: var(--panel-strong);
      color: var(--ink);
      font: inherit;
      outline: none;
    }
    input:focus, select:focus, textarea:focus {
      border-color: rgba(185,71,31,0.45);
    }
    button {
      cursor: pointer;
      font-weight: 800;
    }
    .primary {
      background: linear-gradient(135deg, #b9471f, #dd6b20);
      color: white;
      border: 0;
      box-shadow: 0 12px 26px rgba(185,71,31,0.24);
    }
    .ghost {
      background: rgba(21,82,59,0.1);
      color: var(--accent-2);
    }
    .danger {
      background: rgba(140,63,24,0.1);
      color: var(--warn);
    }
    .axis-row {
      display: grid;
      gap: 10px;
      grid-template-columns: 1.2fr 1fr 1fr 1fr auto;
      align-items: end;
      margin-bottom: 10px;
    }
    .metric {
      border-radius: 18px;
      padding: 16px;
      background: rgba(255,255,255,0.76);
      border: 1px solid rgba(20,32,43,0.08);
    }
    .metric small {
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .metric strong {
      display: block;
      margin-top: 8px;
      font-size: 30px;
      letter-spacing: -0.05em;
    }
    .muted-box, pre {
      margin: 0;
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(255,255,255,0.62);
      border: 1px dashed rgba(20,32,43,0.12);
      color: var(--muted);
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.6;
    }
    .summary-list, .reason-list, .bars {
      display: grid;
      gap: 10px;
    }
    .pill, .reason {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(21,82,59,0.1);
      color: var(--accent-2);
    }
    .reason.down {
      background: rgba(140,63,24,0.1);
      color: var(--warn);
    }
    .bar-row {
      display: grid;
      gap: 6px;
    }
    .bar-meta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 14px;
    }
    .bar-track {
      width: 100%;
      height: 12px;
      border-radius: 999px;
      background: rgba(20,32,43,0.08);
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #15523b, #2f855a, #dd6b20);
    }
    .mapping-grid {
      display: grid;
      gap: 10px;
    }
    .mapping-row {
      display: grid;
      gap: 10px;
      grid-template-columns: 1fr 1fr;
      align-items: center;
    }
    .notice {
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(185,71,31,0.1);
      color: var(--accent);
      border: 1px solid rgba(185,71,31,0.12);
      line-height: 1.6;
    }
    .error {
      display: none;
      padding: 14px 16px;
      border-radius: 14px;
      background: rgba(140,63,24,0.1);
      color: var(--warn);
      border: 1px solid rgba(140,63,24,0.14);
      line-height: 1.6;
    }
    @media (max-width: 960px) {
      .grid-2, .grid-3, .axis-row, .mapping-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <span class="eyebrow">Three View Workflow</span>
      <h1>관리자 설정, 일반 업로드, 결과 리포트를 완전히 분리한 데이터 품질 인증 UI</h1>
      <p>
        일반 사용자 화면에는 peer group 선택, 엔진 선택, K_m 같은 내부 로직을 노출하지 않습니다.
        관리자는 실험 목적과 축 정의만 설정하고, 일반 작업자는 goal 선택과 파일 업로드, 컬럼 매칭만 수행합니다.
      </p>
    </section>

    <div class="shell">
      <div class="tabs">
        <button class="tab active" data-view="admin">View 1. Admin Setup</button>
        <button class="tab" data-view="user">View 2. General User</button>
        <button class="tab" data-view="report">View 3. Result Report</button>
      </div>

      <section class="view active" id="view-admin">
        <div class="view-header">
          <h2>Admin Setup View</h2>
          <p>사양서 Step 0에 해당합니다. 관리자는 Experiment Goal, Axis, Domain Range, Resolution만 정의합니다. K_m과 엔진 선택은 백엔드에서 숨겨집니다.</p>
        </div>
        <div class="view-body">
          <div class="grid-2">
            <div class="card">
              <h3>Goal Selector</h3>
              <label>Existing Goal
                <select id="admin-goal-select"></select>
              </label>
              <div class="grid-2" style="margin-top:12px;">
                <button class="ghost" id="admin-new-goal" type="button">새 Goal</button>
                <button class="danger" id="admin-delete-goal" type="button">선택 Goal 삭제</button>
              </div>
            </div>
            <div class="card">
              <h3>State Logic</h3>
              <div class="muted-box" id="state-shape-box"></div>
            </div>
          </div>

          <div class="card">
            <h3>Goal Editor Form</h3>
            <div class="grid-2">
              <label>Experiment Goal
                <input id="admin-goal-name" placeholder="예: 고온 유량 품질 인증">
              </label>
              <label>Preview
                <div class="muted-box" id="admin-goal-preview"></div>
              </label>
            </div>
            <div style="margin-top:14px;">
              <h3>Axis Table Editor</h3>
              <div id="axis-editor"></div>
              <button class="ghost" id="axis-add-button" type="button">Axis 추가</button>
            </div>
            <div class="grid-2" style="margin-top:14px;">
              <button class="primary" id="admin-save-goal" type="button">Goal 저장</button>
              <button class="ghost" id="admin-load-template" type="button">기본 템플릿 불러오기</button>
            </div>
          </div>
        </div>
      </section>

      <section class="view" id="view-user">
        <div class="view-header">
          <h2>General User View</h2>
          <p>사양서 Step 1에 해당합니다. 일반 사용자는 Experiment Goal 선택, Target Data 업로드, 컬럼 매칭과 Primary Key 지정만 수행합니다. Peer Group과 엔진은 서버가 자동으로 처리합니다.</p>
        </div>
        <div class="view-body">
          <div class="card">
            <h3>Goal Dropdown</h3>
            <label>Experiment Goal
              <select id="user-goal-select"></select>
            </label>
            <div class="notice" id="goal-background-summary"></div>
          </div>

          <div class="card">
            <h3>Upload Panel</h3>
            <label>Target Data Upload
              <input id="file-input" type="file" accept=".csv,.tsv,.txt">
            </label>
            <div class="notice">현재 구현은 CSV/TSV/TXT를 지원합니다. 업로드 파일은 브라우저에서 읽은 뒤 백엔드 분석 요청으로 전달됩니다.</div>
            <div class="muted-box" id="file-meta-box">아직 업로드된 파일이 없습니다.</div>
          </div>

          <div class="card">
            <h3>Column Mapper</h3>
            <div id="mapping-box" class="mapping-grid"></div>
          </div>

          <div class="card">
            <h3>Primary Key Selector</h3>
            <label>Primary Key
              <select id="primary-key-select"></select>
            </label>
            <button class="primary" id="run-analysis" type="button">백엔드 자동 분석 실행</button>
          </div>
          <div class="error" id="user-error-box"></div>
        </div>
      </section>

      <section class="view" id="view-report">
        <div class="view-header">
          <h2>Result Report View</h2>
          <p>사양서 Step 6에 해당합니다. 이질성, 축별 기여도, 최종 신뢰도와 감점 원인을 시각화합니다. 사용자는 내부 엔진 세부 옵션을 보지 않고도 결과의 정당성을 파악할 수 있습니다.</p>
        </div>
        <div class="view-body">
          <div class="grid-3" id="report-metrics"></div>
          <div class="card">
            <h3>이질성 요약</h3>
            <div class="summary-list" id="summary-list"></div>
          </div>
          <div class="card">
            <h3>축별 기여도</h3>
            <div class="bars" id="contribution-bars"></div>
          </div>
          <div class="card">
            <h3>신뢰도 감점 원인 분석</h3>
            <div class="reason-list" id="confidence-reasons"></div>
          </div>
          <div class="card">
            <h3>상세 메타</h3>
            <pre id="report-meta">아직 분석 결과가 없습니다.</pre>
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const bootstrap = __BOOTSTRAP__;
    const adminAllowed = Boolean(bootstrap.adminAllowed);
    const state = {
      view: adminAllowed ? 'admin' : 'user',
      goals: bootstrap.goals,
      admin: {
        selectedGoalId: bootstrap.goals[0]?.id || null,
        draftGoal: JSON.parse(JSON.stringify(bootstrap.goals[0] || { id: '', name: '', axes: [] })),
      },
      user: {
        selectedGoalId: bootstrap.goals[0]?.id || null,
        fileName: null,
        headers: [],
        rows: [],
        axisMapping: {},
        primaryKey: '',
      },
      report: {
        status: 'idle',
        result: null,
        meta: null,
        summary: [],
        confidenceReasons: [],
      },
    };

    function getGoalById(id) {
      return state.goals.find(goal => goal.id === id);
    }

    function clone(data) {
      return JSON.parse(JSON.stringify(data));
    }

    function setView(nextView) {
      state.view = nextView;
      document.querySelectorAll('.tab').forEach(tab => tab.classList.toggle('active', tab.dataset.view === nextView));
      document.querySelectorAll('.view').forEach(view => view.classList.toggle('active', view.id === `view-${nextView}`));
    }

    function renderStateShape() {
      document.getElementById('state-shape-box').textContent =
        `Admin State\\n${bootstrap.stateShape.admin.join(', ')}\\n\\n` +
        `User State\\n${bootstrap.stateShape.user.join(', ')}\\n\\n` +
        `Report State\\n${bootstrap.stateShape.report.join(', ')}`;
    }

    function renderGoalSelectors() {
      const options = state.goals.map(goal => `<option value="${goal.id}">${goal.name}</option>`).join('');
      document.getElementById('admin-goal-select').innerHTML = options;
      document.getElementById('user-goal-select').innerHTML = options;
      document.getElementById('admin-goal-select').value = state.admin.selectedGoalId || '';
      document.getElementById('user-goal-select').value = state.user.selectedGoalId || '';
    }

    function renderAdminDraft() {
      const draft = state.admin.draftGoal;
      document.getElementById('admin-goal-name').value = draft.name || '';
      const axisEditor = document.getElementById('axis-editor');
      axisEditor.innerHTML = draft.axes.map((axis, index) => `
        <div class="axis-row">
          <label>Axis
            <input data-role="axis-name" data-index="${index}" value="${axis.name}">
          </label>
          <label>Domain Min
            <input data-role="axis-min" data-index="${index}" type="number" value="${axis.domainMin}">
          </label>
          <label>Domain Max
            <input data-role="axis-max" data-index="${index}" type="number" value="${axis.domainMax}">
          </label>
          <label>Resolution
            <input data-role="axis-resolution" data-index="${index}" type="number" value="${axis.resolution}" step="any">
          </label>
          <button class="danger" type="button" data-role="axis-remove" data-index="${index}">삭제</button>
        </div>
      `).join('');
      const preview = [
        `Experiment Goal: ${draft.name || '-'}`,
        `Axis Count: ${draft.axes.length}`,
        ...draft.axes.map(axis => `${axis.name} | range [${axis.domainMin}, ${axis.domainMax}] | resolution ${axis.resolution}`)
      ].join('\\n');
      document.getElementById('admin-goal-preview').textContent = preview;
    }

    function renderUserGoalSummary() {
      const goal = getGoalById(state.user.selectedGoalId);
      if (!goal) return;
      const axisNames = goal.axes.map(axis => axis.name).join(', ');
      const rangeText = goal.axes.map(axis => `${axis.name}: [${axis.domainMin}, ${axis.domainMax}] / Δ ${axis.resolution}`).join(' | ');
      document.getElementById('goal-background-summary').textContent =
        `선택된 Goal: ${goal.name}. 백그라운드로 로드된 Axis는 ${axisNames}입니다. Domain Range와 Resolution은 관리자 설정값을 그대로 사용합니다. ${rangeText}`;
    }

    function renderMappingControls() {
      const goal = getGoalById(state.user.selectedGoalId);
      const headers = state.user.headers;
      const mappingBox = document.getElementById('mapping-box');
      if (!goal || headers.length === 0) {
        mappingBox.innerHTML = `<div class="muted-box">파일을 업로드하면 Axis 별 컬럼 매칭 컴포넌트가 여기 나타납니다.</div>`;
        document.getElementById('primary-key-select').innerHTML = '';
        return;
      }
      mappingBox.innerHTML = goal.axes.map(axis => `
        <div class="mapping-row">
          <div class="muted-box">${axis.name}</div>
          <label>
            <select data-axis-map="${axis.name}">
              ${headers.map(header => `<option value="${header}" ${state.user.axisMapping[axis.name] === header ? 'selected' : ''}>${header}</option>`).join('')}
            </select>
          </label>
        </div>
      `).join('');
      document.getElementById('primary-key-select').innerHTML = goal.axes
        .map(axis => `<option value="${axis.name}" ${state.user.primaryKey === axis.name ? 'selected' : ''}>${axis.name}</option>`)
        .join('');
    }

    function renderReport() {
      const metrics = document.getElementById('report-metrics');
      if (!state.report.result) {
        metrics.innerHTML = `
          <div class="metric"><small>Status</small><strong>Idle</strong></div>
          <div class="metric"><small>Heterogeneity</small><strong>-</strong></div>
          <div class="metric"><small>Confidence</small><strong>-</strong></div>
        `;
        document.getElementById('summary-list').innerHTML = `<div class="pill">아직 분석 결과가 없습니다.</div>`;
        document.getElementById('contribution-bars').innerHTML = `<div class="muted-box">축별 기여도 그래프가 여기 표시됩니다.</div>`;
        document.getElementById('confidence-reasons').innerHTML = `<div class="reason">Sample, Coverage, Equitability 기반 원인 분석이 여기 표시됩니다.</div>`;
        document.getElementById('report-meta').textContent = '아직 분석 결과가 없습니다.';
        return;
      }
      const result = state.report.result;
      metrics.innerHTML = `
        <div class="metric"><small>Engine</small><strong>${result.engine.toUpperCase()}</strong></div>
        <div class="metric"><small>Heterogeneity</small><strong>${result.heterogeneity.toFixed(4)}</strong></div>
        <div class="metric"><small>Confidence</small><strong>${result.confidence.toFixed(4)}</strong></div>
      `;
      document.getElementById('summary-list').innerHTML = state.report.summary.map(item => `<div class="pill">${item}</div>`).join('');
      document.getElementById('contribution-bars').innerHTML = result.contributions.map(item => `
        <div class="bar-row">
          <div class="bar-meta">
            <span>${item.axis}</span>
            <strong>${item.percent.toFixed(2)}%</strong>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${Math.max(2, item.percent)}%"></div>
          </div>
        </div>
      `).join('');
      document.getElementById('confidence-reasons').innerHTML = state.report.confidenceReasons.map(item => `
        <div class="reason ${item.impact === 'down' ? 'down' : ''}">
          <strong>${item.label}: ${item.score.toFixed(4)}</strong><br>${item.message}
        </div>
      `).join('');
      document.getElementById('report-meta').textContent = JSON.stringify({ meta: state.report.meta, result: state.report.result }, null, 2);
    }

    function refreshAll() {
      const adminTab = document.querySelector('.tab[data-view="admin"]');
      if (!adminAllowed) {
        if (adminTab) adminTab.style.display = 'none';
        state.view = 'user';
      } else if (adminTab) {
        adminTab.style.display = '';
      }
      renderStateShape();
      renderGoalSelectors();
      renderAdminDraft();
      renderUserGoalSummary();
      renderMappingControls();
      renderReport();
      setView(state.view);
    }

    function newDraftGoal() {
      state.admin.selectedGoalId = null;
      state.admin.draftGoal = {
        id: `goal_${Date.now()}`,
        name: '',
        axes: [{ name: '', domainMin: 0, domainMax: 100, resolution: 1 }],
      };
      renderAdminDraft();
    }

    function loadSelectedGoalIntoDraft(goalId) {
      const goal = getGoalById(goalId);
      if (!goal) return;
      state.admin.selectedGoalId = goalId;
      state.admin.draftGoal = clone(goal);
      renderAdminDraft();
    }

    function bindAdminAxisEvents() {
      document.getElementById('axis-editor').addEventListener('input', event => {
        const role = event.target.dataset.role;
        const index = Number(event.target.dataset.index);
        if (!role || Number.isNaN(index)) return;
        const axis = state.admin.draftGoal.axes[index];
        if (role === 'axis-name') axis.name = event.target.value;
        if (role === 'axis-min') axis.domainMin = Number(event.target.value);
        if (role === 'axis-max') axis.domainMax = Number(event.target.value);
        if (role === 'axis-resolution') axis.resolution = Number(event.target.value);
        renderAdminDraft();
      });

      document.getElementById('axis-editor').addEventListener('click', event => {
        if (event.target.dataset.role !== 'axis-remove') return;
        const index = Number(event.target.dataset.index);
        state.admin.draftGoal.axes.splice(index, 1);
        if (state.admin.draftGoal.axes.length === 0) {
          state.admin.draftGoal.axes.push({ name: '', domainMin: 0, domainMax: 100, resolution: 1 });
        }
        renderAdminDraft();
      });
    }

    async function saveAdminGoal() {
      state.admin.draftGoal.name = document.getElementById('admin-goal-name').value.trim();
      const response = await fetch('/api/admin/goals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.admin.draftGoal),
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.error || 'Goal 저장 중 오류가 발생했습니다.');
        return;
      }
      state.goals = data.goals;
      state.admin.selectedGoalId = data.savedGoal.id;
      state.admin.draftGoal = clone(data.savedGoal);
      state.user.selectedGoalId = data.savedGoal.id;
      refreshAll();
    }

    async function deleteSelectedGoal() {
      if (!state.admin.selectedGoalId) return;
      const response = await fetch('/api/admin/goals/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: state.admin.selectedGoalId }),
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.error || 'Goal 삭제 중 오류가 발생했습니다.');
        return;
      }
      state.goals = data.goals;
      state.admin.selectedGoalId = state.goals[0]?.id || null;
      state.admin.draftGoal = clone(state.goals[0] || { id: '', name: '', axes: [{ name: '', domainMin: 0, domainMax: 100, resolution: 1 }] });
      state.user.selectedGoalId = state.goals[0]?.id || null;
      refreshAll();
    }

    function parseDelimitedText(text, delimiter) {
      const lines = text.split(/\\r?\\n/).filter(Boolean);
      if (!lines.length) return { headers: [], rows: [] };
      const headers = lines[0].split(delimiter).map(item => item.trim());
      const rows = lines.slice(1).map(line => {
        const values = line.split(delimiter).map(item => item.trim());
        const row = {};
        headers.forEach((header, index) => {
          row[header] = values[index] ?? '';
        });
        return row;
      });
      return { headers, rows };
    }

    async function handleFileUpload(file) {
      const text = await file.text();
      const delimiter = file.name.endsWith('.tsv') ? '\\t' : ',';
      const parsed = parseDelimitedText(text, delimiter);
      state.user.fileName = file.name;
      state.user.headers = parsed.headers;
      state.user.rows = parsed.rows;
      const goal = getGoalById(state.user.selectedGoalId);
      state.user.axisMapping = {};
      goal.axes.forEach((axis, index) => {
        state.user.axisMapping[axis.name] = parsed.headers[index] || parsed.headers[0] || '';
      });
      state.user.primaryKey = goal.axes[0]?.name || '';
      document.getElementById('file-meta-box').textContent =
        `파일명: ${file.name}\\n행 수: ${parsed.rows.length}\\n컬럼: ${parsed.headers.join(', ')}`;
      document.getElementById('user-error-box').style.display = 'none';
      renderMappingControls();
    }

    async function runAnalysis() {
      const errorBox = document.getElementById('user-error-box');
      errorBox.style.display = 'none';
      const payload = {
        goalId: state.user.selectedGoalId,
        rows: state.user.rows,
        axisMapping: state.user.axisMapping,
        primaryKey: state.user.primaryKey,
      };
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        errorBox.textContent = data.error || '분석 요청 중 오류가 발생했습니다.';
        errorBox.style.display = 'block';
        return;
      }
      state.report.status = 'ready';
      state.report.result = data.result;
      state.report.meta = data.meta;
      state.report.summary = data.summary;
      state.report.confidenceReasons = data.confidenceReasons;
      renderReport();
      setView('report');
    }

    document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => setView(tab.dataset.view)));
    document.getElementById('admin-goal-select').addEventListener('change', event => loadSelectedGoalIntoDraft(event.target.value));
    document.getElementById('admin-new-goal').addEventListener('click', newDraftGoal);
    document.getElementById('admin-save-goal').addEventListener('click', saveAdminGoal);
    document.getElementById('admin-delete-goal').addEventListener('click', deleteSelectedGoal);
    document.getElementById('admin-load-template').addEventListener('click', () => {
      state.admin.draftGoal = clone(bootstrap.goals[0]);
      renderAdminDraft();
    });
    document.getElementById('axis-add-button').addEventListener('click', () => {
      state.admin.draftGoal.axes.push({ name: '', domainMin: 0, domainMax: 100, resolution: 1 });
      renderAdminDraft();
    });

    document.getElementById('user-goal-select').addEventListener('change', event => {
      state.user.selectedGoalId = event.target.value;
      state.user.axisMapping = {};
      state.user.primaryKey = '';
      renderUserGoalSummary();
      renderMappingControls();
    });
    document.getElementById('file-input').addEventListener('change', event => {
      const file = event.target.files?.[0];
      if (file) handleFileUpload(file);
    });
    document.getElementById('mapping-box').addEventListener('change', event => {
      const axisName = event.target.dataset.axisMap;
      if (!axisName) return;
      state.user.axisMapping[axisName] = event.target.value;
    });
    document.getElementById('primary-key-select').addEventListener('change', event => {
      state.user.primaryKey = event.target.value;
    });
    document.getElementById('run-analysis').addEventListener('click', runAnalysis);

    bindAdminAxisEvents();
    refreshAll();
  </script>
</body>
</html>
"""


PAGE_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Leesin Certification Dashboard</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --surface: #ffffff;
      --surface-soft: #f2f7f5;
      --ink: #18212b;
      --muted: #607080;
      --line: #d8e1e8;
      --teal: #117865;
      --gold: #c48a20;
      --red: #b84c3d;
      --green: #2f8a4f;
      --shadow: 0 12px 30px rgba(24, 33, 43, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Segoe UI", "Pretendard", sans-serif;
      background: linear-gradient(180deg, #f6f8fb 0%, #eef4f6 100%);
    }
    .page {
      width: min(1180px, calc(100% - 28px));
      margin: 18px auto 42px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 18px;
    }
    .brand h1 {
      margin: 0;
      font-size: 26px;
      letter-spacing: 0;
    }
    .brand p {
      margin: 4px 0 0;
      color: var(--muted);
      line-height: 1.5;
    }
    .icon-button {
      width: 44px;
      height: 44px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
      cursor: pointer;
      box-shadow: var(--shadow);
      font-size: 20px;
    }
    .action-panel, .section, dialog {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .action-panel {
      padding: 16px;
      display: grid;
      grid-template-columns: 1fr 1fr auto;
      gap: 14px;
      align-items: end;
    }
    label {
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
    }
    select, input, button {
      width: 100%;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      outline: none;
    }
    select:focus, input:focus {
      border-color: var(--teal);
      box-shadow: 0 0 0 3px rgba(17, 120, 101, 0.12);
    }
    button {
      cursor: pointer;
      font-weight: 800;
    }
    .primary {
      border: 0;
      color: #fff;
      background: linear-gradient(135deg, var(--teal), #1b9a80);
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.58;
      box-shadow: none;
    }
    .secondary {
      color: var(--teal);
      background: var(--surface-soft);
    }
    .danger {
      color: var(--red);
      background: #fff4f1;
    }
    .mapper {
      display: none;
      margin-top: 14px;
      padding: 16px;
    }
    .mapper-note {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .mapper-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    .primary-key-block {
      margin-top: 12px;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px dashed var(--line);
      background: #fbfdff;
    }
    .primary-key-group {
      display: grid;
      gap: 8px;
    }
    .radio-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0;
      text-transform: none;
    }
    .radio-pill input {
      width: auto;
      min-height: auto;
      margin: 0;
      padding: 0;
      accent-color: var(--teal);
    }
    .file-meta, .notice {
      color: var(--muted);
      line-height: 1.55;
      font-size: 14px;
    }
    .mapper-head {
      display: grid;
      grid-template-columns: 1.1fr 1.4fr 116px;
      gap: 10px;
      padding: 0 2px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .mapping-row {
      display: grid;
      grid-template-columns: 1.1fr 1.4fr 116px;
      gap: 10px;
      align-items: center;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .mapping-axis {
      display: grid;
      gap: 4px;
    }
    .mapping-axis strong {
      font-size: 15px;
      line-height: 1.3;
    }
    .mapping-axis span {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .mapping-select {
      display: grid;
      gap: 6px;
    }
    .mapping-select small,
    .primary-choice small {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .primary-choice {
      display: grid;
      justify-items: center;
      gap: 6px;
      text-transform: none;
      color: var(--ink);
    }
    .primary-choice input {
      width: auto;
      min-height: auto;
      margin: 0;
      padding: 0;
      accent-color: var(--teal);
    }
    .report {
      margin-top: 18px;
      display: grid;
      gap: 14px;
    }
    @keyframes issueCertificate {
      from { opacity: 0; transform: translateY(16px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .report.ready {
      animation: issueCertificate 420ms ease-out both;
    }
    .section {
      padding: 18px;
      margin-bottom: 14px;
    }
    .report > .section {
      margin-bottom: 0;
    }
    .section h2, .section h3 {
      margin: 0 0 12px;
      letter-spacing: 0;
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
      margin-bottom: 12px;
    }
    .section-head p {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
    }
    .score-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.8fr);
      gap: 12px;
      align-items: stretch;
    }
    .score-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      background: #fff;
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .score-card small {
      color: var(--muted);
      font-weight: 800;
      text-transform: uppercase;
    }
    .score-card strong {
      display: block;
      margin: 0;
      font-size: 36px;
      line-height: 1;
      letter-spacing: 0;
    }
    .score-card.placeholder strong {
      color: #b9c4cd;
    }
    .score-card.hero {
      border: 0;
      background: linear-gradient(135deg, #117865 0%, #1c9277 100%);
      color: #fff;
      padding: 24px;
    }
    .score-card.hero small,
    .score-card.hero .score-copy {
      color: rgba(255, 255, 255, 0.82);
    }
    .score-card.hero strong {
      display: flex;
      align-items: flex-end;
      gap: 10px;
      font-size: 68px;
    }
    .score-unit {
      font-size: 28px;
      line-height: 1.2;
      opacity: 0.9;
    }
    .score-status {
      width: fit-content;
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0;
    }
    .score-status.high {
      background: #def7ec;
      color: #145d3b;
    }
    .score-status.medium {
      background: #fff3dd;
      color: #8b6200;
    }
    .score-status.low {
      background: #ffe3dc;
      color: #9d3b2c;
    }
    .peer-card {
      gap: 14px;
    }
    .meta-emphasis {
      font-size: 34px;
      font-weight: 800;
      line-height: 1;
      color: var(--ink);
    }
    .meta-list {
      display: grid;
      gap: 10px;
    }
    .meta-item {
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }
    .meta-item strong {
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
    }
    .score-copy, .section-copy, .metric-caption {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }
    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .insight-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }
    .insight-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }
    .insight-card h4 {
      margin: 0 0 12px;
      letter-spacing: 0;
    }
    .bars, .summary-list, .reason-list, .sample-list, .coverage-grid, .equity-grid {
      display: grid;
      gap: 10px;
    }
    .pill, .reason, .sample-item {
      padding: 11px 12px;
      border-radius: 8px;
      background: var(--surface-soft);
      color: var(--ink);
      line-height: 1.45;
    }
    .reason.down {
      background: #fff3ef;
      color: var(--red);
    }
    .axis-diagnostic {
      display: grid;
      gap: 10px;
      padding: 14px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fff;
    }
    .axis-diagnostic-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }
    .axis-diagnostic h4 {
      margin: 0;
      font-size: 17px;
    }
    .axis-readout {
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .direction-chip {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 56px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
    }
    .direction-chip.high {
      background: #fff3dd;
      color: #8b6200;
    }
    .direction-chip.low {
      background: #e8f1ff;
      color: #2450a6;
    }
    .direction-chip.similar {
      background: #edf5f2;
      color: var(--teal);
    }
    .axis-note {
      padding: 10px 12px;
      border-radius: 8px;
      background: var(--surface-soft);
      line-height: 1.55;
    }
    .bar-row {
      display: grid;
      gap: 6px;
    }
    .bar-meta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 14px;
    }
    .bar-track {
      height: 12px;
      border-radius: 999px;
      overflow: hidden;
      background: #e8eef2;
    }
    .bar-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--teal), var(--gold));
    }
    .metric-summary {
      display: grid;
      gap: 8px;
      padding: 14px;
      border-radius: 8px;
      border: 1px dashed var(--line);
      background: #fbfdff;
    }
    .metric-summary.wide {
      grid-column: 1 / -1;
    }
    .metric-score {
      font-size: 30px;
      font-weight: 800;
      line-height: 1;
      color: var(--teal);
    }
    .metric-guide {
      padding: 10px 12px;
      border-radius: 8px;
      background: #fff3ef;
      color: var(--red);
      font-size: 13px;
      line-height: 1.55;
    }
    .metric-guide.good {
      background: #eef8f1;
      color: var(--green);
    }
    .sample-mini {
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fff;
      line-height: 1.5;
    }
    .sample-mini strong {
      display: block;
      margin-bottom: 4px;
    }
    .placeholder-card {
      padding: 14px;
      border-radius: 8px;
      border: 1px dashed var(--line);
      background: linear-gradient(180deg, #fbfdff 0%, #f4f7fa 100%);
    }
    .placeholder-stack {
      display: grid;
      gap: 10px;
    }
    .placeholder-line {
      height: 11px;
      border-radius: 999px;
      background: linear-gradient(90deg, #e3eaf0 0%, #f7fafc 45%, #e3eaf0 100%);
    }
    .placeholder-line.short { width: 42%; }
    .placeholder-line.mid { width: 68%; }
    .empty-chart {
      min-height: 150px;
      padding: 12px;
      border-radius: 8px;
      border: 1px dashed var(--line);
      background: linear-gradient(180deg, #fbfdff 0%, #f4f7fa 100%);
      display: grid;
      align-content: center;
      gap: 8px;
    }
    .empty-coverage {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .empty-column {
      display: grid;
      justify-items: center;
      gap: 8px;
    }
    .empty-rail {
      width: 24px;
      height: 110px;
      border-radius: 999px;
      background: #e7edf2;
      position: relative;
      overflow: hidden;
    }
    .empty-rail::after {
      content: "";
      position: absolute;
      inset: auto 0 20% 0;
      height: 30%;
      background: rgba(17, 120, 101, 0.32);
    }
    .empty-svg {
      width: 100%;
      height: 150px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,251,253,0.95)),
        repeating-linear-gradient(90deg, #eef3f6 0, #eef3f6 1px, transparent 1px, transparent 34px),
        repeating-linear-gradient(180deg, #eef3f6 0, #eef3f6 1px, transparent 1px, transparent 28px);
    }
    .coverage-grid {
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    }
    .coverage-axis, .equity-axis {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }
    .coverage-axis h4, .equity-axis h4 {
      margin: 0 0 10px;
    }
    .coverage-meta {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
    }
    .coverage-line {
      position: relative;
      width: 28px;
      height: 190px;
      margin: 10px auto;
      border-radius: 999px;
      background: #e7edf2;
      overflow: hidden;
    }
    .coverage-segment {
      position: absolute;
      left: 0;
      width: 100%;
      min-height: 2px;
      background: var(--teal);
    }
    .target-marker {
      position: absolute;
      left: -7px;
      width: 42px;
      height: 3px;
      background: var(--red);
    }
    .axis-range {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
    }
    .equity-grid {
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }
    .equity-svg {
      width: 100%;
      height: 180px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdfe;
    }
    .status-balanced { color: var(--green); font-weight: 800; }
    .status-imbalanced { color: var(--red); font-weight: 800; }
    dialog {
      width: min(880px, calc(100% - 28px));
      padding: 0;
    }
    dialog::backdrop { background: rgba(24, 33, 43, 0.38); }
    .modal-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }
    .modal-body {
      padding: 16px;
      display: grid;
      gap: 14px;
    }
    .modal-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
    }
    .toolbar-actions {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .text-toggle {
      width: auto;
      min-height: auto;
      padding: 0;
      border: 0;
      background: transparent;
      color: var(--teal);
      font-weight: 800;
      text-decoration: underline;
      text-underline-offset: 3px;
      box-shadow: none;
    }
    .advanced-panel {
      display: none;
      padding: 14px;
      border-radius: 8px;
      border: 1px dashed var(--line);
      background: #fbfdff;
      gap: 12px;
    }
    .advanced-panel.open {
      display: grid;
    }
    .axis-row {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr 1fr 1fr 1fr auto;
      gap: 8px;
      align-items: end;
      margin-bottom: 8px;
    }
    .help-chip {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      margin-left: 6px;
      border-radius: 999px;
      background: var(--surface-soft);
      color: var(--teal);
      font-size: 12px;
      font-weight: 800;
      cursor: help;
    }
    .field-help {
      display: block;
      margin-top: -2px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
      font-weight: 600;
      text-transform: none;
      letter-spacing: 0;
    }
    .error {
      display: none;
      margin-top: 12px;
      padding: 12px;
      border-radius: 8px;
      color: var(--red);
      background: #fff3ef;
      border: 1px solid #ffd3c7;
    }
    @media (max-width: 900px) {
      .action-panel, .mapper-grid, .score-grid, .two-col, .insight-grid, .axis-row, .mapper-head, .mapping-row { grid-template-columns: 1fr; }
      .modal-toolbar, .toolbar-actions { flex-direction: column; align-items: stretch; }
      .axis-diagnostic-head { flex-direction: column; align-items: flex-start; }
      .primary-choice { justify-items: start; }
      .score-card.hero strong { font-size: 52px; }
      .score-card strong { font-size: 34px; }
    }
  </style>
</head>
<body>
  <main class="page">
    <header class="topbar">
      <div class="brand">
        <h1>Leesin</h1>
        <p>데이터를 넣으면 아래에 인증 리포트가 발급됩니다.</p>
      </div>
      <button class="icon-button" id="settings-button" type="button" title="Admin Setup">⚙</button>
    </header>

    <section class="action-panel">
      <label>실험 목적 선택
        <select id="goal-select"></select>
      </label>
      <label>파일 업로드
        <input id="file-input" type="file" accept=".csv,.tsv,.txt">
      </label>
      <button class="primary" id="run-analysis" type="button">분석</button>
    </section>

    <section class="section mapper" id="mapper-section">
      <h3>데이터 컬럼 매칭 및 Primary Key 설정</h3>
      <div class="file-meta" id="file-meta-box">파일을 업로드하면 아래에서 Axis와 업로드 컬럼을 1:1로 연결하고 Primary Key를 지정할 수 있습니다.</div>
      <div class="mapper-head"><span>Axis</span><span>업로드 컬럼</span><span>Primary Key</span></div>
      <div class="mapper-grid" id="mapping-box"></div>
      <div class="primary-key-block">
        <div class="primary-key-group" id="primary-key-box">각 Axis 행 오른쪽에서 Primary Key를 하나만 선택하세요.</div>
      </div>
      <div class="mapper-note" id="mapper-hint">모든 필수 Axis를 매칭하고 Primary Key를 선택해야 분석 버튼이 활성화됩니다.</div>
      <div class="error" id="user-error-box"></div>
    </section>

    <section class="report" id="report-section">
      <div class="section">
        <h2>인증 리포트</h2>
        <div class="section-head">
          <div>
            <h2>인증 리포트</h2>
            <p>신뢰도가 가장 먼저 크게 표시되고, 아래에서 Axis별 이탈 진단과 원인, 개선 가이드를 순서대로 읽을 수 있습니다.</p>
          </div>
        </div>
        <div class="score-grid" id="score-grid"></div>
      </div>

      <div class="two-col">
        <section class="section">
          <h3>축별 이질성 기여도</h3>
          <div class="bars" id="contribution-bars"></div>
        </section>
        <section class="section">
          <h3>판정 요약</h3>
          <div class="summary-list" id="summary-list"></div>
        </section>
      </div>

      <section class="section">
        <h3>Sample Size</h3>
        <div class="sample-list" id="sample-list"></div>
      </section>

      <section class="section">
        <h3>Coverage</h3>
        <div class="coverage-grid" id="coverage-grid"></div>
      </section>

      <section class="section">
        <h3>Equitability</h3>
        <div class="equity-grid" id="equity-grid"></div>
      </section>

      <section class="section">
        <h3>신뢰도 감점 원인</h3>
        <div class="reason-list" id="confidence-reasons"></div>
      </section>
    </section>
  </main>

  <dialog id="settings-modal">
    <div class="modal-head">
      <div>
        <strong>Admin Setup</strong>
        <div class="notice">Experiment Goal, Axis, Unit, Domain Range, Resolution, K_m</div>
      </div>
      <button class="icon-button" id="settings-close" type="button" title="Close">×</button>
    </div>
    <div class="modal-body">
      <div class="modal-toolbar">
        <label>Existing Goal
          <select id="admin-goal-select"></select>
        </label>
        <div class="toolbar-actions">
          <button class="secondary" id="admin-new-goal" type="button">+ 새 Goal 만들기</button>
        </div>
      </div>
      <label>Experiment Goal
        <input id="admin-goal-name" placeholder="예: 고온 유량 품질 인증">
      </label>
      <div class="notice">Goal은 축의 템플릿입니다. 실제 분석에 포함할 축은 메인 화면에서 체크해서 선택합니다.</div>
      <div class="notice">Domain Range는 실험 목적상 유효하다고 보는 전체 물리적 값의 범위이고, Resolution은 데이터를 구분하는 최소 분석 간격입니다.</div>
      <div id="axis-editor"></div>
      <div class="two-col">
        <button class="secondary" id="axis-add-button" type="button">Axis 추가</button>
        <button class="primary" id="admin-save-goal" type="button">저장</button>
      </div>
      <button class="text-toggle" id="advanced-toggle" type="button">고급 설정 펼치기</button>
      <div class="advanced-panel" id="advanced-panel">
        <label>K_m
          <input id="admin-km-input" type="number" min="0.0001" step="any" value="10">
        </label>
        <div class="notice">K_m는 Sample Size 신뢰도 가중치 계산에 쓰이는 백엔드 상수입니다. 관리자만 조정하세요.</div>
      </div>
      <button class="danger" id="admin-delete-goal" type="button">선택 Goal 삭제</button>
    </div>
  </dialog>

  <script>
    const bootstrap = __BOOTSTRAP__;
    const state = {
      goals: bootstrap.goals,
      adminAllowed: Boolean(bootstrap.adminAllowed),
      selectedGoalId: bootstrap.goals[0]?.id || null,
      draftGoal: JSON.parse(JSON.stringify(bootstrap.goals[0] || { id: '', name: '', K_m: 10, axes: [{ name: '', unit: '', domainMin: 0, domainMax: 100, resolution: 1 }] })),
      adminAdvancedOpen: false,
      fileName: null,
      headers: [],
      rows: [],
      selectedAxes: Object.fromEntries((bootstrap.goals[0]?.axes || []).map(axis => [axis.name, true])),
      axisMapping: {},
      primaryKey: '',
      report: null,
    };

    function goalById(id) {
      return state.goals.find(goal => goal.id === id);
    }

    function clone(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function renderGoalSelects() {
      const options = state.goals.map(goal => `<option value="${escapeHtml(goal.id)}">${escapeHtml(goal.name)}</option>`).join('');
      document.getElementById('goal-select').innerHTML = options || '<option value="">등록된 Goal 없음</option>';
      document.getElementById('admin-goal-select').innerHTML = `<option value="">기존 Goal 선택</option>${options}`;
      document.getElementById('goal-select').value = state.selectedGoalId || '';
      document.getElementById('admin-goal-select').value = state.goals.some(goal => goal.id === state.draftGoal?.id) ? state.draftGoal.id : '';
    }

    function renderAdmin() {
      document.getElementById('admin-goal-name').value = state.draftGoal.name || '';
      document.getElementById('axis-editor').innerHTML = state.draftGoal.axes.map((axis, index) => `
        <div class="axis-row">
          <label>Axis <input data-axis-field="name" data-index="${index}" value="${axis.name}"></label>
          <label>Domain Min <input data-axis-field="domainMin" data-index="${index}" type="number" value="${axis.domainMin}"></label>
          <label>Domain Max <input data-axis-field="domainMax" data-index="${index}" type="number" value="${axis.domainMax}"></label>
          <label>Resolution <input data-axis-field="resolution" data-index="${index}" type="number" step="any" value="${axis.resolution}"></label>
          <button class="danger" data-axis-remove="${index}" type="button">삭제</button>
        </div>
      `).join('');
    }

    function renderMapping() {
      const goal = goalById(state.selectedGoalId);
      const mapper = document.getElementById('mapper-section');
      if (!goal || state.headers.length === 0) {
        mapper.style.display = 'none';
        return;
      }
      mapper.style.display = 'block';
      document.getElementById('mapping-box').innerHTML = goal.axes.map(axis => `
        <label>${axis.name}
          <select data-axis-map="${axis.name}">
            ${state.headers.map(header => `<option value="${header}" ${state.axisMapping[axis.name] === header ? 'selected' : ''}>${header}</option>`).join('')}
          </select>
        </label>
      `).join('');
      document.getElementById('primary-key-select').innerHTML = goal.axes
        .map(axis => `<option value="${axis.name}" ${state.primaryKey === axis.name ? 'selected' : ''}>${axis.name}</option>`)
        .join('');
    }

    function parseDelimitedText(text, delimiter) {
      const lines = text.split(/\\r?\\n/).filter(Boolean);
      if (!lines.length) return { headers: [], rows: [] };
      const headers = lines[0].split(delimiter).map(item => item.trim());
      const rows = lines.slice(1).map(line => {
        const values = line.split(delimiter).map(item => item.trim());
        const row = {};
        headers.forEach((header, index) => row[header] = values[index] ?? '');
        return row;
      });
      return { headers, rows };
    }

    async function handleFileUpload(file) {
      const text = await file.text();
      const delimiter = file.name.endsWith('.tsv') ? '\\t' : ',';
      const parsed = parseDelimitedText(text, delimiter);
      state.fileName = file.name;
      state.headers = parsed.headers;
      state.rows = parsed.rows;
      const goal = goalById(state.selectedGoalId);
      state.axisMapping = {};
      goal.axes.forEach((axis, index) => state.axisMapping[axis.name] = parsed.headers[index] || parsed.headers[0] || '');
      state.primaryKey = goal.axes[0]?.name || '';
      document.getElementById('file-meta-box').textContent = `${file.name} · ${parsed.rows.length} rows · ${parsed.headers.join(', ')}`;
      renderMapping();
    }

    function animateNumber(element, endValue, decimals = 0) {
      const duration = 720;
      const startTime = performance.now();
      function frame(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const value = endValue * (1 - Math.pow(1 - progress, 3));
        element.textContent = value.toFixed(decimals);
        if (progress < 1) requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    }

    function renderScores(report) {
      const result = report.result;
      document.getElementById('score-grid').innerHTML = `
        <div class="score-card heterogeneity"><small>이질성</small><strong data-count="${result.heterogeneity}" data-decimals="4">0</strong></div>
        <div class="score-card confidence"><small>신뢰도</small><strong data-count="${result.confidence}" data-decimals="4">0</strong></div>
        <div class="score-card"><small>Peer Group</small><strong data-count="${report.meta.peer_group_size}" data-decimals="0">0</strong></div>
      `;
      document.querySelectorAll('[data-count]').forEach(node => animateNumber(node, Number(node.dataset.count), Number(node.dataset.decimals)));
    }

    function renderContribution(report) {
      const diagnostics = axisDiagnostics(report);
      document.getElementById('contribution-bars').innerHTML = diagnostics.map(item => `
        <div class="axis-diagnostic">
          <div class="axis-diagnostic-head">
            <div>
              <h4>${escapeHtml(item.label)}</h4>
              <div class="axis-readout">현재 ${escapeHtml(item.currentText)} · 비교 중심 ${escapeHtml(item.centerText)}</div>
            </div>
            <span class="direction-chip ${item.directionClass}">${escapeHtml(item.directionText)}</span>
          </div>
          <div class="bar-meta">
            <span>${escapeHtml(item.deltaText)}</span>
            <strong>${item.contribution.toFixed(1)}%</strong>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, item.contribution)}%"></div></div>
          <div class="axis-note">${escapeHtml(item.interpretation)}</div>
        </div>
      `).join('');
    }

    function renderCoverage(report) {
      document.getElementById('coverage-grid').innerHTML = report.visualizations.coverage.axes.map(axis => {
        const span = axis.domainMax - axis.domainMin;
        const segments = axis.occupiedBinsDetail.map(bin => {
          const top = 100 - ((bin.end - axis.domainMin) / span * 100);
          const height = Math.max(((bin.end - bin.start) / span * 100), 1.5);
          return `<span class="coverage-segment" style="top:${top}%;height:${height}%"></span>`;
        }).join('');
        const marker = 100 - ((axis.targetValue - axis.domainMin) / span * 100);
        return `
          <div class="coverage-axis">
            <h4>${axis.axis}</h4>
            <div class="axis-range"><span>${axis.domainMax}</span><span>${axis.domainMin}</span></div>
            <div class="coverage-line">${segments}<span class="target-marker" style="top:${Math.max(0, Math.min(100, marker))}%"></span></div>
            <div class="axis-range"><span>${axis.occupiedBins}/${axis.totalBins} bins</span><span>target ${axis.targetValue}</span></div>
          </div>
        `;
      }).join('');
    }

    function renderEquitability(report) {
      document.getElementById('equity-grid').innerHTML = report.visualizations.equitability.axes.map(axis => {
        const bins = axis.bins;
        const maxCount = Math.max(1, ...bins.map(bin => bin.count));
        const width = 320;
        const height = 150;
        const gap = 2;
        const barWidth = Math.max(2, (width - gap * (bins.length - 1)) / bins.length);
        const bars = bins.map((bin, index) => {
          const barHeight = bin.count / maxCount * (height - 18);
          const x = index * (barWidth + gap);
          const y = height - barHeight;
          return `<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="2" fill="${bin.count ? '#117865' : '#dbe5eb'}"></rect>`;
        }).join('');
        const statusClass = axis.status === 'balanced' ? 'status-balanced' : 'status-imbalanced';
        const statusText = axis.status === 'balanced' ? '균형' : '불균형';
        return `
          <div class="equity-axis">
            <h4>${axis.axis} <span class="${statusClass}">${statusText}</span></h4>
            <svg class="equity-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${bars}</svg>
          </div>
        `;
      }).join('');
    }

    function renderReport(report) {
      state.report = report;
      document.getElementById('report-section').style.display = 'block';
      renderScores(report);
      renderContribution(report);
      document.getElementById('summary-list').innerHTML = report.summary.map(item => `<div class="pill">${item}</div>`).join('');
      document.getElementById('confidence-reasons').innerHTML = report.confidenceReasons.map(item => `
        <div class="reason ${item.impact === 'down' ? 'down' : ''}"><strong>${item.label}: ${item.score.toFixed(4)}</strong><br>${item.message}</div>
      `).join('');
      document.getElementById('sample-list').innerHTML = report.visualizations.sampleSize.items.map(item => `
        <div class="sample-item"><strong>${item.axis}</strong> · Peer samples <span data-count="${item.count}" data-decimals="0">0</span> · Z ${item.z.toFixed(4)}</div>
      `).join('');
      document.querySelectorAll('#sample-list [data-count]').forEach(node => animateNumber(node, Number(node.dataset.count), 0));
      renderCoverage(report);
      renderEquitability(report);
      document.getElementById('report-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    async function runAnalysis() {
      const errorBox = document.getElementById('user-error-box');
      errorBox.style.display = 'none';
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          goalId: state.selectedGoalId,
          rows: state.rows,
          axisMapping: state.axisMapping,
          primaryKey: state.primaryKey,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        errorBox.textContent = data.error || '분석 요청 중 오류가 발생했습니다.';
        errorBox.style.display = 'block';
        return;
      }
      renderReport(data);
    }

    async function saveGoal() {
      state.draftGoal.name = document.getElementById('admin-goal-name').value.trim();
      state.draftGoal.K_m = Number(document.getElementById('admin-km-input').value || state.draftGoal.K_m || 10);
      const response = await fetch('/api/admin/goals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.draftGoal),
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.error || 'Goal 저장 중 오류가 발생했습니다.');
        return;
      }
      state.goals = data.goals;
      state.selectedGoalId = data.savedGoal.id;
      state.draftGoal = clone(data.savedGoal);
      renderAll();
    }

    async function deleteGoal() {
      const response = await fetch('/api/admin/goals/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: state.draftGoal.id }),
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.error || 'Goal 삭제 중 오류가 발생했습니다.');
        return;
      }
      state.goals = data.goals;
      state.selectedGoalId = state.goals[0]?.id || null;
      state.draftGoal = clone(state.goals[0]);
      renderAll();
    }

    function renderAll() {
      renderGoalSelects();
      renderAdmin();
      renderMapping();
      document.getElementById('settings-button').style.display = state.adminAllowed ? '' : 'none';
    }

    function escapeHtml(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function normalizeKey(value) {
      return String(value || '')
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9가-힣]+/g, '');
    }

    function buildSuggestedMapping(goal, headers) {
      if (!goal || !headers.length) return {};
      const headerByKey = new Map(headers.map(header => [normalizeKey(header), header]));
      const mapping = {};
      const used = new Set();
      goal.axes.forEach(axis => {
        const exact = headerByKey.get(normalizeKey(axis.name));
        if (exact && !used.has(exact)) {
          mapping[axis.name] = exact;
          used.add(exact);
        } else {
          mapping[axis.name] = '';
        }
      });
      return mapping;
    }

    function selectedAxesForGoal(goal) {
      if (!goal) return [];
      return goal.axes.filter(axis => Boolean(state.selectedAxes[axis.name]));
    }

    function selectedAxisNames(goal) {
      return selectedAxesForGoal(goal).map(axis => axis.name);
    }

    function mappedHeadersForGoal(goal) {
      if (!goal) return [];
      return selectedAxesForGoal(goal).map(axis => state.axisMapping[axis.name]).filter(Boolean);
    }

    function hasDuplicateMappings(goal) {
      const mapped = mappedHeadersForGoal(goal);
      return mapped.length !== new Set(mapped).size;
    }

    function peerGroupPreviewCount(goal) {
      const selected = selectedAxisNames(goal);
      if (!goal || selected.length === 0) {
        return 0;
      }
      const subsetKey = selected.map(name => String(name || '').trim().toLowerCase()).join('|');
      const subsetCount = Number(bootstrap.peerSubsetCounts?.[goal.id]?.[subsetKey]);
      if (Number.isFinite(subsetCount)) {
        return subsetCount;
      }
      return Number(bootstrap.peerCounts?.[goal.id] || 0);
    }

    function rowCountWarning(goal) {
      const selectedCount = selectedAxisNames(goal).length;
      if (!goal || selectedCount === 0 || state.rows.length === 0) {
        return '';
      }
      if (state.rows.length < selectedCount + 1) {
        return '데이터가 너무 적어 공분산 연산이 불가능합니다. 데이터를 추가하거나 축 선택을 줄이세요.';
      }
      return '';
    }

    function isMappingComplete() {
      const goal = goalById(state.selectedGoalId);
      const selected = selectedAxesForGoal(goal);
      if (!goal || selected.length === 0 || state.headers.length === 0 || state.rows.length === 0 || !state.primaryKey) {
        return false;
      }
      if (!selected.some(axis => axis.name === state.primaryKey)) {
        return false;
      }
      if (hasDuplicateMappings(goal) || rowCountWarning(goal)) {
        return false;
      }
      return selected.every(axis => {
        const mappedHeader = state.axisMapping[axis.name];
        return Boolean(mappedHeader) && state.headers.includes(mappedHeader);
      });
    }

    function updateAnalyzeButton() {
      const goal = goalById(state.selectedGoalId);
      const button = document.getElementById('run-analysis');
      const errorBox = document.getElementById('user-error-box');
      const warning = rowCountWarning(goal);
      const ready = isMappingComplete();
      button.disabled = !ready;
      if (warning) {
        errorBox.textContent = warning;
        errorBox.style.display = 'block';
      } else {
        errorBox.style.display = 'none';
      }
      button.title = ready
        ? '선택한 축과 매핑으로 분석을 시작합니다.'
        : '축 선택, 컬럼 매핑, Primary Key, 데이터 개수 조건을 모두 충족해야 분석할 수 있습니다.';
    }

    function resetUserSelection(goal) {
      state.selectedAxes = goal
        ? Object.fromEntries(goal.axes.map(axis => [axis.name, true]))
        : {};
      state.axisMapping = buildSuggestedMapping(goal, state.headers);
      state.primaryKey = '';
      state.report = null;
    }

    function blankAxis() {
      return { name: '', unit: '', domainMin: 0, domainMax: 100, resolution: 1 };
    }

    function blankGoal() {
      return { id: `goal_${Date.now()}`, name: '', K_m: 10, axes: [blankAxis()] };
    }

    function axisDisplayLabel(axis) {
      if (!axis) return '';
      const unit = String(axis.unit || '').trim();
      return unit ? `${axis.name} (${unit})` : axis.name;
    }

    function renderAdmin() {
      const draft = state.draftGoal || blankGoal();
      if (!Array.isArray(draft.axes) || draft.axes.length === 0) {
        draft.axes = [blankAxis()];
      }
      if (!(Number(draft.K_m) > 0)) {
        draft.K_m = 10;
      }
      document.getElementById('admin-goal-name').value = draft.name || '';
      document.getElementById('admin-km-input').value = Number(draft.K_m || 10);
      document.getElementById('advanced-panel').classList.toggle('open', Boolean(state.adminAdvancedOpen));
      document.getElementById('advanced-toggle').textContent = state.adminAdvancedOpen ? '고급 설정 접기' : '고급 설정 펼치기';
      document.getElementById('axis-editor').innerHTML = draft.axes.map((axis, index) => `
        <div class="axis-row">
          <label>Axis <input data-axis-field="name" data-index="${index}" value="${escapeHtml(axis.name || '')}"></label>
          <label>Unit <input data-axis-field="unit" data-index="${index}" value="${escapeHtml(axis.unit || '')}" placeholder="℃ / bar / kg/h"></label>
          <label>Domain Min <span class="help-chip" title="물리적 유효 범위">?</span><input data-axis-field="domainMin" data-index="${index}" type="number" value="${Number(axis.domainMin ?? 0)}"><span class="field-help">Domain Range 시작값입니다.</span></label>
          <label>Domain Max <span class="help-chip" title="물리적 유효 범위">?</span><input data-axis-field="domainMax" data-index="${index}" type="number" value="${Number(axis.domainMax ?? 100)}"><span class="field-help">Domain Range 종료값입니다.</span></label>
          <label>Resolution <span class="help-chip" title="분석 격자 크기 (권장: 전체 범위의 5~10%)">?</span><input data-axis-field="resolution" data-index="${index}" type="number" step="any" value="${Number(axis.resolution ?? 1)}" placeholder="예: 1, 0.5, 0.1"><span class="field-help">데이터를 구분하는 최소 분석 간격입니다.</span></label>
          <button class="danger" data-axis-remove="${index}" type="button">삭제</button>
        </div>
      `).join('');
    }

    function buildReportScaffold() {
      const reportSection = document.getElementById('report-section');
      if (reportSection.dataset.scaffolded === 'true') {
        return;
      }

      const heroSection = Array.from(reportSection.children).find(node => node.classList && node.classList.contains('section'));
      if (!heroSection) {
        return;
      }
      const firstChild = heroSection ? heroSection.firstElementChild : null;
      if (firstChild && firstChild.tagName === 'H2') {
        firstChild.remove();
      }

      const summaryList = document.getElementById('summary-list');
      const contributionBars = document.getElementById('contribution-bars');
      const sampleList = document.getElementById('sample-list');
      const coverageGrid = document.getElementById('coverage-grid');
      const equityGrid = document.getElementById('equity-grid');
      const reasonsList = document.getElementById('confidence-reasons');

      const contributionSection = document.createElement('section');
      contributionSection.className = 'section';
      contributionSection.innerHTML = `
        <div class="section-head">
          <div>
            <h3>Axis별 이탈 진단</h3>
            <p>각 Axis가 비교 집단 중심보다 높거나 낮은 방향으로 얼마나 벗어났는지, 그리고 전체 이탈에 얼마나 기여했는지 함께 읽습니다.</p>
          </div>
        </div>
        <div data-report-slot="contribution"></div>
      `;
      contributionSection.querySelector('[data-report-slot="contribution"]').appendChild(contributionBars);

      const interpretationSection = document.createElement('section');
      interpretationSection.className = 'section';
      interpretationSection.innerHTML = `
        <div class="section-head">
          <div>
            <h3>리포트 해석</h3>
            <p>신뢰도와 Axis별 이탈 진단을 조합해 바로 판단할 수 있도록 설명형 문장으로 정리합니다.</p>
          </div>
        </div>
        <div data-report-slot="summary"></div>
      `;
      interpretationSection.querySelector('[data-report-slot="summary"]').appendChild(summaryList);

      const insightSection = document.createElement('section');
      insightSection.className = 'section';
      insightSection.innerHTML = `
        <div class="section-head">
          <div>
            <h3>신뢰도 3대 지표</h3>
            <p>Sample Size, Coverage, Equitability를 각각 설명과 개선 가이드와 함께 읽을 수 있게 구성했습니다.</p>
          </div>
        </div>
        <div class="insight-grid">
          <div class="insight-card">
            <h4>Sample Size</h4>
            <div data-report-slot="sample"></div>
          </div>
          <div class="insight-card">
            <h4>Coverage</h4>
            <div data-report-slot="coverage"></div>
          </div>
          <div class="insight-card">
            <h4>Equitability</h4>
            <div data-report-slot="equity"></div>
          </div>
        </div>
      `;
      insightSection.querySelector('[data-report-slot="sample"]').appendChild(sampleList);
      insightSection.querySelector('[data-report-slot="coverage"]').appendChild(coverageGrid);
      insightSection.querySelector('[data-report-slot="equity"]').appendChild(equityGrid);

      const reasonSection = document.createElement('section');
      reasonSection.className = 'section';
      reasonSection.innerHTML = `
        <div class="section-head">
          <div>
            <h3>원인 및 개선 가이드</h3>
            <p>어떤 요인이 신뢰도를 깎았는지와, 다음 실험에서 무엇을 보완해야 하는지 함께 안내합니다.</p>
          </div>
        </div>
        <div data-report-slot="reasons"></div>
      `;
      reasonSection.querySelector('[data-report-slot="reasons"]').appendChild(reasonsList);

      reportSection.innerHTML = '';
      reportSection.append(heroSection, contributionSection, interpretationSection, insightSection, reasonSection);
      reportSection.dataset.scaffolded = 'true';
    }

    function renderReportSkeleton() {
      const goal = goalById(state.selectedGoalId);
      const previewCount = peerGroupPreviewCount(goal);
      const reportSection = document.getElementById('report-section');
      reportSection.classList.remove('ready');
      document.getElementById('score-grid').innerHTML = `
        <div class="score-card placeholder">
          <small>신뢰도</small>
          <strong>--<span class="score-unit">%</span></strong>
          <div class="score-status low">판정 대기</div>
          <p class="score-copy">분석이 완료되면 신뢰도가 가장 먼저 크게 표시됩니다.</p>
        </div>
        <div class="score-card placeholder peer-card">
          <small>Peer Group</small>
          <div class="meta-emphasis">${previewCount || '--'}개</div>
          <p class="score-copy">현재 데이터와 같은 실험 목적을 가진 비교 기준 데이터 집단입니다.</p>
        </div>
      `;
      document.getElementById('summary-list').innerHTML = `
        <div class="placeholder-card placeholder-stack">
          <div class="placeholder-line mid"></div>
          <div class="placeholder-line"></div>
        </div>
        <div class="placeholder-card placeholder-stack">
          <div class="placeholder-line short"></div>
          <div class="placeholder-line"></div>
        </div>
      `;
      document.getElementById('confidence-reasons').innerHTML = `
        <div class="placeholder-card placeholder-stack">
          <div class="placeholder-line mid"></div>
          <div class="placeholder-line"></div>
          <div class="placeholder-line short"></div>
        </div>
      `;
      document.getElementById('contribution-bars').innerHTML = `
        <div class="placeholder-card placeholder-stack">
          <div class="placeholder-line short"></div>
          <div class="placeholder-line"></div>
          <div class="placeholder-line mid"></div>
        </div>
        <div class="placeholder-card placeholder-stack">
          <div class="placeholder-line short"></div>
          <div class="placeholder-line"></div>
          <div class="placeholder-line mid"></div>
        </div>
        <div class="placeholder-card placeholder-stack">
          <div class="placeholder-line short"></div>
          <div class="placeholder-line"></div>
          <div class="placeholder-line mid"></div>
        </div>
      `;
      document.getElementById('sample-list').innerHTML = `
        <div class="placeholder-card placeholder-stack">
          <div class="placeholder-line mid"></div>
          <div class="placeholder-line short"></div>
        </div>
        <div class="placeholder-card placeholder-stack">
          <div class="placeholder-line mid"></div>
          <div class="placeholder-line short"></div>
        </div>
      `;
      document.getElementById('coverage-grid').innerHTML = `
        <div class="empty-chart">
          <div class="placeholder-line mid"></div>
          <div class="empty-coverage">
            <div class="empty-column"><div class="empty-rail"></div><div class="placeholder-line short"></div></div>
            <div class="empty-column"><div class="empty-rail"></div><div class="placeholder-line short"></div></div>
            <div class="empty-column"><div class="empty-rail"></div><div class="placeholder-line short"></div></div>
          </div>
        </div>
      `;
      document.getElementById('equity-grid').innerHTML = `
        <div class="empty-chart">
          <div class="placeholder-line mid"></div>
          <div class="empty-svg"></div>
        </div>
      `;
    }

    function renderMapping() {
      const goal = goalById(state.selectedGoalId);
      const mapper = document.getElementById('mapper-section');
      const hint = document.getElementById('mapper-hint');
      if (!goal) {
        mapper.style.display = 'none';
        renderReportSkeleton();
        updateAnalyzeButton();
        return;
      }
      mapper.style.display = 'block';
      document.getElementById('mapping-box').innerHTML = goal.axes.map(axis => `
        <div class="mapping-row">
          <div class="mapping-axis">
            <label class="radio-pill">
              <input type="checkbox" data-axis-toggle="${escapeHtml(axis.name)}" ${state.selectedAxes[axis.name] ? 'checked' : ''}>
              <span>${escapeHtml(axisDisplayLabel(axis))}</span>
            </label>
            <span>${state.selectedAxes[axis.name] ? '이번 분석에 포함됩니다.' : '이번 분석에서 제외됩니다.'}</span>
          </div>
          <label class="mapping-select">
            <small>업로드 컬럼</small>
            <select data-axis-map="${escapeHtml(axis.name)}" ${!state.selectedAxes[axis.name] || state.headers.length === 0 ? 'disabled' : ''}>
              <option value="">CSV 헤더 선택</option>
              ${state.headers.map(header => `<option value="${escapeHtml(header)}" ${state.axisMapping[axis.name] === header ? 'selected' : ''}>${escapeHtml(header)}</option>`).join('')}
            </select>
          </label>
          <label class="primary-choice">
            <small>Primary</small>
            <input type="radio" name="primary-key" value="${escapeHtml(axis.name)}" ${state.primaryKey === axis.name ? 'checked' : ''} ${state.selectedAxes[axis.name] ? '' : 'disabled'}>
          </label>
        </div>
      `).join('');
      document.getElementById('primary-key-box').textContent = `선택된 축 ${selectedAxisNames(goal).length}개 · 매칭 가능한 Peer Group N ${peerGroupPreviewCount(goal)}개`;
      if (state.fileName) {
        document.getElementById('file-meta-box').textContent = `${state.fileName} · ${state.rows.length} rows · ${state.headers.join(', ')}`;
      } else {
        document.getElementById('file-meta-box').textContent = 'Goal의 축 템플릿에서 이번 분석에 사용할 축을 먼저 고르고, 파일 업로드 후 체크된 축만 컬럼 매핑하세요.';
      }
      if (selectedAxisNames(goal).length === 0) {
        hint.textContent = '분석에 포함할 축을 하나 이상 체크하세요.';
      } else if (state.rows.length === 0) {
        hint.textContent = '파일을 업로드하면 체크된 축에 대해서만 컬럼 매핑 드롭다운이 활성화됩니다.';
      } else if (rowCountWarning(goal)) {
        hint.textContent = rowCountWarning(goal);
      } else if (hasDuplicateMappings(goal)) {
        hint.textContent = '하나의 CSV 헤더를 여러 Axis에 중복 연결할 수 없습니다. 각 Axis를 서로 다른 컬럼에 매핑해주세요.';
      } else if (isMappingComplete()) {
        hint.textContent = '매핑이 완료되었습니다. 분석을 누르면 바로 아래 인증 리포트 카드에 결과가 채워집니다.';
      } else {
        hint.textContent = '체크된 축만 컬럼에 매핑하고, 그중 하나를 Primary Key로 선택해야 분석 버튼이 활성화됩니다.';
      }
      renderReportSkeleton();
      updateAnalyzeButton();
    }

    async function handleFileUpload(file) {
      const text = await file.text();
      const preferredDelimiter = file.name.endsWith('.tsv') ? '\\t' : ',';
      let parsed = parseDelimitedText(text, preferredDelimiter);
      if (parsed.headers.length <= 1 && text.includes('\\t')) {
        parsed = parseDelimitedText(text, '\\t');
      }
      const goal = goalById(state.selectedGoalId);
      const preservedSelection = goal
        ? Object.fromEntries(goal.axes.map(axis => [axis.name, state.selectedAxes[axis.name] !== false]))
        : {};
      const preservedPrimaryKey = preservedSelection[state.primaryKey] ? state.primaryKey : '';
      state.fileName = file.name;
      state.headers = parsed.headers;
      state.rows = parsed.rows;
      state.selectedAxes = preservedSelection;
      state.axisMapping = buildSuggestedMapping(goal, parsed.headers);
      state.primaryKey = preservedPrimaryKey;
      state.report = null;
      document.getElementById('file-meta-box').textContent = `${file.name} · ${parsed.rows.length} rows · ${parsed.headers.join(', ')}`;
      renderMapping();
      renderReportSkeleton();
    }

    function formatNumber(value, decimals = 2) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) {
        return '-';
      }
      return numeric.toFixed(decimals).replace(/\.0+$|(\.\d*[1-9])0+$/, '$1');
    }

    function formatPercent(value, decimals = 0) {
      return `${formatNumber(Number(value) * 100, decimals)}%`;
    }

    function formatWithUnit(value, unit = '', decimals = 2) {
      const formatted = formatNumber(value, decimals);
      return unit ? `${formatted} ${unit}` : formatted;
    }

    function confidenceDescriptor(score) {
      const numeric = Number(score);
      if (numeric >= 0.75) {
        return {
          label: '높음',
          tone: 'high',
          description: '비교 집단의 Sample Size, Coverage, Equitability가 전반적으로 양호해 결과 해석의 정당성이 높은 편입니다.',
        };
      }
      if (numeric >= 0.45) {
        return {
          label: '보통',
          tone: 'medium',
          description: '기본 해석은 가능하지만, 비교 집단의 폭이나 균형이 더 보강되면 판단의 정당성이 더 높아집니다.',
        };
      }
      return {
        label: '낮음',
        tone: 'low',
        description: '현재 비교 집단이 충분히 넓거나 고르지 않아, 결과 해석을 보수적으로 받아들이는 편이 안전합니다.',
      };
    }

    function metricGuide(label, score) {
      const numeric = Number(score);
      if (label === 'Sample Size') {
        return numeric < 0.6
          ? { good: false, text: '같은 Goal의 Peer Group을 더 확보해 Sample Size를 높이세요.' }
          : { good: true, text: '현재 Peer Group 수는 기본 해석에 활용할 수 있는 수준입니다.' };
      }
      if (label === 'Coverage') {
        return numeric < 0.3
          ? { good: false, text: 'Domain Range 전역을 더 넓게 대표하는 비교 데이터를 확보하세요.' }
          : { good: true, text: '비교 데이터가 Domain Range 안에서 비교적 넓은 구간을 채우고 있습니다.' };
      }
      if (label === 'Equitability') {
        return numeric < 0.5
          ? { good: false, text: '특정 bin에 몰린 데이터를 분산시켜 균형 있게 추가 수집하세요.' }
          : { good: true, text: '채워진 구간 안에서는 데이터가 비교적 고르게 분포합니다.' };
      }
      if (label === 'Engine Robustness') {
        return numeric < 0.7
          ? { good: false, text: '비정규 분포 영향이 큰 상태입니다. 비교 데이터 분포를 더 확보해 재확인하세요.' }
          : { good: true, text: '현재 분포에서는 엔진 효율이 크게 흔들리지 않습니다.' };
      }
      return { good: numeric >= 0.5, text: '현재 지표를 유지하면서 추가 비교 데이터를 확보해보세요.' };
    }

    function axisDiagnostics(report) {
      const axes = report.meta.axes || [];
      const coverageAxes = new Map((report.visualizations.coverage.axes || []).map(axis => [axis.axis, axis]));
      const contributionAxes = new Map((report.result.contributions || []).map(item => [item.axis, item]));
      return axes.map((axis, index) => {
        const coverage = coverageAxes.get(axis.name) || {};
        const contribution = Number(contributionAxes.get(axis.name)?.percent || 0);
        const center = Number(report.result.center?.[index] ?? 0);
        const current = Number(coverage.targetValue ?? 0);
        const resolution = Number(coverage.resolution ?? axis.resolution ?? 0);
        const tolerance = Math.max(Math.abs(resolution) * 0.25, 1e-9);
        let directionClass = 'similar';
        let directionText = '유사';
        if (current - center > tolerance) {
          directionClass = 'high';
          directionText = '높음';
        } else if (current - center < -tolerance) {
          directionClass = 'low';
          directionText = '낮음';
        }
        const difference = current - center;
        const label = axisDisplayLabel(axis);
        const interpretation = directionClass === 'similar'
          ? `${label}: 비교 집단 중심과 유사합니다. 전체 이탈 기여도 ${contribution.toFixed(1)}%.`
          : `${label}: 비교 집단 중심보다 ${directionText} 방향으로 이탈했습니다. 전체 이탈 기여도 ${contribution.toFixed(1)}%.`;
        return {
          label,
          contribution,
          directionClass,
          directionText,
          currentText: formatWithUnit(current, axis.unit),
          centerText: formatWithUnit(center, axis.unit),
          deltaText: Math.abs(difference) <= tolerance
            ? '중심과 거의 같습니다.'
            : `차이 ${difference > 0 ? '+' : '-'}${formatWithUnit(Math.abs(difference), axis.unit)}`,
          interpretation,
        };
      }).sort((left, right) => right.contribution - left.contribution);
    }

    function buildInterpretationLines(report) {
      const result = report.result;
      const diagnostics = axisDiagnostics(report);
      const topAxis = diagnostics[0];
      const confidence = confidenceDescriptor(result.confidence);
      const lines = [];
      if (topAxis) {
        if (topAxis.directionClass === 'similar') {
          lines.push(`${topAxis.label} 축은 비교 집단 중심과 크게 다르지 않으며, 현재 신뢰도는 ${confidence.label}입니다.`);
        } else {
          lines.push(`${topAxis.label} 축에서 비교 집단보다 ${topAxis.directionText} 방향의 이탈이 확인되며, 현재 신뢰도는 ${confidence.label}입니다.`);
        }
      }
      if (result.confidence >= 0.75 && topAxis && topAxis.directionClass !== 'similar') {
        lines.push('현재 조건에서는 의미 있는 차이일 가능성이 있습니다. 동일 Goal의 재현 실험으로 확인해보세요.');
      } else if (result.coverage_C < 0.3) {
        lines.push('이탈은 확인되지만 Coverage가 낮아, 추가 비교 데이터 확보 전까지 해석을 보류하는 것이 좋습니다.');
      } else if (result.equitability_E < 0.5) {
        lines.push('점유 구간 안의 분포가 고르지 않아, 특정 bin에 데이터가 몰리지 않았는지 먼저 점검하는 것이 좋습니다.');
      } else if (result.sample_size_Z < 0.6) {
        lines.push('Peer Group 수가 아직 충분하지 않아, 같은 Goal의 비교 기준 데이터를 더 확보하는 편이 좋습니다.');
      } else {
        lines.push('현재 결과는 비교 집단 안에서 의미를 검토할 수 있는 기본 조건을 갖추고 있습니다.');
      }
      lines.push(`Peer Group은 현재 데이터와 같은 실험 목적을 가진 비교 기준 데이터 집단이며, Primary Key는 '${report.meta.primary_key}'입니다.`);
      return lines;
    }

    function renderScores(report) {
      const result = report.result;
      const confidence = confidenceDescriptor(result.confidence);
      document.getElementById('score-grid').innerHTML = `
        <div class="score-card hero">
          <small>신뢰도</small>
          <strong><span data-score-count="${result.confidence * 100}" data-decimals="0">0</span><span class="score-unit">%</span></strong>
          <div class="score-status ${confidence.tone}">${confidence.label}</div>
          <p class="score-copy">${escapeHtml(confidence.description)}</p>
        </div>
        <div class="score-card peer-card">
          <small>Peer Group</small>
          <div class="meta-emphasis"><span data-score-count="${report.meta.peer_group_size}" data-decimals="0">0</span>개</div>
          <p class="score-copy">현재 데이터와 같은 실험 목적을 가진 비교 기준 데이터 집단입니다.</p>
          <div class="meta-list">
            <div class="meta-item">
              <strong>Primary Key</strong>
              <span>${escapeHtml(report.meta.primary_key)}</span>
            </div>
            <div class="meta-item">
              <strong>분석 엔진</strong>
              <span>${escapeHtml(String(result.engine || '').toUpperCase())}</span>
            </div>
          </div>
        </div>
      `;
      document.querySelectorAll('#score-grid [data-score-count]').forEach(node => animateNumber(node, Number(node.dataset.scoreCount), Number(node.dataset.decimals || 0)));
    }

    function renderSampleSize(report) {
      const sample = report.visualizations.sampleSize;
      const guide = metricGuide('Sample Size', sample.z);
      document.getElementById('sample-list').innerHTML = `
        <div class="metric-summary">
          <small>Sample Size</small>
          <div class="metric-score">${escapeHtml(formatPercent(sample.z, 0))}</div>
          <p class="metric-caption">현재 데이터와 같은 실험 목적을 가진 비교 기준 데이터 집단(Peer Group)의 충분성을 반영합니다.</p>
          <div class="metric-guide ${guide.good ? 'good' : ''}">${escapeHtml(guide.text)}</div>
        </div>
        <div class="sample-item">
          <strong>Peer Group <span data-sample-count="${sample.peerGroupCount}">0</span>개</strong><br>
          <span class="metric-caption">Peer Group은 현재 데이터와 같은 실험 목적을 가진 비교 기준 데이터 집단입니다.</span>
        </div>
        ${(sample.items || []).map(item => `
          <div class="sample-mini">
            <strong>${escapeHtml(item.label || item.axis)}</strong>
            <span>비교 샘플 <span data-sample-count="${item.count}">0</span>개 · Z ${escapeHtml(formatPercent(item.z, 0))}</span>
          </div>
        `).join('')}
      `;
      document.querySelectorAll('#sample-list [data-sample-count]').forEach(node => animateNumber(node, Number(node.dataset.sampleCount), 0));
    }

    function renderCoverage(report) {
      const coverage = report.visualizations.coverage;
      const guide = metricGuide('Coverage', coverage.score);
      document.getElementById('coverage-grid').innerHTML = `
        <div class="metric-summary wide">
          <small>Coverage</small>
          <div class="metric-score">${escapeHtml(formatPercent(coverage.score, 1))}</div>
          <p class="metric-caption">실험 목적상 가능한 전체 범위 중, 비교 데이터가 실제로 채운 범위의 비율입니다.</p>
          <p class="metric-caption">Domain Range는 실험 목적상 유효하다고 보는 전체 물리적 값의 범위이고, Resolution은 데이터를 구분하는 최소 분석 간격입니다. 예: 1℃, 0.1 bar</p>
          <div class="metric-guide ${guide.good ? 'good' : ''}">${escapeHtml(guide.text)}</div>
        </div>
        ${(coverage.axes || []).map(axis => {
          const span = Math.max(axis.domainMax - axis.domainMin, 1e-9);
          const segments = axis.occupiedBinsDetail.map(bin => {
            const top = 100 - ((bin.end - axis.domainMin) / span * 100);
            const height = Math.max(((bin.end - bin.start) / span * 100), 1.5);
            return `<span class="coverage-segment" style="top:${top}%;height:${height}%"></span>`;
          }).join('');
          const marker = 100 - ((axis.targetValue - axis.domainMin) / span * 100);
          return `
            <div class="coverage-axis">
              <h4>${escapeHtml(axis.label || axis.axis)}</h4>
              <div class="axis-range"><span>${escapeHtml(formatWithUnit(axis.domainMax, axis.unit))}</span><span>${escapeHtml(formatWithUnit(axis.domainMin, axis.unit))}</span></div>
              <div class="coverage-line">${segments}<span class="target-marker" style="top:${Math.max(0, Math.min(100, marker))}%"></span></div>
              <div class="axis-range"><span>${axis.occupiedBins}/${axis.totalBins} bins</span><span>현재 ${escapeHtml(formatWithUnit(axis.targetValue, axis.unit))}</span></div>
              <div class="coverage-meta">Domain Range ${escapeHtml(formatWithUnit(axis.domainMin, axis.unit))} ~ ${escapeHtml(formatWithUnit(axis.domainMax, axis.unit))} · Resolution ${escapeHtml(formatWithUnit(axis.resolution, axis.unit))}</div>
            </div>
          `;
        }).join('')}
      `;
    }

    function renderEquitability(report) {
      const equitability = report.visualizations.equitability;
      const guide = metricGuide('Equitability', equitability.score);
      document.getElementById('equity-grid').innerHTML = `
        <div class="metric-summary wide">
          <small>Equitability</small>
          <div class="metric-score">${escapeHtml(formatPercent(equitability.score, 1))}</div>
          <p class="metric-caption">채워진 구간 안에서 데이터가 얼마나 균등하게 분포하는지를 나타냅니다.</p>
          <div class="metric-guide ${guide.good ? 'good' : ''}">${escapeHtml(guide.text)}</div>
        </div>
        ${(equitability.axes || []).map(axis => {
          const bins = axis.bins;
          const maxCount = Math.max(1, ...bins.map(bin => bin.count));
          const width = 320;
          const height = 150;
          const gap = 2;
          const barWidth = Math.max(2, (width - gap * (bins.length - 1)) / Math.max(bins.length, 1));
          const bars = bins.map((bin, index) => {
            const barHeight = bin.count / maxCount * (height - 18);
            const x = index * (barWidth + gap);
            const y = height - barHeight;
            return `<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="2" fill="${bin.count ? '#117865' : '#dbe5eb'}"></rect>`;
          }).join('');
          const balanced = axis.status === 'balanced';
          return `
            <div class="equity-axis">
              <h4>${escapeHtml(axis.label || axis.axis)} <span class="${balanced ? 'status-balanced' : 'status-imbalanced'}">${balanced ? '균형' : '불균형'}</span></h4>
              <svg class="equity-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${bars}</svg>
              <div class="coverage-meta">X축은 Resolution으로 나뉜 bin 번호, Y축은 해당 bin 안의 sample 개수입니다.</div>
            </div>
          `;
        }).join('')}
      `;
    }

    function renderConfidenceReasons(report) {
      const descriptions = {
        'Sample Size': '현재 데이터와 같은 실험 목적을 가진 비교 기준 데이터 집단의 충분성을 반영합니다.',
        'Coverage': '실험 목적상 가능한 전체 범위 중, 비교 데이터가 실제로 채운 범위의 비율입니다.',
        'Equitability': '채워진 구간 안에서 데이터가 얼마나 균등하게 분포하는지를 나타냅니다.',
        'Engine Robustness': '비정규 분포에서도 분석 엔진의 효율이 얼마나 유지되는지를 반영합니다.',
      };
      document.getElementById('confidence-reasons').innerHTML = (report.confidenceReasons || []).map(item => {
        const guide = metricGuide(item.label, item.score);
        return `
          <div class="reason ${item.impact === 'down' ? 'down' : ''}">
            <strong>${escapeHtml(item.label)} · ${escapeHtml(formatPercent(item.score, item.label === 'Coverage' ? 1 : 0))}</strong>
            <p class="section-copy">${escapeHtml(descriptions[item.label] || item.message)}</p>
            <div class="metric-caption">${escapeHtml(item.message)}</div>
            <div class="metric-guide ${guide.good ? 'good' : ''}">${escapeHtml(guide.text)}</div>
          </div>
        `;
      }).join('');
    }

    function renderReport(report, options = {}) {
      if (!report) {
        renderReportSkeleton();
        return;
      }
      state.report = report;
      const reportSection = document.getElementById('report-section');
      renderScores(report);
      renderContribution(report);
      document.getElementById('summary-list').innerHTML = buildInterpretationLines(report).map(item => `<div class="pill">${escapeHtml(item)}</div>`).join('');
      renderConfidenceReasons(report);
      renderSampleSize(report);
      renderCoverage(report);
      renderEquitability(report);
      reportSection.classList.remove('ready');
      void reportSection.offsetWidth;
      reportSection.classList.add('ready');
      if (options.scroll !== false) {
        reportSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }

    async function runAnalysis() {
      const errorBox = document.getElementById('user-error-box');
      const button = document.getElementById('run-analysis');
      const goal = goalById(state.selectedGoalId);
      errorBox.style.display = 'none';
      if (!isMappingComplete()) {
        errorBox.textContent =
          rowCountWarning(goal) ||
          (selectedAxisNames(goal).length === 0
            ? '분석에 포함할 축을 하나 이상 체크하세요.'
            : hasDuplicateMappings(goal)
              ? '하나의 CSV 헤더를 여러 Axis에 중복 연결할 수 없습니다.'
              : '체크된 축의 컬럼 매핑과 Primary Key 선택을 먼저 완료해주세요.');
        errorBox.style.display = 'block';
        return;
      }
      const originalLabel = '분석';
      button.disabled = true;
      button.textContent = '분석 중...';
      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            goalId: state.selectedGoalId,
            rows: state.rows,
            selectedAxes: selectedAxisNames(goal),
            axisMapping: state.axisMapping,
            primaryKey: state.primaryKey,
          }),
        });
        const data = await response.json();
        if (!response.ok) {
          errorBox.textContent = data.error || '분석 요청 중 오류가 발생했습니다.';
          errorBox.style.display = 'block';
          return;
        }
        renderReport(data);
      } finally {
        button.textContent = originalLabel;
        updateAnalyzeButton();
      }
    }

    async function saveGoal() {
      state.draftGoal.name = document.getElementById('admin-goal-name').value.trim();
      state.draftGoal.K_m = Number(document.getElementById('admin-km-input').value || state.draftGoal.K_m || 10);
      const response = await fetch('/api/admin/goals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.draftGoal),
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.error || 'Goal 저장 중 오류가 발생했습니다.');
        return;
      }
      state.goals = data.goals;
      state.selectedGoalId = data.savedGoal.id;
      state.draftGoal = clone(data.savedGoal);
      state.adminAdvancedOpen = false;
      resetUserSelection(goalById(state.selectedGoalId));
      renderAll();
    }

    async function deleteGoal() {
      const response = await fetch('/api/admin/goals/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: state.draftGoal.id }),
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.error || 'Goal 삭제 중 오류가 발생했습니다.');
        return;
      }
      state.goals = data.goals;
      state.selectedGoalId = state.goals[0]?.id || null;
      state.draftGoal = clone(goalById(state.selectedGoalId) || blankGoal());
      state.adminAdvancedOpen = false;
      resetUserSelection(goalById(state.selectedGoalId));
      renderAll();
    }

    function renderAll() {
      buildReportScaffold();
      renderGoalSelects();
      renderAdmin();
      renderMapping();
      document.getElementById('settings-button').style.display = state.adminAllowed ? '' : 'none';
      if (state.report) {
        renderReport(state.report, { scroll: false });
      } else {
        renderReportSkeleton();
      }
    }

    document.getElementById('settings-button').addEventListener('click', () => document.getElementById('settings-modal').showModal());
    document.getElementById('settings-close').addEventListener('click', () => document.getElementById('settings-modal').close());
    document.getElementById('goal-select').addEventListener('change', event => {
      state.selectedGoalId = event.target.value;
      const goal = goalById(state.selectedGoalId);
      state.draftGoal = clone(goal || blankGoal());
      resetUserSelection(goal);
      renderAll();
    });
    document.getElementById('file-input').addEventListener('change', event => {
      const file = event.target.files?.[0];
      if (file) handleFileUpload(file);
    });
    document.getElementById('mapping-box').addEventListener('change', event => {
      const toggleAxis = event.target.dataset.axisToggle;
      const axis = event.target.dataset.axisMap;
      if (toggleAxis) {
        state.selectedAxes[toggleAxis] = event.target.checked;
        if (!event.target.checked && state.primaryKey === toggleAxis) {
          state.primaryKey = '';
        }
      }
      if (axis) {
        state.axisMapping[axis] = event.target.value;
      }
      if (event.target.name === 'primary-key') {
        state.primaryKey = event.target.value;
      }
      state.report = null;
      renderMapping();
    });
    document.getElementById('run-analysis').addEventListener('click', runAnalysis);
    document.getElementById('admin-goal-select').addEventListener('change', event => {
      state.draftGoal = clone(goalById(event.target.value) || blankGoal());
      state.adminAdvancedOpen = false;
      renderAdmin();
    });
    document.getElementById('admin-goal-name').addEventListener('input', event => {
      state.draftGoal.name = event.target.value;
    });
    document.getElementById('axis-editor').addEventListener('input', event => {
      const field = event.target.dataset.axisField;
      if (!field) return;
      const axis = state.draftGoal.axes[Number(event.target.dataset.index)];
      if (!axis) return;
      axis[field] = field === 'name' || field === 'unit' ? event.target.value : Number(event.target.value);
    });
    document.getElementById('axis-editor').addEventListener('click', event => {
      const index = event.target.dataset.axisRemove;
      if (index === undefined) return;
      state.draftGoal.axes.splice(Number(index), 1);
      if (state.draftGoal.axes.length === 0) {
        state.draftGoal.axes.push(blankAxis());
      }
      renderAdmin();
    });
    document.getElementById('axis-add-button').addEventListener('click', () => {
      state.draftGoal.axes.push(blankAxis());
      renderAdmin();
    });
    document.getElementById('admin-new-goal').addEventListener('click', () => {
      state.draftGoal = blankGoal();
      state.adminAdvancedOpen = false;
      renderGoalSelects();
      renderAdmin();
    });
    document.getElementById('advanced-toggle').addEventListener('click', () => {
      state.adminAdvancedOpen = !state.adminAdvancedOpen;
      renderAdmin();
    });
    document.getElementById('admin-km-input').addEventListener('input', event => {
      state.draftGoal.K_m = Number(event.target.value || state.draftGoal.K_m || 10);
    });
    document.getElementById('admin-save-goal').addEventListener('click', saveGoal);
    document.getElementById('admin-delete-goal').addEventListener('click', deleteGoal);
    renderAll();
  </script>
</body>
</html>
"""


class AppHandler(BaseHTTPRequestHandler):
    def _admin_allowed(self) -> bool:
        if os.environ.get("ALLOW_REMOTE_ADMIN", "").lower() in {"1", "true", "yes", "on"}:
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
            if parsed.path == "/api/admin/goals":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin endpoints are only available on localhost."}, status=HTTPStatus.FORBIDDEN)
                    return
                saved_goal = validate_goal(payload)
                goals = load_goal_store()
                existing_index = next((index for index, item in enumerate(goals) if item["id"] == saved_goal["id"]), None)
                if existing_index is None:
                    goals.append(saved_goal)
                else:
                    goals[existing_index] = saved_goal
                save_goal_store(goals)
                self._send_json({"savedGoal": saved_goal, "goals": goals})
                return
            if parsed.path == "/api/admin/goals/delete":
                if not self._admin_allowed():
                    self._send_json({"error": "Admin endpoints are only available on localhost."}, status=HTTPStatus.FORBIDDEN)
                    return
                goal_id = payload.get("id")
                goals = [goal for goal in load_goal_store() if goal["id"] != goal_id]
                if not goals:
                    goals = json.loads(json.dumps(DEFAULT_GOALS, ensure_ascii=False))
                save_goal_store(goals)
                self._send_json({"goals": goals})
                return
            if parsed.path == "/api/analyze":
                self._send_json(analyze_request(payload))
                return
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def render_page(admin_allowed: bool) -> str:
    return PAGE_HTML.replace("__BOOTSTRAP__", json.dumps(build_bootstrap_payload(admin_allowed=admin_allowed), ensure_ascii=False))


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Serving Data Quality Analyzer at http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Data Quality Certification web app.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
