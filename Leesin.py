from __future__ import annotations

import argparse
import ipaddress
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
        "Missing dependency. Run this app with the virtual environment interpreter:\n"
        r"  C:\Leesin_project\Lee_sin.venv\Scripts\python.exe C:\Leesin_project\Leesin.py"
        "\nOr install requirements with:\n"
        r"  C:\Leesin_project\Lee_sin.venv\Scripts\python.exe -m pip install -r C:\Leesin_project\requirements.txt"
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
        "name": "고온 유량 품질 인증",
        "axes": [
            {"name": "temperature", "domainMin": 0.0, "domainMax": 200.0, "resolution": 10.0},
            {"name": "pressure", "domainMin": 0.0, "domainMax": 100.0, "resolution": 5.0},
            {"name": "flow_rate", "domainMin": 0.0, "domainMax": 50.0, "resolution": 2.5},
        ],
    },
    {
        "id": "goal_vacuum",
        "name": "진공 유지 품질 인증",
        "axes": [
            {"name": "vacuum_level", "domainMin": 0.0, "domainMax": 100.0, "resolution": 5.0},
            {"name": "hold_time", "domainMin": 0.0, "domainMax": 300.0, "resolution": 15.0},
            {"name": "leak_rate", "domainMin": 0.0, "domainMax": 20.0, "resolution": 1.0},
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


def load_goal_store() -> list[dict[str, Any]]:
    if GOAL_STORE_PATH.exists():
        return json.loads(GOAL_STORE_PATH.read_text(encoding="utf-8"))
    GOAL_STORE_PATH.write_text(json.dumps(DEFAULT_GOALS, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.loads(json.dumps(DEFAULT_GOALS, ensure_ascii=False))


def save_goal_store(goals: list[dict[str, Any]]) -> None:
    GOAL_STORE_PATH.write_text(json.dumps(goals, ensure_ascii=False, indent=2), encoding="utf-8")


def axis_signature(axis_names: list[str]) -> tuple[str, ...]:
    return tuple(name.strip().lower() for name in axis_names)


def validate_goal(goal: dict[str, Any]) -> dict[str, Any]:
    name = str(goal.get("name", "")).strip()
    if not name:
        raise ValueError("Experiment Goal name is required.")
    axes = goal.get("axes", [])
    if not isinstance(axes, list) or not axes:
        raise ValueError("At least one axis is required.")
    normalized_axes = []
    for axis in axes:
        axis_name = str(axis.get("name", "")).strip()
        domain_min = float(axis.get("domainMin"))
        domain_max = float(axis.get("domainMax"))
        resolution = float(axis.get("resolution"))
        normalized_axes.append(
            {
                "name": axis_name,
                "domainMin": domain_min,
                "domainMax": domain_max,
                "resolution": resolution,
            }
        )
    ExperimentConfig(
        axis_names=[axis["name"] for axis in normalized_axes],
        domain_range=[(axis["domainMin"], axis["domainMax"]) for axis in normalized_axes],
        resolution=[axis["resolution"] for axis in normalized_axes],
    )
    return {
        "id": str(goal.get("id") or f"goal_{abs(hash(name))}"),
        "name": name,
        "axes": normalized_axes,
    }


def build_bootstrap_payload(admin_allowed: bool) -> dict[str, Any]:
    goals = load_goal_store()
    return {
        "adminAllowed": admin_allowed,
        "goals": goals,
        "acceptedUploadTypes": [".csv", ".tsv", ".txt"],
        "stateShape": {
            "admin": ["selectedGoalId", "draftGoal"],
            "user": ["selectedGoalId", "fileName", "headers", "rows", "axisMapping", "primaryKey"],
            "report": ["status", "result", "meta", "summary", "confidenceReasons"],
        },
        "componentStructure": {
            "AppShell": ["ViewTabs", "AdminSetupView", "GeneralUserView", "ResultReportView"],
            "AdminSetupView": ["GoalSelector", "GoalEditorForm", "AxisTableEditor"],
            "GeneralUserView": ["GoalDropdown", "UploadPanel", "ColumnMapper", "PrimaryKeySelector"],
            "ResultReportView": ["MetricCards", "ContributionChart", "ConfidenceDiagnostics"],
        },
    }


def pick_peer_group(goal: dict[str, Any]) -> np.ndarray:
    signature = axis_signature([axis["name"] for axis in goal["axes"]])
    peer_rows = PEER_GROUP_LIBRARY.get(signature)
    if peer_rows is None:
        raise ValueError("No automatic peer group could be matched for this Experiment Goal.")
    return np.asarray(peer_rows, dtype=float)


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

    primary_key = str(payload.get("primaryKey", "")).strip()
    if primary_key not in [axis["name"] for axis in goal["axes"]]:
        raise ValueError("Primary Key must be one of the configured axes.")

    config = ExperimentConfig(
        axis_names=[axis["name"] for axis in goal["axes"]],
        domain_range=[(axis["domainMin"], axis["domainMax"]) for axis in goal["axes"]],
        resolution=[axis["resolution"] for axis in goal["axes"]],
    )

    target_vector, dataset_meta = build_target_vector(rows, axis_mapping, goal)
    peer_group = pick_peer_group(goal)

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
            "config": asdict(config),
        },
        "result": result.to_payload(config.axis_names),
        "summary": summary,
        "confidenceReasons": confidence_reasons(result),
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
