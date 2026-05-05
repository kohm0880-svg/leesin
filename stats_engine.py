from __future__ import annotations

import json
import math

import numpy as np
from numpy.linalg import LinAlgError, pinv
from scipy import stats
from sklearn.covariance import LedoitWolf

from models import DiagnosisResult, ExperimentConfig


class BinGridTracker:
    """Tracks occupied multidimensional bins with a hashmap keyed by bin coordinates."""

    def __init__(self, domain_range: list[tuple[float, float]], resolution: list[float]):
        self.domain_range = domain_range
        self.resolution = resolution
        self._bins: dict[str, int] = {}

    def _bin_index(self, value: float, lo: float, hi: float, step: float) -> int:
        clipped = min(max(value, lo), hi - np.finfo(float).eps)
        return int(np.floor((clipped - lo) / step))

    def bin_indices(self, row: np.ndarray | list[float]) -> list[int]:
        return [
            self._bin_index(float(value), lo, hi, step)
            for value, (lo, hi), step in zip(row, self.domain_range, self.resolution)
        ]

    def bin_key(self, row: np.ndarray | list[float]) -> str:
        return json.dumps(self.bin_indices(row), separators=(",", ":"))

    def add(self, row: np.ndarray) -> None:
        key = self.bin_key(row)
        self._bins[key] = self._bins.get(key, 0) + 1

    def add_bin_counts(self, bin_counts: dict[str, int]) -> None:
        for key, count in bin_counts.items():
            try:
                indices = json.loads(str(key))
            except json.JSONDecodeError:
                continue
            if not isinstance(indices, list) or len(indices) != len(self.domain_range):
                continue
            try:
                normalized_indices = [int(index) for index in indices]
                increment = int(count)
            except (TypeError, ValueError):
                continue
            axis_totals = [
                max(1, int(math.ceil((hi - lo) / step)))
                for (lo, hi), step in zip(self.domain_range, self.resolution)
            ]
            if any(index < 0 or index >= total for index, total in zip(normalized_indices, axis_totals)):
                continue
            normalized_key = json.dumps(normalized_indices, separators=(",", ":"))
            if increment <= 0:
                continue
            self._bins[normalized_key] = self._bins.get(normalized_key, 0) + increment

    @classmethod
    def from_cluster_occupancies(
        cls,
        domain_range: list[tuple[float, float]],
        resolution: list[float],
        occupancies: list[dict[str, int]],
    ) -> "BinGridTracker":
        tracker = cls(domain_range, resolution)
        for bin_counts in occupancies:
            tracker.add_bin_counts(bin_counts)
        return tracker

    def count_for(self, row: np.ndarray | list[float]) -> int:
        return self._bins.get(self.bin_key(row), 0)

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
    def observation_count(self) -> int:
        return int(sum(self._bins.values()))

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


def sherman_morrison_update(inverse: np.ndarray, vector: np.ndarray) -> np.ndarray:
    column = vector.reshape(-1, 1)
    numerator = inverse @ column @ column.T @ inverse
    denominator = float(1.0 + (column.T @ inverse @ column).item())
    if abs(denominator) < 1e-12:
        raise LinAlgError("Sherman-Morrison update became numerically unstable.")
    return inverse - numerator / denominator


def regularized_sscm_inverse(X: np.ndarray, center: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    shifted = X - center
    norms = np.linalg.norm(shifted, axis=1, keepdims=True)
    unit_vectors = np.divide(shifted, np.where(norms < 1e-12, 1.0, norms))
    inverse = np.eye(X.shape[1], dtype=float) / ridge
    scale = math.sqrt(max(len(unit_vectors), 1))
    for vector in unit_vectors / scale:
        inverse = sherman_morrison_update(inverse, vector)
    return inverse


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
        self._coverage_grid = BinGridTracker(config.domain_range, config.resolution)

    def add_peers(self, X: np.ndarray | list[list[float]]) -> None:
        rows = np.asarray(X, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != self.p:
            raise ValueError(f"Peer data must have shape (N, {self.p}).")
        for row in rows:
            vector = np.asarray(row, dtype=float)
            self._peers.append(vector)

    def add_coverage_bin_counts(self, bin_counts: dict[str, int]) -> None:
        self._coverage_grid.add_bin_counts(bin_counts)

    def set_coverage_bin_counts(self, global_bin_counts: dict[str, int]) -> None:
        self._coverage_grid = BinGridTracker(self.config.domain_range, self.config.resolution)
        self._coverage_grid.add_bin_counts(global_bin_counts)

    def _select_engine(self, X: np.ndarray) -> tuple[str, bool | None, dict[str, float | bool]]:
        if len(X) < 10 or self.p < 2:
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
            try:
                covariance_inv = np.linalg.inv(covariance)
            except LinAlgError:
                covariance_inv = pinv(covariance)
        else:
            center = spatial_median(X)
            covariance_inv = regularized_sscm_inverse(X, center)

        # D2 = (x - center)^T Sigma^-1 (x - center).
        # p_value = 1 - F_chi2,p(D2), heterogeneity = F_chi2,p(D2).
        # Therefore larger D2 always produces larger heterogeneity.
        diff = x_target - center
        D2 = float(diff @ covariance_inv @ diff)
        p_value = float(1.0 - stats.chi2.cdf(D2, df=p))

        # Contribution decomposition includes covariance/SSCM interactions:
        # raw_j = d_j * (Sigma^-1 d)_j, percent_j = abs(raw_j) / sum(abs(raw)).
        contribution_raw = diff * (covariance_inv @ diff)
        contributions = (np.abs(contribution_raw) / (np.abs(contribution_raw).sum() + 1e-12)) * 100
        return center, D2, p_value, contributions

    def diagnose(self, x_target: np.ndarray | list[float]) -> DiagnosisResult:
        if not self._peers:
            raise ValueError("Peer Group is empty.")

        target = np.asarray(x_target, dtype=float)
        if target.shape != (self.p,):
            raise ValueError(f"Target vector must have shape ({self.p},).")

        X = np.asarray(self._peers, dtype=float)
        count = len(X)
        if count < self.p + 1:
            raise ValueError(
                f"Peer Group N={count} is too small for p={self.p}. "
                "Add more saved clusters or reduce the selected axes."
            )

        engine, is_normal, mardia = self._select_engine(X)
        if engine != "spatial_rank" and count <= self.p + 1:
            raise ValueError(
                f"Peer Group N={count} is too small for stable covariance-based judgment with p={self.p}. "
                "Add more peer clusters or reduce selected axes."
            )

        b2p = float(mardia["b2p"]) if "b2p" in mardia else None
        w_eff = self._w_eff(engine, b2p)
        center, D2, p_value, contributions = self._compute_heterogeneity(target, X, engine)

        # K_m is a Michaelis-Menten-shaped half-saturation constant:
        # Z = N / (N + K_m), so Z reaches 0.5 when N == K_m.
        sample_size_Z = count / (count + self.config.K_m)
        coverage_C = self._coverage_grid.coverage
        equitability_E = self._coverage_grid.equitability
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
            total_bins=self._coverage_grid.total_bins,
            occupied_bins=self._coverage_grid.occupied_bins,
            mardia_skew_stat=float(mardia["skew_stat"]) if "skew_stat" in mardia else None,
            mardia_skew_pval=float(mardia["skew_pval"]) if "skew_pval" in mardia else None,
            mardia_kurt_stat=float(mardia["kurt_stat"]) if "kurt_stat" in mardia else None,
            mardia_kurt_pval=float(mardia["kurt_pval"]) if "kurt_pval" in mardia else None,
            b2p=b2p,
        )
