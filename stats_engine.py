from __future__ import annotations

import json
import math

import numpy as np

from models import DensityDiagnosisResult, ExperimentConfig


def _normalize_axis_name(name: str) -> str:
    return str(name or "").strip().lower()


def canonical_experiment_config(config: ExperimentConfig) -> ExperimentConfig:
    items = sorted(
        zip(config.axis_names, config.domain_range, config.resolution),
        key=lambda item: _normalize_axis_name(item[0]),
    )
    return ExperimentConfig(
        axis_names=[axis_name for axis_name, _domain, _step in items],
        domain_range=[domain for _axis_name, domain, _step in items],
        resolution=[step for _axis_name, _domain, step in items],
        K_m=config.K_m,
    )


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
    def bin_counts(self) -> dict[str, int]:
        return dict(self._bins)

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


class DensityGridAnalyzer:
    def __init__(self, config: ExperimentConfig):
        self.config = canonical_experiment_config(config)
        self._peer_density = BinGridTracker(self.config.domain_range, self.config.resolution)

    def set_peer_bin_counts(self, bin_counts: dict[str, int]) -> None:
        self._peer_density = BinGridTracker(self.config.domain_range, self.config.resolution)
        self._peer_density.add_bin_counts(bin_counts)

    def add_peer_bin_counts(self, bin_counts: dict[str, int]) -> None:
        self._peer_density.add_bin_counts(bin_counts)

    def diagnose(
        self,
        target_bin_counts: dict[str, int],
        target_meta: dict[str, int] | None = None,
    ) -> DensityDiagnosisResult:
        peer_bin_counts = self._peer_density.bin_counts
        peer_valid_rows = self._peer_density.observation_count
        if peer_valid_rows <= 0:
            raise ValueError(
                "Density grid analysis requires at least one peer row-level bin occupancy observation. "
                "Saved legacy clusters without row-level bin occupancy/grid signature are excluded."
            )

        target_tracker = BinGridTracker(self.config.domain_range, self.config.resolution)
        target_tracker.add_bin_counts(target_bin_counts)
        normalized_target_counts = target_tracker.bin_counts
        target_valid_rows = target_tracker.observation_count

        meta = target_meta or {}
        invalid_target_rows = int(meta.get("invalidRowCount") or 0)
        out_of_domain_rows = int(meta.get("outOfDomainRowCount") or 0)
        target_total_rows = int(meta.get("totalRows") or (target_valid_rows + invalid_target_rows + out_of_domain_rows))
        if target_valid_rows <= 0:
            raise ValueError("Density grid analysis requires at least one valid target row-level observation.")

        total_bins = self._peer_density.total_bins
        alpha = 0.5
        denominator = peer_valid_rows + alpha * total_bins
        weighted_rarity = 0.0
        max_rarity = 0.0
        unseen_target_rows = 0
        rare_target_rows = 0

        for key, target_count in normalized_target_counts.items():
            peer_count = int(peer_bin_counts.get(key, 0))
            p_hat_g = (peer_count + alpha) / denominator
            rarity_g = -math.log(p_hat_g)
            weighted_rarity += int(target_count) * rarity_g
            max_rarity = max(max_rarity, rarity_g)
            if peer_count == 0:
                unseen_target_rows += int(target_count)
            if peer_count <= 1:
                rare_target_rows += int(target_count)

        mean_rarity = weighted_rarity / target_valid_rows
        scale = max(1e-12, math.log(peer_valid_rows + total_bins + 1))
        specificity_score = 1.0 - math.exp(-mean_rarity / scale)

        observation_support_S = peer_valid_rows / (peer_valid_rows + self.config.K_m)
        coverage_C = self._peer_density.coverage
        equitability_E = self._peer_density.equitability
        confidence = float((observation_support_S * coverage_C * equitability_E) ** (1.0 / 3.0))

        return DensityDiagnosisResult(
            engine="density_grid",
            specificity_score=float(specificity_score),
            mean_rarity=float(mean_rarity),
            max_rarity=float(max_rarity),
            unseen_bin_rate=float(unseen_target_rows / target_valid_rows),
            rare_bin_rate=float(rare_target_rows / target_valid_rows),
            out_of_domain_rate=float(out_of_domain_rows / target_total_rows) if target_total_rows > 0 else 0.0,
            observation_support_S=float(observation_support_S),
            coverage_C=float(coverage_C),
            equitability_E=float(equitability_E),
            confidence=confidence,
            peer_observation_count=int(peer_valid_rows),
            valid_target_rows=int(target_valid_rows),
            invalid_target_rows=int(invalid_target_rows),
            out_of_domain_rows=int(out_of_domain_rows),
            target_total_rows=int(target_total_rows),
            total_bins=int(total_bins),
            occupied_bins=int(self._peer_density.occupied_bins),
        )
