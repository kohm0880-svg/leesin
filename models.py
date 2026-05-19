from __future__ import annotations

from dataclasses import dataclass
from typing import Any


K_M = 10.0


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
        if self.K_m <= 0:
            raise ValueError("K_m must be greater than 0.")
        for index, ((lo, hi), step) in enumerate(zip(self.domain_range, self.resolution)):
            if hi <= lo:
                raise ValueError(f"Axis {index} has an invalid Domain Range.")
            if step <= 0:
                raise ValueError(f"Axis {index} must have a positive Resolution.")


@dataclass
class DensityDiagnosisResult:
    engine: str
    specificity_score: float
    mean_bin_specificity: float
    max_specificity: float
    extreme_specificity_rate: float
    mean_rarity: float
    max_rarity: float
    unseen_bin_rate: float
    rare_bin_rate: float
    out_of_domain_rate: float
    observation_support_S: float
    coverage_C: float
    equitability_E: float
    confidence: float
    peer_observation_count: int
    valid_target_rows: int
    invalid_target_rows: int
    out_of_domain_rows: int
    masked_out_target_rows: int
    target_total_rows: int
    total_bins: int
    valid_bins: int
    masked_bins: int
    occupied_bins: int
    feasible_mask_enabled: bool

    def to_payload(self, axis_names: list[str]) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "axis_names": list(axis_names),
            "specificity_method": "occupied_bin_count_ecdf",
            "specificity_interpretation": "Higher means target rows fall in lower-density or unseen peer bins.",
            "specificity_score": round(float(self.specificity_score), 6),
            "mean_bin_specificity": round(float(self.mean_bin_specificity), 6),
            "max_specificity": round(float(self.max_specificity), 6),
            "extreme_specificity_rate": round(float(self.extreme_specificity_rate), 6),
            "mean_rarity": round(float(self.mean_rarity), 6),
            "max_rarity": round(float(self.max_rarity), 6),
            "unseen_bin_rate": round(float(self.unseen_bin_rate), 6),
            "rare_bin_rate": round(float(self.rare_bin_rate), 6),
            "out_of_domain_rate": round(float(self.out_of_domain_rate), 6),
            "observation_support_S": round(float(self.observation_support_S), 6),
            "coverage_C": round(float(self.coverage_C), 6),
            "equitability_E": round(float(self.equitability_E), 6),
            "confidence": round(float(self.confidence), 6),
            "peer_observation_count": int(self.peer_observation_count),
            "valid_target_rows": int(self.valid_target_rows),
            "invalid_target_rows": int(self.invalid_target_rows),
            "out_of_domain_rows": int(self.out_of_domain_rows),
            "masked_out_target_rows": int(self.masked_out_target_rows),
            "infeasible_target_rows": int(self.masked_out_target_rows),
            "target_total_rows": int(self.target_total_rows),
            "total_bins": self.total_bins,
            "valid_bins": self.valid_bins,
            "masked_bins": self.masked_bins,
            "valid_domain_ratio": round(float(self.valid_bins / self.total_bins), 6) if self.total_bins else 0.0,
            "occupied_bins": self.occupied_bins,
            "feasible_mask_enabled": bool(self.feasible_mask_enabled),
        }


DiagnosisResult = DensityDiagnosisResult
