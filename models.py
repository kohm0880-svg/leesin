from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
