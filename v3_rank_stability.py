"""Exact eCDF comparison-retention stability candidates for Leesin V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class PairwiseComparisonStability:
    target_bin: Hashable
    comparison_bin: Hashable
    target_count: int
    comparison_count: int
    current_indicator: int
    retention_probability: float


@dataclass(frozen=True)
class BinRankStability:
    target_bin: Hashable
    target_count: int
    occupied_reference_bins: int
    mean_all_comparisons: float
    mean_nontrivial_comparisons: float | None
    minimum_nontrivial_comparison: float | None
    comparisons: tuple[PairwiseComparisonStability, ...]


@dataclass(frozen=True)
class TargetRankStability:
    weighted_mean_nontrivial: float | None
    weighted_minimum_nontrivial: float | None
    target_rows_evaluated: int
    unseen_target_rows: int
    per_bin: tuple[BinRankStability, ...]


def _validate_counts(reference_counts: Mapping[Hashable, int]) -> dict[Hashable, int]:
    cleaned: dict[Hashable, int] = {}
    for key, value in reference_counts.items():
        value = int(value)
        if value < 0:
            raise ValueError("reference counts must be nonnegative")
        if value > 0:
            cleaned[key] = value
    if not cleaned:
        raise ValueError("reference must contain at least one positive count")
    return cleaned


def probability_right_ecdf_indicator_retained(*, N: int, c_i: int, c_j: int) -> float:
    """Exact probability that V2's comparison I(c_j <= c_i) keeps its value.

    Empirical multinomial resampling is used. Conditional on the resampled
    number M landing in either bin i or j, C_i* is binomial.
    """
    N = int(N)
    c_i = int(c_i)
    c_j = int(c_j)
    if N <= 0:
        raise ValueError("N must be positive")
    if c_i < 0 or c_j < 0 or c_i + c_j > N:
        raise ValueError("invalid pair counts")

    pair_total = c_i + c_j
    if pair_total == 0:
        return 0.5

    q = pair_total / N
    theta = c_i / pair_total
    m = np.arange(N + 1, dtype=np.int64)
    pm = binom.pmf(m, N, q)

    if c_j <= c_i:
        # Retain indicator=1: C_j* <= C_i*.
        threshold = (m + 1) // 2
        cond = binom.sf(threshold - 1, m, theta)
    else:
        # Retain indicator=0: C_j* > C_i*.
        threshold = (m - 1) // 2
        cond = binom.cdf(threshold, m, theta)

    p = float(np.sum(pm * cond))
    return min(1.0, max(0.0, p))


def bin_rank_stability(
    reference_counts: Mapping[Hashable, int],
    *,
    target_bin: Hashable,
) -> BinRankStability | None:
    """Exact comparison-retention stability for one target bin.

    Unseen target bins return None: empirical resampling gives them zero
    probability, so assigning a numerical rank stability would create false
    certainty.
    """
    ref = _validate_counts(reference_counts)
    if target_bin not in ref:
        return None

    N = sum(ref.values())
    c_i = ref[target_bin]
    comparisons: list[PairwiseComparisonStability] = []

    for comparison_bin, c_j in ref.items():
        if comparison_bin == target_bin:
            retention = 1.0
        else:
            retention = probability_right_ecdf_indicator_retained(
                N=N, c_i=c_i, c_j=c_j
            )
        comparisons.append(
            PairwiseComparisonStability(
                target_bin=target_bin,
                comparison_bin=comparison_bin,
                target_count=c_i,
                comparison_count=c_j,
                current_indicator=int(c_j <= c_i),
                retention_probability=retention,
            )
        )

    all_probs = [c.retention_probability for c in comparisons]
    nontrivial = [
        c.retention_probability
        for c in comparisons
        if c.comparison_bin != target_bin
    ]

    return BinRankStability(
        target_bin=target_bin,
        target_count=c_i,
        occupied_reference_bins=len(ref),
        mean_all_comparisons=sum(all_probs) / len(all_probs),
        mean_nontrivial_comparisons=(sum(nontrivial) / len(nontrivial) if nontrivial else None),
        minimum_nontrivial_comparison=(min(nontrivial) if nontrivial else None),
        comparisons=tuple(comparisons),
    )


def target_rank_stability(
    reference_counts: Mapping[Hashable, int],
    target_counts: Mapping[Hashable, int],
) -> TargetRankStability:
    """Target-batch rank stability weighted by target row counts."""
    ref = _validate_counts(reference_counts)

    per_bin: list[BinRankStability] = []
    weighted_sum = 0.0
    evaluated_rows = 0
    unseen_rows = 0
    weighted_min = None

    for target_bin, raw_weight in target_counts.items():
        weight = int(raw_weight)
        if weight <= 0:
            continue

        result = bin_rank_stability(ref, target_bin=target_bin)
        if result is None:
            unseen_rows += weight
            continue

        per_bin.append(result)
        if result.mean_nontrivial_comparisons is not None:
            weighted_sum += weight * result.mean_nontrivial_comparisons
            evaluated_rows += weight
        if result.minimum_nontrivial_comparison is not None:
            weighted_min = (
                result.minimum_nontrivial_comparison
                if weighted_min is None
                else min(weighted_min, result.minimum_nontrivial_comparison)
            )

    return TargetRankStability(
        weighted_mean_nontrivial=(weighted_sum / evaluated_rows if evaluated_rows else None),
        weighted_minimum_nontrivial=weighted_min,
        target_rows_evaluated=evaluated_rows,
        unseen_target_rows=unseen_rows,
        per_bin=tuple(per_bin),
    )
