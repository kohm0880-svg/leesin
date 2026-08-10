"""Leesin V3 candidate metrics and mathematical validation helpers.

This module is intentionally separate from the V2 production engine.
It does not alter V2 outputs.  Its first purpose is to test candidate
Specificity/equitability/stability definitions against exact finite-grid
perturbations before any candidate is promoted into the application.

The key stability quantity is the exact maximum change in a target's
Specificity after moving one external-reference row from one occupied valid
bin to any other valid bin.  Domain Range, Resolution, and the Feasible Mask
are represented by ``valid_bin_count``; the full grid is never materialized.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import log, sqrt
from typing import Hashable, Mapping


BinKey = Hashable
SUPPORTED_SPECIFICITY_METHODS = frozenset({"v2_right", "midrank"})


@dataclass(frozen=True)
class SpecificityResult:
    """Target-weighted bin Specificity result."""

    specificity_score: float
    max_specificity: float
    unseen_target_rows: int
    target_valid_rows: int
    bin_specificities: dict[BinKey, float]

    @property
    def unseen_bin_rate(self) -> float:
        return self.unseen_target_rows / self.target_valid_rows


@dataclass(frozen=True)
class OneRowStabilityResult:
    """Exact finite-grid certificate for one reference-row replacement.

    ``max_change`` is the exact maximum absolute Specificity change among all
    valid moves of one reference row.  ``stability_score`` is merely the
    display-direction transform ``1 - max_change``; it is not a probability
    and is not, by itself, a proof of population representativeness.
    """

    specificity_method: str
    baseline_score: float
    max_change: float
    stability_score: float
    certified_min_score: float
    certified_max_score: float
    evaluated_neighbor_count: int
    source_bin: str | None
    destination_bin: str | None
    valid_bin_count: int


@dataclass(frozen=True)
class EvennessCandidates:
    """Two V3 evenness candidates calculated on occupied bins."""

    occupied_bins: int
    observation_count: int
    pielou_evenness: float | None
    simpson_effective_bins: float | None
    simpson_effective_evenness: float | None


class _AnonymousEmptyBin:
    """Representative for any valid bin with no reference or target rows."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<anonymous-empty-valid-bin>"


_ANONYMOUS_EMPTY_BIN = _AnonymousEmptyBin()


def _positive_counts(counts: Mapping[BinKey, int]) -> dict[BinKey, int]:
    normalized: dict[BinKey, int] = {}
    for key, raw_count in counts.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Bin count for {key!r} is not an integer: {raw_count!r}") from exc
        if count < 0:
            raise ValueError(f"Bin count for {key!r} must be non-negative.")
        if count > 0:
            normalized[key] = count
    return normalized


def _validate_specificity_method(method: str) -> str:
    normalized = str(method or "").strip().lower()
    if normalized not in SUPPORTED_SPECIFICITY_METHODS:
        allowed = ", ".join(sorted(SUPPORTED_SPECIFICITY_METHODS))
        raise ValueError(f"Unknown specificity method {method!r}; expected one of: {allowed}.")
    return normalized


def compute_specificity(
    reference_bin_counts: Mapping[BinKey, int],
    target_bin_counts: Mapping[BinKey, int],
    *,
    method: str = "v2_right",
) -> SpecificityResult:
    """Compute target-weighted occupied-bin-count eCDF Specificity.

    ``v2_right`` reproduces the current V2 rule: all ties are accumulated on
    the right side of the eCDF (``count(C_j <= C_i)``).

    ``midrank`` gives tied count values their middle rank:

        F_mid(c) = [#(C_j < c) + 0.5 #(C_j = c)] / b

    A target bin unseen in the reference receives the boundary score 1.0 in
    both methods.  The result is an engineering rank score, not a probability.
    """

    method = _validate_specificity_method(method)
    reference = _positive_counts(reference_bin_counts)
    target = _positive_counts(target_bin_counts)

    if not reference:
        raise ValueError("Specificity requires at least one positive reference-bin count.")
    if not target:
        raise ValueError("Specificity requires at least one positive target-bin count.")

    occupied_counts = sorted(reference.values())
    occupied_bin_count = len(occupied_counts)

    weighted_score = 0.0
    max_specificity = 0.0
    unseen_target_rows = 0
    target_valid_rows = 0
    bin_specificities: dict[BinKey, float] = {}

    for key, target_count in target.items():
        target_valid_rows += target_count
        reference_count = reference.get(key, 0)

        if reference_count <= 0:
            specificity = 1.0
            unseen_target_rows += target_count
        elif method == "v2_right":
            right_rank = bisect_right(occupied_counts, reference_count)
            specificity = 1.0 - (right_rank / occupied_bin_count)
        else:
            left_rank = bisect_left(occupied_counts, reference_count)
            right_rank = bisect_right(occupied_counts, reference_count)
            tie_count = right_rank - left_rank
            mid_cdf = (left_rank + 0.5 * tie_count) / occupied_bin_count
            specificity = 1.0 - mid_cdf

        specificity = min(1.0, max(0.0, float(specificity)))
        bin_specificities[key] = specificity
        weighted_score += target_count * specificity
        max_specificity = max(max_specificity, specificity)

    return SpecificityResult(
        specificity_score=weighted_score / target_valid_rows,
        max_specificity=max_specificity,
        unseen_target_rows=unseen_target_rows,
        target_valid_rows=target_valid_rows,
        bin_specificities=bin_specificities,
    )


def _move_one_row(
    reference: Mapping[BinKey, int],
    source: BinKey,
    destination: BinKey,
) -> dict[BinKey, int]:
    moved = dict(reference)
    source_count = moved.get(source, 0)
    if source_count <= 0:
        raise ValueError("The source bin must contain at least one reference row.")
    if source == destination:
        return moved

    if source_count == 1:
        moved.pop(source, None)
    else:
        moved[source] = source_count - 1
    moved[destination] = moved.get(destination, 0) + 1
    return moved


def _display_key(key: BinKey | None) -> str | None:
    if key is None:
        return None
    if key is _ANONYMOUS_EMPTY_BIN:
        return repr(_ANONYMOUS_EMPTY_BIN)
    return str(key)


def exact_one_row_sensitivity_bruteforce(
    reference_bin_counts: Mapping[BinKey, int],
    target_bin_counts: Mapping[BinKey, int],
    *,
    valid_bin_count: int,
    method: str = "v2_right",
) -> OneRowStabilityResult:
    """Exact one-row replacement certificate by explicit known-bin search.

    The algorithm never materializes the full valid grid.  It considers every
    occupied source bin, every known reference/target bin as a destination,
    and one representative anonymous empty bin when additional valid bins
    exist.  All anonymous empty non-target bins are equivalent for this
    count-eCDF Specificity, so one representative is exact rather than an
    approximation.
    """

    method = _validate_specificity_method(method)
    reference = _positive_counts(reference_bin_counts)
    target = _positive_counts(target_bin_counts)

    if not reference:
        raise ValueError("One-row sensitivity requires at least one reference row.")
    if not target:
        raise ValueError("One-row sensitivity requires at least one target row.")

    valid_bin_count = int(valid_bin_count)
    if valid_bin_count <= 0:
        raise ValueError("valid_bin_count must be positive.")

    known_keys = set(reference) | set(target)
    if len(known_keys) > valid_bin_count:
        raise ValueError(
            "valid_bin_count is smaller than the number of distinct valid reference/target bins."
        )

    destination_keys: list[BinKey] = list(known_keys)
    if valid_bin_count > len(known_keys):
        destination_keys.append(_ANONYMOUS_EMPTY_BIN)

    baseline = compute_specificity(reference, target, method=method).specificity_score
    certified_min = baseline
    certified_max = baseline
    max_change = 0.0
    witness_source: BinKey | None = None
    witness_destination: BinKey | None = None
    evaluated = 0

    for source in reference:
        for destination in destination_keys:
            if source == destination:
                continue
            moved = _move_one_row(reference, source, destination)
            score = compute_specificity(moved, target, method=method).specificity_score
            evaluated += 1
            certified_min = min(certified_min, score)
            certified_max = max(certified_max, score)
            change = abs(score - baseline)
            if change > max_change + 1e-15:
                max_change = change
                witness_source = source
                witness_destination = destination

    max_change = min(1.0, max(0.0, max_change))
    return OneRowStabilityResult(
        specificity_method=method,
        baseline_score=baseline,
        max_change=max_change,
        stability_score=1.0 - max_change,
        certified_min_score=certified_min,
        certified_max_score=certified_max,
        evaluated_neighbor_count=evaluated,
        source_bin=_display_key(witness_source),
        destination_bin=_display_key(witness_destination),
        valid_bin_count=valid_bin_count,
    )


def _reduced_candidate_keys(
    reference: Mapping[BinKey, int],
    target: Mapping[BinKey, int],
    valid_bin_count: int,
) -> tuple[list[BinKey], list[BinKey]]:
    """Return an exact reduced source/destination key set.

    Every target bin is retained because changing its reference count directly
    changes a weighted target term.  Non-target occupied bins are equivalent
    when their counts are equal; at most two representatives per count are
    retained so a move between two distinct equal-count bins remains possible.
    """

    target_keys = set(target)
    non_target_by_count: dict[int, list[BinKey]] = {}
    for key, count in reference.items():
        if key not in target_keys:
            non_target_by_count.setdefault(count, []).append(key)

    non_target_representatives: list[BinKey] = []
    for keys in non_target_by_count.values():
        non_target_representatives.extend(keys[:2])

    source_keys = [key for key in target_keys if reference.get(key, 0) > 0]
    source_keys.extend(non_target_representatives)

    destination_keys: list[BinKey] = list(target_keys)
    destination_keys.extend(non_target_representatives)

    # Preserve order while removing duplicates.
    source_keys = list(dict.fromkeys(source_keys))
    destination_keys = list(dict.fromkeys(destination_keys))

    known_keys = set(reference) | target_keys
    if valid_bin_count > len(known_keys):
        destination_keys.append(_ANONYMOUS_EMPTY_BIN)

    return source_keys, destination_keys


def exact_one_row_sensitivity(
    reference_bin_counts: Mapping[BinKey, int],
    target_bin_counts: Mapping[BinKey, int],
    *,
    valid_bin_count: int,
    method: str = "v2_right",
) -> OneRowStabilityResult:
    """Exact one-row certificate using count-type reduction.

    This has the same mathematical result as
    :func:`exact_one_row_sensitivity_bruteforce`, but avoids comparing every
    pair of non-target bins when many bins share the same reference count.
    """

    method = _validate_specificity_method(method)
    reference = _positive_counts(reference_bin_counts)
    target = _positive_counts(target_bin_counts)

    if not reference:
        raise ValueError("One-row sensitivity requires at least one reference row.")
    if not target:
        raise ValueError("One-row sensitivity requires at least one target row.")

    valid_bin_count = int(valid_bin_count)
    if valid_bin_count <= 0:
        raise ValueError("valid_bin_count must be positive.")
    if len(set(reference) | set(target)) > valid_bin_count:
        raise ValueError(
            "valid_bin_count is smaller than the number of distinct valid reference/target bins."
        )

    source_keys, destination_keys = _reduced_candidate_keys(reference, target, valid_bin_count)
    baseline = compute_specificity(reference, target, method=method).specificity_score
    certified_min = baseline
    certified_max = baseline
    max_change = 0.0
    witness_source: BinKey | None = None
    witness_destination: BinKey | None = None
    evaluated = 0

    for source in source_keys:
        for destination in destination_keys:
            if source == destination:
                continue
            moved = _move_one_row(reference, source, destination)
            score = compute_specificity(moved, target, method=method).specificity_score
            evaluated += 1
            certified_min = min(certified_min, score)
            certified_max = max(certified_max, score)
            change = abs(score - baseline)
            if change > max_change + 1e-15:
                max_change = change
                witness_source = source
                witness_destination = destination

    max_change = min(1.0, max(0.0, max_change))
    return OneRowStabilityResult(
        specificity_method=method,
        baseline_score=baseline,
        max_change=max_change,
        stability_score=1.0 - max_change,
        certified_min_score=certified_min,
        certified_max_score=certified_max,
        evaluated_neighbor_count=evaluated,
        source_bin=_display_key(witness_source),
        destination_bin=_display_key(witness_destination),
        valid_bin_count=valid_bin_count,
    )


def compute_raw_coverage(
    reference_bin_counts: Mapping[BinKey, int],
    *,
    valid_bin_count: int,
) -> float:
    """Return occupied valid bins divided by all valid bins."""

    valid_bin_count = int(valid_bin_count)
    if valid_bin_count <= 0:
        raise ValueError("valid_bin_count must be positive.")
    occupied = len(_positive_counts(reference_bin_counts))
    if occupied > valid_bin_count:
        raise ValueError("Occupied bin count cannot exceed valid_bin_count.")
    return occupied / valid_bin_count


def compute_evenness_candidates(
    reference_bin_counts: Mapping[BinKey, int],
) -> EvennessCandidates:
    """Calculate Pielou and order-2 effective evenness candidates.

    V3 convention for one occupied bin is evenness 1.0: there is no imbalance
    *within the occupied set*.  The fact that the explored region is narrow is
    represented by Coverage instead.  With zero observations, evenness is
    undefined and all candidate values are ``None``.
    """

    reference = _positive_counts(reference_bin_counts)
    occupied = len(reference)
    observation_count = sum(reference.values())
    if observation_count <= 0:
        return EvennessCandidates(0, 0, None, None, None)
    if occupied == 1:
        return EvennessCandidates(1, observation_count, 1.0, 1.0, 1.0)

    proportions = [count / observation_count for count in reference.values()]
    entropy = -sum(p * log(p) for p in proportions)
    pielou = entropy / log(occupied)

    concentration = sum(p * p for p in proportions)
    effective_bins = 1.0 / concentration
    simpson_evenness = effective_bins / occupied

    return EvennessCandidates(
        occupied_bins=occupied,
        observation_count=observation_count,
        pielou_evenness=min(1.0, max(0.0, pielou)),
        simpson_effective_bins=effective_bins,
        simpson_effective_evenness=min(1.0, max(0.0, simpson_evenness)),
    )


def expected_total_variation_upper_bound(
    observation_count: int,
    valid_bin_count: int,
) -> float:
    """Distribution-free expected-TV upper bound under iid multinomial rows.

    For an empirical distribution on ``A`` valid bins with ``N`` iid rows,

        E[TV(P_hat, P)] <= 0.5 * sqrt((A - 1) / N).

    The return value is clipped to 1 because total variation is at most 1.
    This is an assumption-bearing comparison candidate, not an exact observed
    stability certificate.
    """

    observation_count = int(observation_count)
    valid_bin_count = int(valid_bin_count)
    if observation_count <= 0:
        raise ValueError("observation_count must be positive.")
    if valid_bin_count <= 0:
        raise ValueError("valid_bin_count must be positive.")
    if valid_bin_count == 1:
        return 0.0
    return min(1.0, 0.5 * sqrt((valid_bin_count - 1) / observation_count))
