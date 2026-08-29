from __future__ import annotations

import csv
from pathlib import Path
import sys

PATCH_ROOT = Path("/mnt/data/leesin_v3_candidate_patch")
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from v3_metrics import (
    compute_evenness_candidates,
    compute_raw_coverage,
    expected_total_variation_upper_bound,
    exact_one_row_sensitivity,
)

OUT = Path("/mnt/data/v3_sampling_stability_validation.csv")
SUMMARY = Path("/mnt/data/v3_sampling_stability_summary.csv")


def balanced_counts(total: int, bins: int, prefix: str = "g") -> dict[str, int]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if total < bins:
        raise ValueError("total must be >= bins")
    q, r = divmod(total, bins)
    return {f"{prefix}{i}": q + (1 if i < r else 0) for i in range(bins)}


def scaled_counts(base: list[int], multiplier: int) -> dict[str, int]:
    return {f"g{i}": int(v * multiplier) for i, v in enumerate(base)}


def add_case(rows, *, family, case, counts, valid_bin_count, local=False):
    N = sum(counts.values())
    even = compute_evenness_candidates(counts)
    C = compute_raw_coverage(counts, valid_bin_count=valid_bin_count)
    U_tv = expected_total_variation_upper_bound(N, valid_bin_count)
    S_tv = 1.0 - U_tv

    delta_right = None
    delta_mid = None
    if local:
        target = {next(iter(counts)): 1}
        delta_right = exact_one_row_sensitivity(
            counts, target, valid_bin_count=valid_bin_count, method="v2_right"
        ).max_change
        delta_mid = exact_one_row_sensitivity(
            counts, target, valid_bin_count=valid_bin_count, method="midrank"
        ).max_change

    rows.append({
        "family": family,
        "case": case,
        "valid_bin_count_A": valid_bin_count,
        "occupied_bins_b": even.occupied_bins,
        "N": N,
        "coverage_C": C,
        "pielou_E": even.pielou_evenness,
        "simpson_effective_E": even.simpson_effective_evenness,
        "expected_TV_upper_bound": U_tv,
        "sampling_stability_candidate_1_minus_TV_bound": S_tv,
        "one_row_delta_v2_right": delta_right,
        "one_row_delta_midrank": delta_mid,
    })


def main():
    rows = []

    for m in (1, 10, 100, 1000):
        add_case(
            rows,
            family="scale_same_shape",
            case=f"[1,2,3,4] x {m}",
            counts=scaled_counts([1, 2, 3, 4], m),
            valid_bin_count=4,
            local=True,
        )

    for m in (1, 10, 100, 1000, 10000):
        add_case(
            rows,
            family="uniform_scale",
            case=f"[1,1,1,1] x {m}",
            counts=scaled_counts([1, 1, 1, 1], m),
            valid_bin_count=4,
            local=True,
        )

    for name, base in (
        ("uniform", [250, 250, 250, 250]),
        ("graded", [400, 300, 200, 100]),
        ("concentrated", [700, 100, 100, 100]),
        ("extreme_concentration", [970, 10, 10, 10]),
    ):
        add_case(
            rows,
            family="shape_same_N_C",
            case=name,
            counts=scaled_counts(base, 1),
            valid_bin_count=4,
        )

    for A in (4, 10, 20, 50, 100, 200, 500, 1000):
        add_case(
            rows,
            family="grid_size_same_N_full_coverage",
            case=f"A={A}",
            counts=balanced_counts(1000, A),
            valid_bin_count=A,
        )

    A = 100
    for occupied in (5, 10, 20, 50, 100):
        add_case(
            rows,
            family="coverage_same_N_A_even",
            case=f"occupied={occupied}",
            counts=balanced_counts(1000, occupied, prefix="o"),
            valid_bin_count=A,
        )

    fieldnames = list(rows[0].keys())
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    by_family = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)

    checks = []

    scale = by_family["scale_same_shape"]
    checks.append({
        "check": "same shape/C/E, N increases",
        "expected": "sampling stability candidate increases",
        "passed": all(
            scale[i]["sampling_stability_candidate_1_minus_TV_bound"]
            < scale[i + 1]["sampling_stability_candidate_1_minus_TV_bound"]
            for i in range(len(scale) - 1)
        ),
    })

    shape = by_family["shape_same_N_C"]
    checks.append({
        "check": "same N/A/C, E changes",
        "expected": "TV candidate remains unchanged; it is not re-measuring E",
        "passed": max(r["expected_TV_upper_bound"] for r in shape)
                  - min(r["expected_TV_upper_bound"] for r in shape) < 1e-15,
    })

    grid = by_family["grid_size_same_N_full_coverage"]
    checks.append({
        "check": "same N/C≈1/E≈1, A increases",
        "expected": "sampling stability candidate decreases",
        "passed": all(
            grid[i]["sampling_stability_candidate_1_minus_TV_bound"]
            > grid[i + 1]["sampling_stability_candidate_1_minus_TV_bound"]
            for i in range(len(grid) - 1)
        ),
    })

    cov = by_family["coverage_same_N_A_even"]
    checks.append({
        "check": "same N/A/E=1, C changes",
        "expected": "TV candidate remains unchanged; it is not re-measuring C",
        "passed": max(r["expected_TV_upper_bound"] for r in cov)
                  - min(r["expected_TV_upper_bound"] for r in cov) < 1e-15,
    })

    uniform = by_family["uniform_scale"]
    checks.append({
        "check": "known one-row eCDF sensitivity failure",
        "expected": "for m>=10, right-eCDF delta stays 0.75 despite N growth",
        "passed": all(abs(r["one_row_delta_v2_right"] - 0.75) < 1e-12 for r in uniform[1:]),
    })

    with SUMMARY.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check", "expected", "passed"])
        w.writeheader()
        w.writerows(checks)

    print(f"Wrote {OUT}")
    print(f"Wrote {SUMMARY}")
    for c in checks:
        print(f"{'PASS' if c['passed'] else 'FAIL'} - {c['check']}")


if __name__ == "__main__":
    main()
