from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v3_metrics import (
    compute_evenness_candidates,
    compute_raw_coverage,
    expected_total_variation_upper_bound,
)
from v3_rank_stability import target_rank_stability


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT = RESULTS_DIR / "v3_rank_stability_validation.csv"
SUMMARY = RESULTS_DIR / "v3_rank_stability_summary.csv"


def counts(values):
    return {f"g{i}": int(v) for i, v in enumerate(values)}


def add_case(rows, family, case, values, target_bin="g0", valid_bin_count=None):
    ref = counts(values)
    if valid_bin_count is None:
        valid_bin_count = len(ref)

    N = sum(ref.values())
    even = compute_evenness_candidates(ref)
    C = compute_raw_coverage(ref, valid_bin_count=valid_bin_count)
    tv = expected_total_variation_upper_bound(N, valid_bin_count)
    rank = target_rank_stability(ref, {target_bin: 1})

    rows.append({
        "family": family,
        "case": case,
        "target_bin": target_bin,
        "N": N,
        "valid_bin_count_A": valid_bin_count,
        "coverage_C": C,
        "pielou_E": even.pielou_evenness,
        "simpson_effective_E": even.simpson_effective_evenness,
        "distribution_TV_upper_bound": tv,
        "distribution_stability_1_minus_TV": 1.0 - tv,
        "rank_stability_mean_nontrivial": rank.weighted_mean_nontrivial,
        "rank_stability_min_nontrivial": rank.weighted_minimum_nontrivial,
        "unseen_target_rows": rank.unseen_target_rows,
    })


def main():
    rows = []

    # A. Same C/E/shape, only N scales.
    for m in (1, 2, 5, 10, 50, 100, 1000):
        add_case(
            rows,
            "scale_same_shape",
            f"[1,2,3,4] x {m}",
            [1*m, 2*m, 3*m, 4*m],
            target_bin="g0",
        )

    # B. Perfect ties: E=1 but directional rank should remain ambiguous.
    for m in (1, 2, 5, 10, 50, 100, 1000):
        add_case(
            rows,
            "uniform_tie_scale",
            f"[1,1,1,1] x {m}",
            [m, m, m, m],
            target_bin="g0",
        )

    # C. Same N/C, different separation from the target bin.
    for name, vals in (
        ("near_tie", [249, 251, 250, 250]),
        ("moderate_gap", [200, 300, 250, 250]),
        ("large_gap", [100, 400, 250, 250]),
    ):
        add_case(rows, "gap_same_N_C", name, vals, target_bin="g0")

    # D. Same reference, different target bin.
    reference = [100, 101, 300, 499]
    for target_bin in ("g0", "g1", "g2", "g3"):
        add_case(
            rows,
            "target_specificity_same_reference",
            f"target={target_bin}",
            reference,
            target_bin=target_bin,
        )

    # E. Same N/A, different E. TV bound stays fixed while rank stability may change.
    for name, vals in (
        ("uniform", [250, 250, 250, 250]),
        ("graded", [400, 300, 200, 100]),
        ("concentrated", [700, 100, 100, 100]),
        ("extreme", [970, 10, 10, 10]),
    ):
        add_case(rows, "shape_same_N_A", name, vals, target_bin="g0")

    # F. Unseen target: do not fabricate certainty.
    ref = counts([250, 250, 250, 250])
    N = sum(ref.values())
    even = compute_evenness_candidates(ref)
    C = compute_raw_coverage(ref, valid_bin_count=5)
    tv = expected_total_variation_upper_bound(N, 5)
    rank = target_rank_stability(ref, {"g4": 1})
    rows.append({
        "family": "unseen_target",
        "case": "target in empty valid bin",
        "target_bin": "g4",
        "N": N,
        "valid_bin_count_A": 5,
        "coverage_C": C,
        "pielou_E": even.pielou_evenness,
        "simpson_effective_E": even.simpson_effective_evenness,
        "distribution_TV_upper_bound": tv,
        "distribution_stability_1_minus_TV": 1.0 - tv,
        "rank_stability_mean_nontrivial": rank.weighted_mean_nontrivial,
        "rank_stability_min_nontrivial": rank.weighted_minimum_nontrivial,
        "unseen_target_rows": rank.unseen_target_rows,
    })

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
        "check": "same C/E/shape, N increases",
        "expected": "mean rank stability increases toward 1",
        "passed": all(
            float(scale[i]["rank_stability_mean_nontrivial"])
            < float(scale[i+1]["rank_stability_mean_nontrivial"])
            for i in range(len(scale)-1)
        ),
    })

    ties = by_family["uniform_tie_scale"]
    checks.append({
        "check": "uniform ties as N grows",
        "expected": "rank stability does not falsely converge to 1",
        "passed": float(ties[-1]["rank_stability_mean_nontrivial"]) < 0.55,
    })

    gaps = by_family["gap_same_N_C"]
    checks.append({
        "check": "same N/C, larger density gap",
        "expected": "rank stability rises as target/comparison separation becomes clearer",
        "passed": (
            float(gaps[0]["rank_stability_mean_nontrivial"])
            < float(gaps[1]["rank_stability_mean_nontrivial"])
            < float(gaps[2]["rank_stability_mean_nontrivial"])
        ),
    })

    targets = by_family["target_specificity_same_reference"]
    vals = [float(r["rank_stability_mean_nontrivial"]) for r in targets]
    checks.append({
        "check": "same reference, different target bin",
        "expected": "rank stability can differ while C/E/N are identical",
        "passed": max(vals) - min(vals) > 0.05,
    })

    shape = by_family["shape_same_N_A"]
    tv_vals = [float(r["distribution_TV_upper_bound"]) for r in shape]
    rank_vals = [float(r["rank_stability_mean_nontrivial"]) for r in shape]
    checks.append({
        "check": "distribution stability vs rank stability are distinct",
        "expected": "TV bound stays fixed at same N/A while rank stability changes",
        "passed": (max(tv_vals)-min(tv_vals) < 1e-15) and (max(rank_vals)-min(rank_vals) > 0.05),
    })

    unseen = by_family["unseen_target"][0]
    checks.append({
        "check": "unseen target policy",
        "expected": "rank stability is undefined rather than falsely certain",
        "passed": unseen["rank_stability_mean_nontrivial"] is None and int(unseen["unseen_target_rows"]) == 1,
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
