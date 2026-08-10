"""Small deterministic V3 math checks.

Run from the repository root:

    python -m experiments.v3_math_validation

The script deliberately demonstrates that the current right-eCDF one-row
sensitivity does not vanish when all reference counts are scaled upward.  It
therefore treats exact one-row sensitivity as a diagnostic/certificate, not as
an already-validated Sample Sufficiency score.
"""

from __future__ import annotations

import csv
from pathlib import Path

from v3_metrics import (
    compute_evenness_candidates,
    exact_one_row_sensitivity,
    expected_total_variation_upper_bound,
)


def main() -> None:
    rows: list[dict[str, float | int | str]] = []
    target = {"a": 1}

    for count_per_bin in (10, 100, 1_000, 10_000):
        reference = {
            "a": count_per_bin,
            "b": count_per_bin,
            "c": count_per_bin,
            "d": count_per_bin,
        }
        n = sum(reference.values())
        for method in ("v2_right", "midrank"):
            certificate = exact_one_row_sensitivity(
                reference,
                target,
                valid_bin_count=4,
                method=method,
            )
            rows.append(
                {
                    "scenario": "uniform_four_bins",
                    "specificity_method": method,
                    "count_per_bin": count_per_bin,
                    "N": n,
                    "baseline_specificity": certificate.baseline_score,
                    "exact_one_row_max_change": certificate.max_change,
                    "display_stability_1_minus_delta": certificate.stability_score,
                    "expected_tv_upper_bound": expected_total_variation_upper_bound(n, 4),
                }
            )

    concentrated = compute_evenness_candidates({"a": 397, "b": 1, "c": 1, "d": 1})
    uniform = compute_evenness_candidates({"a": 100, "b": 100, "c": 100, "d": 100})

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v3_math_validation.csv"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("V3 candidate validation")
    print("=" * 78)
    print("Uniform four-bin rank-jump counterexample")
    for row in rows:
        print(
            f"method={row['specificity_method']:>9}  "
            f"N={row['N']:>6}  baseline={row['baseline_specificity']:.3f}  "
            f"exact delta_1={row['exact_one_row_max_change']:.3f}  "
            f"TV bound={row['expected_tv_upper_bound']:.4f}"
        )
    print()
    print("Evenness candidates at the same N=400 and b=4")
    print(
        "uniform      : "
        f"Pielou={uniform.pielou_evenness:.4f}, "
        f"Simpson E2={uniform.simpson_effective_evenness:.4f}"
    )
    print(
        "concentrated : "
        f"Pielou={concentrated.pielou_evenness:.4f}, "
        f"Simpson E2={concentrated.simpson_effective_evenness:.4f}"
    )
    print()
    print(f"CSV written to: {output_path}")


if __name__ == "__main__":
    main()
