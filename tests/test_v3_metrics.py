from __future__ import annotations

import random
import unittest

from v3_metrics import (
    compute_evenness_candidates,
    compute_raw_coverage,
    compute_specificity,
    exact_one_row_sensitivity,
    exact_one_row_sensitivity_bruteforce,
    expected_total_variation_upper_bound,
)


class SpecificityCompatibilityTests(unittest.TestCase):
    def test_v2_right_matches_current_right_ecdf_rule(self) -> None:
        reference = {"a": 1, "b": 2, "c": 4}
        target = {"a": 1, "b": 1, "missing": 2}
        result = compute_specificity(reference, target, method="v2_right")

        # a: 1 - 1/3 = 2/3; b: 1 - 2/3 = 1/3; missing: 1.
        expected = ((2 / 3) + (1 / 3) + 2.0) / 4
        self.assertAlmostEqual(result.specificity_score, expected)
        self.assertEqual(result.unseen_target_rows, 2)
        self.assertAlmostEqual(result.unseen_bin_rate, 0.5)

    def test_midrank_handles_ties_at_their_middle_rank(self) -> None:
        reference = {"a": 10, "b": 10, "c": 10, "d": 10}
        target = {"a": 1}
        result = compute_specificity(reference, target, method="midrank")
        self.assertAlmostEqual(result.specificity_score, 0.5)


class ExactOneRowCertificateTests(unittest.TestCase):
    def test_known_right_ecdf_rank_jump_does_not_vanish_with_large_counts(self) -> None:
        target = {"a": 1}
        small = exact_one_row_sensitivity(
            {"a": 10, "b": 10, "c": 10, "d": 10},
            target,
            valid_bin_count=4,
            method="v2_right",
        )
        large = exact_one_row_sensitivity(
            {"a": 10_000, "b": 10_000, "c": 10_000, "d": 10_000},
            target,
            valid_bin_count=4,
            method="v2_right",
        )
        self.assertAlmostEqual(small.max_change, 0.75)
        self.assertAlmostEqual(large.max_change, 0.75)

    def test_known_midrank_jump_is_smaller_but_still_nonvanishing(self) -> None:
        target = {"a": 1}
        small = exact_one_row_sensitivity(
            {"a": 10, "b": 10, "c": 10, "d": 10},
            target,
            valid_bin_count=4,
            method="midrank",
        )
        large = exact_one_row_sensitivity(
            {"a": 10_000, "b": 10_000, "c": 10_000, "d": 10_000},
            target,
            valid_bin_count=4,
            method="midrank",
        )
        self.assertAlmostEqual(small.max_change, 0.375)
        self.assertAlmostEqual(large.max_change, 0.375)

    def test_optimized_search_matches_bruteforce_on_random_small_grids(self) -> None:
        rng = random.Random(20260810)
        for _ in range(250):
            valid_bins = rng.randint(1, 8)
            keys = [f"g{i}" for i in range(valid_bins)]
            reference = {key: rng.randint(0, 5) for key in keys}
            target = {key: rng.randint(0, 3) for key in keys}
            if not any(reference.values()):
                reference[keys[0]] = 1
            if not any(target.values()):
                target[keys[-1]] = 1

            for method in ("v2_right", "midrank"):
                brute = exact_one_row_sensitivity_bruteforce(
                    reference,
                    target,
                    valid_bin_count=valid_bins,
                    method=method,
                )
                optimized = exact_one_row_sensitivity(
                    reference,
                    target,
                    valid_bin_count=valid_bins,
                    method=method,
                )
                self.assertAlmostEqual(optimized.baseline_score, brute.baseline_score)
                self.assertAlmostEqual(optimized.max_change, brute.max_change)
                self.assertAlmostEqual(optimized.certified_min_score, brute.certified_min_score)
                self.assertAlmostEqual(optimized.certified_max_score, brute.certified_max_score)

    def test_large_valid_grid_is_not_materialized(self) -> None:
        result = exact_one_row_sensitivity(
            {"a": 2, "b": 3},
            {"a": 1},
            valid_bin_count=10**12,
            method="v2_right",
        )
        self.assertGreaterEqual(result.max_change, 0.0)
        self.assertLessEqual(result.max_change, 1.0)
        self.assertLess(result.evaluated_neighbor_count, 100)

    def test_single_valid_bin_has_no_nontrivial_replacement(self) -> None:
        result = exact_one_row_sensitivity(
            {"only": 20},
            {"only": 1},
            valid_bin_count=1,
            method="midrank",
        )
        self.assertEqual(result.evaluated_neighbor_count, 0)
        self.assertEqual(result.max_change, 0.0)
        self.assertEqual(result.stability_score, 1.0)


class CoverageAndEvennessTests(unittest.TestCase):
    def test_raw_coverage(self) -> None:
        self.assertAlmostEqual(
            compute_raw_coverage({"a": 10, "b": 1, "zero": 0}, valid_bin_count=8),
            0.25,
        )

    def test_evenness_is_one_for_uniform_distribution(self) -> None:
        result = compute_evenness_candidates({"a": 5, "b": 5, "c": 5, "d": 5})
        self.assertAlmostEqual(result.pielou_evenness or 0.0, 1.0)
        self.assertAlmostEqual(result.simpson_effective_bins or 0.0, 4.0)
        self.assertAlmostEqual(result.simpson_effective_evenness or 0.0, 1.0)

    def test_evenness_one_bin_convention_separates_evenness_from_coverage(self) -> None:
        result = compute_evenness_candidates({"a": 100})
        self.assertEqual(result.pielou_evenness, 1.0)
        self.assertEqual(result.simpson_effective_evenness, 1.0)

    def test_simpson_evenness_detects_concentration(self) -> None:
        result = compute_evenness_candidates({"a": 97, "b": 1, "c": 1, "d": 1})
        self.assertIsNotNone(result.simpson_effective_evenness)
        self.assertLess(result.simpson_effective_evenness or 1.0, 0.3)


class TheoreticalCandidateTests(unittest.TestCase):
    def test_tv_bound_uses_only_n_and_valid_grid_size(self) -> None:
        bound = expected_total_variation_upper_bound(400, 4)
        self.assertAlmostEqual(bound, 0.5 * ((3 / 400) ** 0.5))

    def test_tv_bound_is_zero_for_one_valid_bin(self) -> None:
        self.assertEqual(expected_total_variation_upper_bound(10, 1), 0.0)


if __name__ == "__main__":
    unittest.main()
