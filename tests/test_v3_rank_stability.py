from __future__ import annotations

import math
import unittest

from v3_rank_stability import (
    bin_rank_stability,
    probability_right_ecdf_indicator_retained,
    target_rank_stability,
)


def brute_pair_retention(N: int, c_i: int, c_j: int) -> float:
    """Exact small-N multinomial enumeration for test cross-checking."""
    p_i = c_i / N
    p_j = c_j / N
    p_r = 1.0 - p_i - p_j
    current = int(c_j <= c_i)
    total = 0.0

    for x_i in range(N + 1):
        for x_j in range(N - x_i + 1):
            x_r = N - x_i - x_j
            coeff = math.factorial(N) / (
                math.factorial(x_i) * math.factorial(x_j) * math.factorial(x_r)
            )
            prob = coeff * (p_i ** x_i) * (p_j ** x_j) * (p_r ** x_r)
            new_indicator = int(x_j <= x_i)
            if new_indicator == current:
                total += prob
    return total


class PairwiseExactnessTests(unittest.TestCase):
    def test_exact_formula_matches_bruteforce_small_cases(self):
        for N, c_i, c_j in (
            (4, 1, 1),
            (5, 1, 2),
            (6, 2, 1),
            (8, 2, 3),
            (10, 1, 4),
        ):
            expected = brute_pair_retention(N, c_i, c_j)
            actual = probability_right_ecdf_indicator_retained(
                N=N, c_i=c_i, c_j=c_j
            )
            self.assertAlmostEqual(actual, expected, places=12)


class RankStabilityBehaviorTests(unittest.TestCase):
    def test_same_shape_more_rows_increases_rank_stability(self):
        vals = []
        for m in (1, 2, 5, 10, 50):
            ref = {"a": m, "b": 2*m, "c": 3*m, "d": 4*m}
            s = bin_rank_stability(ref, target_bin="a")
            self.assertIsNotNone(s)
            vals.append(s.mean_nontrivial_comparisons)
        self.assertTrue(all(vals[i] < vals[i+1] for i in range(len(vals)-1)))

    def test_equal_counts_do_not_converge_to_false_certainty(self):
        small = bin_rank_stability(
            {"a": 10, "b": 10, "c": 10, "d": 10}, target_bin="a"
        )
        large = bin_rank_stability(
            {"a": 1000, "b": 1000, "c": 1000, "d": 1000}, target_bin="a"
        )
        self.assertIsNotNone(small)
        self.assertIsNotNone(large)
        self.assertGreater(small.mean_nontrivial_comparisons, 0.5)
        self.assertLess(large.mean_nontrivial_comparisons, 0.52)

    def test_same_reference_different_target_can_have_different_stability(self):
        ref = {"a": 100, "b": 101, "c": 300, "d": 499}
        a = bin_rank_stability(ref, target_bin="a")
        c = bin_rank_stability(ref, target_bin="c")
        self.assertIsNotNone(a)
        self.assertIsNotNone(c)
        self.assertGreater(
            c.mean_nontrivial_comparisons - a.mean_nontrivial_comparisons,
            0.1,
        )

    def test_unseen_target_is_not_assigned_false_stability(self):
        result = target_rank_stability(
            {"a": 10, "b": 10},
            {"missing": 3},
        )
        self.assertIsNone(result.weighted_mean_nontrivial)
        self.assertEqual(result.target_rows_evaluated, 0)
        self.assertEqual(result.unseen_target_rows, 3)


if __name__ == "__main__":
    unittest.main()
