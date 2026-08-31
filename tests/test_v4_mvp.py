from __future__ import annotations

import unittest

from v4_mvp.modules import analyze_question


def cluster(name: str, rows: list[tuple[int, str, float]], *, protocol: str = "prime-v1", context: str = "same-machine"):
    csv_text = "N,algorithm,runtime_ms\n" + "\n".join(f"{n},{algorithm},{runtime}" for n, algorithm, runtime in rows)
    return {"id": name, "name": name, "csvText": csv_text, "protocol": protocol, "context": context}


class PrimeBoundaryModuleTests(unittest.TestCase):
    def test_brackets_boundary_and_proposes_midpoint(self):
        data = [
            cluster("low", [(100, "trial", 1.0), (100, "sieve", 2.0)]),
            cluster("high", [(300, "trial", 4.0), (300, "sieve", 2.0)]),
        ]
        result = analyze_question("prime_crossover", data)
        self.assertEqual(result["status"], "ok")
        self.assertIn("100 < N* ≤ 300", result["summary"])
        self.assertEqual(result["proposal"]["input"]["N"], 200)

    def test_rejects_multiple_crossovers(self):
        data = [cluster("multi", [(100, "trial", 1.0), (100, "sieve", 2.0), (200, "trial", 3.0), (200, "sieve", 2.0), (300, "trial", 1.0), (300, "sieve", 2.0)])]
        result = analyze_question("prime_crossover", data)
        self.assertEqual(result["status"], "assumption_failed")
        self.assertIsNone(result["proposal"])

    def test_rejects_protocol_mismatch(self):
        data = [
            cluster("one", [(100, "trial", 1.0), (100, "sieve", 2.0)], protocol="v1"),
            cluster("two", [(300, "trial", 4.0), (300, "sieve", 2.0)], protocol="v2"),
        ]
        result = analyze_question("prime_crossover", data)
        self.assertEqual(result["status"], "protocol_mismatch")

    def test_general_question_refuses_undefined_scope(self):
        data = [cluster("one", [(100, "trial", 1.0), (100, "sieve", 2.0)])]
        result = analyze_question("general_prime_speed", data)
        self.assertEqual(result["status"], "insufficient_scope")
        self.assertIsNone(result["proposal"])


if __name__ == "__main__":
    unittest.main()
