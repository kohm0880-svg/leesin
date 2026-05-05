from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import app
import storage
from models import ExperimentConfig
from stats_engine import DataQualityAnalyzer


def axis(name: str) -> dict:
    return {"name": name, "unit": "", "domainMin": 0.0, "domainMax": 10.0, "resolution": 1.0}


class CoreBehaviorTests(unittest.TestCase):
    def test_csv_rows_become_one_cluster_vector(self) -> None:
        goal = {"id": "goal_a", "name": "A", "K_m": 10.0, "axes": [axis("x"), axis("y")]}
        rows = [{"x_col": "1", "y_col": "10"}, {"x_col": "3", "y_col": "14"}]
        vector, meta = app.build_cluster_vector(rows, {"x": "x_col", "y": "y_col"}, goal)
        self.assertEqual(meta["row_count"], 2)
        np.testing.assert_allclose(vector, np.array([2.0, 12.0]))
        self.assertEqual(meta["values_mean"], [2.0, 12.0])
        self.assertEqual(meta["values_variance"], [2.0, 8.0])
        self.assertEqual(meta["values_std"], [1.414213562373, 2.828427124746])
        self.assertIn("bin_occupancy", meta)
        self.assertEqual(meta["bin_occupancy_meta"]["totalRows"], 2)

    def test_same_axes_different_goal_id_are_separated(self) -> None:
        clusters = [
            {"id": "a1", "goalId": "goal_a", "axisNames": ["x", "y"], "values": [1, 2], "rowCount": 2},
            {"id": "b1", "goalId": "goal_b", "axisNames": ["x", "y"], "values": [9, 8], "rowCount": 2},
        ]
        goal = {"id": "goal_a", "name": "A", "K_m": 10.0, "axes": [axis("x"), axis("y")]}
        with patch("storage.load_cluster_store", return_value=clusters):
            peers = storage.get_peer_group(goal, ["x", "y"])
        np.testing.assert_allclose(peers, np.array([[1.0, 2.0]]))

    def test_same_goal_and_axis_signature_match_peer_group(self) -> None:
        clusters = [{"id": "a1", "goalId": "goal_a", "axisNames": ["x", "y"], "values": [1, 2], "rowCount": 2}]
        goal = {"id": "goal_a", "name": "A", "K_m": 10.0, "axes": [axis("x"), axis("y")]}
        with patch("storage.load_cluster_store", return_value=clusters):
            peers = storage.get_peer_group(goal, ["x", "y"])
        self.assertEqual(peers.shape, (1, 2))

    def test_axis_matching_is_order_case_and_space_insensitive(self) -> None:
        clusters = [
            {
                "id": "a1",
                "goalId": "goal_a",
                "axisNames": [" Y ", "X"],
                "values": [20, 10],
                "rowCount": 2,
            }
        ]
        goal = {"id": "goal_a", "name": "A", "K_m": 10.0, "axes": [axis("x"), axis("y")]}
        with patch("storage.load_cluster_store", return_value=clusters):
            peers = storage.get_peer_group(goal, [" x ", "y"])
            diagnostics = storage.explain_peer_filter("goal_a", ["y", "x"])
        np.testing.assert_allclose(peers, np.array([[10.0, 20.0]]))
        self.assertEqual(diagnostics["compatibleAxisCount"], 1)

    def test_peer_group_uses_only_saved_clusters(self) -> None:
        goal = {"id": "goal_a", "name": "A", "K_m": 10.0, "axes": [axis("x"), axis("y")]}
        with patch("storage.load_cluster_store", return_value=[]):
            with self.assertRaises(ValueError):
                storage.get_peer_group(goal, ["x", "y"])

    def test_selected_axis_subset_uses_same_goal_clusters(self) -> None:
        clusters = [{"id": "a1", "goalId": "goal_a", "axisNames": ["a", "b", "c"], "values": [1, 2, 3], "rowCount": 2}]
        goal = {"id": "goal_a", "name": "A", "K_m": 10.0, "axes": [axis("a"), axis("b"), axis("c")]}
        with patch("storage.load_cluster_store", return_value=clusters):
            peers = storage.get_peer_group(goal, ["a", "c"])
        np.testing.assert_allclose(peers, np.array([[1.0, 3.0]]))

    def test_coverage_uses_row_level_bins_while_sample_size_uses_cluster_count(self) -> None:
        config = ExperimentConfig(["x", "y"], [(0, 10), (0, 10)], [1, 1])
        analyzer = DataQualityAnalyzer(config)
        peers = np.array([[1, 1], [2, 2], [2.1, 2.1], [3, 3]], dtype=float)
        analyzer.add_peers(peers)
        analyzer.add_coverage_bin_counts(
            {
                "[0,0]": 10,
                "[1,1]": 12,
                "[2,2]": 8,
                "[3,3]": 6,
                "[4,4]": 5,
                "[5,5]": 4,
            }
        )
        result = analyzer.diagnose([4, 4])
        self.assertEqual(result.occupied_bins, 6)
        self.assertGreater(result.occupied_bins, len(peers))
        self.assertAlmostEqual(result.sample_size_Z, len(peers) / (len(peers) + config.K_m))

    def test_row_level_bin_occupancy_counts_valid_invalid_and_out_of_domain_rows(self) -> None:
        goal = {
            "id": "goal_a",
            "name": "A",
            "K_m": 10.0,
            "axes": [
                {**axis("x"), "resolution": 5.0},
                {**axis("y"), "resolution": 5.0},
            ],
        }
        rows = [
            {"x_col": "0", "y_col": "0"},
            {"x_col": "4.9", "y_col": "10"},
            {"x_col": "10", "y_col": "10"},
            {"x_col": "bad", "y_col": "2"},
            {"x_col": "11", "y_col": "2"},
        ]
        _vector, meta = app.build_cluster_vector(rows, {"x": "x_col", "y": "y_col"}, goal)
        occupancy = meta["bin_occupancy"]
        occupancy_meta = meta["bin_occupancy_meta"]
        self.assertEqual(sum(occupancy.values()), occupancy_meta["validMultidimensionalRowCount"])
        self.assertEqual(occupancy_meta["validMultidimensionalRowCount"], 3)
        self.assertEqual(occupancy_meta["invalidRowCount"], 1)
        self.assertEqual(occupancy_meta["outOfDomainRowCount"], 1)
        self.assertEqual(occupancy["[0,0]"], 1)
        self.assertEqual(occupancy["[0,1]"], 1)
        self.assertEqual(occupancy["[1,1]"], 1)

    def test_legacy_clusters_remain_heterogeneity_peers_but_not_coverage_peers(self) -> None:
        clusters = [
            {"id": "legacy", "storedAxisSignature": storage.axis_subset_key(["x", "y"]), "axisNames": ["x", "y"], "values": [1, 1], "rowCount": 3},
            {
                "id": "eligible",
                "storedAxisSignature": storage.axis_subset_key(["x", "y"]),
                "axisNames": ["x", "y"],
                "values": [2, 2],
                "rowCount": 3,
                "binOccupancy": {"[0,0]": 2, "[1,1]": 1},
                "axisBinOccupancy": {"x": {"0": 2, "1": 1}, "y": {"0": 2, "1": 1}},
                "binOccupancyMeta": {"validMultidimensionalRowCount": 3},
            },
            {
                "id": "axis_mismatch",
                "storedAxisSignature": storage.axis_subset_key(["x", "y", "z"]),
                "axisNames": ["x", "y"],
                "values": [3, 3],
                "rowCount": 3,
                "binOccupancy": {"[0,0,0]": 3},
            },
        ]
        coverage = app.build_global_bin_counts(clusters, ["x", "y"])
        self.assertEqual(coverage["coverageEligibleClusterCount"], 1)
        self.assertEqual(coverage["coverageLegacyExcludedClusterCount"], 1)
        self.assertEqual(coverage["coverageAxisSignatureExcludedClusterCount"], 1)
        self.assertEqual(coverage["rowLevelObservationCount"], 3)
        self.assertEqual(coverage["occupiedBins"], 2)

    def test_heterogeneity_increases_with_d2(self) -> None:
        config = ExperimentConfig(["x", "y"], [(-10, 10), (-10, 10)], [1, 1])
        peers = np.array(
            [
                [0, 0],
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1],
                [1, 1],
                [-1, -1],
                [1, -1],
                [-1, 1],
                [0.5, 0.2],
                [-0.4, 0.3],
                [0.2, -0.6],
            ],
            dtype=float,
        )
        analyzer = DataQualityAnalyzer(config)
        analyzer.add_peers(peers)
        near = analyzer.diagnose([0.1, 0.1])
        far = analyzer.diagnose([4, 4])
        self.assertGreater(far.D2, near.D2)
        self.assertGreater(far.heterogeneity, near.heterogeneity)

    def test_non_numeric_values_are_excluded_from_row_level_bins(self) -> None:
        goal = {"id": "goal_a", "name": "A", "K_m": 10.0, "axes": [axis("x")]}
        vector, meta = app.build_cluster_vector([{"x_col": "1"}, {"x_col": "bad"}], {"x": "x_col"}, goal)
        np.testing.assert_allclose(vector, np.array([1.0]))
        self.assertEqual(meta["bin_occupancy_meta"]["validMultidimensionalRowCount"], 1)
        self.assertEqual(meta["bin_occupancy_meta"]["invalidRowCount"], 1)
        with self.assertRaisesRegex(ValueError, "numeric row"):
            app.build_cluster_vector([{"x_col": "bad"}], {"x": "x_col"}, goal)

    def test_batch_analysis_uses_only_preexisting_peer_group(self) -> None:
        goal = {"id": "goal_a", "name": "A", "K_m": 10.0, "axes": [axis("x"), axis("y")]}
        clusters = [
            {"id": "p1", "goalId": "goal_a", "axisNames": ["x", "y"], "values": [1, 1], "rowCount": 1},
            {"id": "p2", "goalId": "goal_a", "axisNames": ["x", "y"], "values": [2, 2], "rowCount": 1},
            {"id": "p3", "goalId": "goal_a", "axisNames": ["x", "y"], "values": [3, 3], "rowCount": 1},
        ]
        payload = {
            "goalId": "goal_a",
            "selectedAxes": ["x", "y"],
            "files": [
                {"displayName": "a.csv", "axisMapping": {"x": "x", "y": "y"}, "rows": [{"x": "4", "y": "4"}]},
                {"displayName": "b.csv", "axisMapping": {"x": "x", "y": "y"}, "rows": [{"x": "5", "y": "5"}]},
            ],
        }
        with patch("app.load_goal_store", return_value=[goal]), patch("storage.load_cluster_store", return_value=clusters), patch("app.load_cluster_store", return_value=clusters):
            result = app.analyze_batch_request(payload)
        self.assertEqual([item["analysisSummary"]["peer_group_size"] for item in result["items"]], [3, 3])

    def test_batch_save_makes_saved_clusters_available_to_next_analysis(self) -> None:
        goal = {"id": "goal_batch", "name": "Batch", "K_m": 10.0, "axes": [axis("a"), axis("b"), axis("c"), axis("d")]}
        backing_store: list[dict] = []

        def save_store(clusters: list[dict]) -> None:
            backing_store[:] = clusters

        records = []
        for index in range(6):
            vector = np.array([index + 1, index + 2, index + 3, index + 4], dtype=float)
            records.append(
                app.make_cluster_record(
                    goal,
                    goal,
                    vector,
                    {
                        "row_count": 1,
                        "summary_method": "mean",
                        "values_mean": [float(value) for value in vector],
                    },
                    source_batch_id="batch_test",
                )
            )

        with (
            patch("app.load_goal_store", return_value=[goal]),
            patch("storage.load_goal_store", return_value=[goal]),
            patch("storage.load_cluster_store", side_effect=lambda: list(backing_store)),
            patch("storage.save_cluster_store", side_effect=save_store),
            patch("app.load_cluster_store", side_effect=lambda: list(backing_store)),
        ):
            response = app.batch_save_request(
                {
                    "goalId": "goal_batch",
                    "selectedAxisNames": [" d ", "C", "b", "A"],
                    "records": records,
                }
            )
            peers = storage.get_peer_group(goal, ["A", "b", "c", "d"])

        self.assertEqual(response["compatiblePeerCountForSelectedAxes"], 6)
        self.assertEqual(peers.shape, (6, 4))

    def test_current_reevaluation_excludes_self_from_peer_group(self) -> None:
        goal = {"id": "goal_a", "name": "A", "K_m": 10.0, "axes": [axis("x"), axis("y")]}
        clusters = [
            {"id": "target", "goalId": "goal_a", "axisNames": ["x", "y"], "values": [4, 4], "rowCount": 1},
            {"id": "p1", "goalId": "goal_a", "axisNames": ["x", "y"], "values": [1, 1], "rowCount": 1},
            {"id": "p2", "goalId": "goal_a", "axisNames": ["x", "y"], "values": [2, 2], "rowCount": 1},
            {"id": "p3", "goalId": "goal_a", "axisNames": ["x", "y"], "values": [3, 3], "rowCount": 1},
        ]
        with patch("app.load_goal_store", return_value=[goal]), patch("storage.load_cluster_store", return_value=clusters), patch("app.load_cluster_store", return_value=clusters):
            result = app.reevaluate_cluster("target")
        self.assertEqual(result["currentPeerGroupSize"], 3)

    def test_out_of_domain_warning_is_reported(self) -> None:
        goal = {"id": "goal_a", "name": "A", "K_m": 10.0, "axes": [axis("x"), axis("y")]}
        clusters = [
            {"id": "p1", "goalId": "goal_a", "axisNames": ["x", "y"], "values": [1, 1], "rowCount": 1},
            {"id": "p2", "goalId": "goal_a", "axisNames": ["x", "y"], "values": [2, 2], "rowCount": 1},
            {"id": "p3", "goalId": "goal_a", "axisNames": ["x", "y"], "values": [3, 3], "rowCount": 1},
        ]
        with patch("storage.load_cluster_store", return_value=clusters):
            result = app.run_vector_analysis(goal, goal, np.array([11.0, 2.0]))
        self.assertEqual(result["resultPayload"]["outOfDomainWarningCount"], 1)
        self.assertEqual(result["resultPayload"]["outOfDomainWarnings"][0]["role"], "target")

    def test_export_json_and_csv_are_download_payloads(self) -> None:
        report = {
            "meta": {"experiment_goal": "A", "goal_id": "goal_a", "axis_names": ["x"], "peer_group_size": 3},
            "result": {"engine": "spatial_rank", "confidence": 0.5, "heterogeneity": 0.1},
            "summary": ["ok"],
        }
        exported_json = app.export_report_request({"format": "json", "report": report})
        exported_csv = app.export_report_request({"format": "csv", "report": report})
        self.assertTrue(exported_json["filename"].endswith(".json"))
        self.assertTrue(exported_csv["filename"].endswith(".csv"))
        self.assertIn("experiment_goal", exported_csv["content"])


if __name__ == "__main__":
    unittest.main()
