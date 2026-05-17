from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import app
import storage
from models import ExperimentConfig
from stats_engine import DensityGridAnalyzer


def axis(name: str, domain_max: float = 10.0, resolution: float = 1.0) -> dict:
    return {"name": name, "unit": "", "domainMin": 0.0, "domainMax": domain_max, "resolution": resolution}


def goal(goal_id: str = "goal_a", axes: list[dict] | None = None) -> dict:
    return {"id": goal_id, "name": "A", "K_m": 10.0, "axes": axes or [axis("x"), axis("y")]}


def record_from_rows(goal_payload: dict, rows: list[dict[str, str]], record_id: str) -> dict:
    mapping = {item["name"]: item["name"] for item in goal_payload["axes"]}
    vector, meta = app.build_dataset_summary(rows, mapping, goal_payload)
    record = app.make_cluster_record(goal_payload, goal_payload, vector, meta)
    record["id"] = record_id
    return record


class DensityGridBehaviorTests(unittest.TestCase):
    def test_csv_rows_become_cluster_vector_and_bin_occupancy(self) -> None:
        goal_payload = goal(axes=[axis("x"), axis("y", domain_max=20.0)])
        rows = [{"x": "1", "y": "10"}, {"x": "3", "y": "14"}]
        vector, meta = app.build_dataset_summary(rows, {"x": "x", "y": "y"}, goal_payload)
        np.testing.assert_allclose(vector, np.array([2.0, 12.0]))
        self.assertEqual(meta["bin_occupancy_meta"]["validMultidimensionalRowCount"], 2)
        self.assertEqual(sum(meta["bin_occupancy"].values()), 2)

    def test_unseen_peer_bin_increases_unseen_bin_rate(self) -> None:
        analyzer = DensityGridAnalyzer(ExperimentConfig(["x"], [(0, 10)], [1]))
        analyzer.set_peer_bin_counts({"[0]": 10})
        result = analyzer.diagnose({"[5]": 3}, {"validMultidimensionalRowCount": 3, "totalRows": 3})
        self.assertEqual(result.unseen_bin_rate, 1.0)
        self.assertGreater(result.specificity_score, 0.0)

    def test_low_peer_bin_count_increases_mean_rarity(self) -> None:
        analyzer = DensityGridAnalyzer(ExperimentConfig(["x"], [(0, 10)], [1]))
        analyzer.set_peer_bin_counts({"[0]": 100, "[5]": 1})
        common = analyzer.diagnose({"[0]": 4}, {"validMultidimensionalRowCount": 4, "totalRows": 4})
        rare = analyzer.diagnose({"[5]": 4}, {"validMultidimensionalRowCount": 4, "totalRows": 4})
        self.assertGreater(rare.mean_rarity, common.mean_rarity)
        self.assertGreater(rare.specificity_score, common.specificity_score)

    def test_rare_peer_bin_increases_rare_bin_rate(self) -> None:
        analyzer = DensityGridAnalyzer(ExperimentConfig(["x"], [(0, 10)], [1]))
        analyzer.set_peer_bin_counts({"[0]": 10, "[5]": 1})
        common = analyzer.diagnose({"[0]": 3}, {"validMultidimensionalRowCount": 3, "totalRows": 3})
        rare = analyzer.diagnose({"[5]": 3}, {"validMultidimensionalRowCount": 3, "totalRows": 3})
        self.assertEqual(common.rare_bin_rate, 0.0)
        self.assertEqual(rare.rare_bin_rate, 1.0)

    def test_empty_peer_density_raises_clear_error(self) -> None:
        analyzer = DensityGridAnalyzer(ExperimentConfig(["x"], [(0, 10)], [1]))
        with self.assertRaisesRegex(ValueError, "at least one peer row-level bin occupancy observation"):
            analyzer.diagnose({"[0]": 1}, {"validMultidimensionalRowCount": 1, "totalRows": 1})

    def test_same_aggregated_bin_count_gives_same_density_result(self) -> None:
        selected_goal = goal(axes=[axis("x")])
        one_cluster = record_from_rows(selected_goal, [{"x": "1"}, {"x": "1"}, {"x": "2"}, {"x": "8"}], "one")
        split_a = record_from_rows(selected_goal, [{"x": "1"}, {"x": "1"}], "split_a")
        split_b = record_from_rows(selected_goal, [{"x": "2"}, {"x": "8"}], "split_b")

        target_counts = {"[1]": 2, "[2]": 1}
        target_meta = {"validMultidimensionalRowCount": 3, "totalRows": 3}
        with patch("storage.load_cluster_store", return_value=[one_cluster]):
            one = app.run_density_analysis(selected_goal, selected_goal, target_counts, target_meta)
        with patch("storage.load_cluster_store", return_value=[split_a, split_b]):
            split = app.run_density_analysis(selected_goal, selected_goal, target_counts, target_meta)

        self.assertEqual(one["coverageInfo"]["binCounts"], split["coverageInfo"]["binCounts"])
        self.assertEqual(one["resultPayload"]["specificity_score"], split["resultPayload"]["specificity_score"])
        self.assertEqual(one["resultPayload"]["mean_rarity"], split["resultPayload"]["mean_rarity"])

    def test_density_analyzer_result_same_for_same_bin_counts(self) -> None:
        config = ExperimentConfig(["x"], [(0, 10)], [1])
        one = DensityGridAnalyzer(config)
        one.set_peer_bin_counts({"[1]": 2, "[2]": 1, "[8]": 1})
        split = DensityGridAnalyzer(config)
        split.add_peer_bin_counts({"[1]": 2})
        split.add_peer_bin_counts({"[2]": 1, "[8]": 1})
        target = {"[1]": 2, "[2]": 1}
        meta = {"validMultidimensionalRowCount": 3, "totalRows": 3}
        one_result = one.diagnose(target, meta)
        split_result = split.diagnose(target, meta)
        self.assertEqual(one_result.to_payload(["x"]), split_result.to_payload(["x"]))

    def test_out_of_domain_rows_are_reflected_in_rate(self) -> None:
        analyzer = DensityGridAnalyzer(ExperimentConfig(["x"], [(0, 10)], [1]))
        analyzer.set_peer_bin_counts({"[0]": 4})
        result = analyzer.diagnose(
            {"[0]": 1},
            {"validMultidimensionalRowCount": 1, "outOfDomainRowCount": 1, "totalRows": 2},
        )
        self.assertEqual(result.out_of_domain_rate, 0.5)

    def test_grid_signature_mismatch_recomputes_when_row_vectors_exist(self) -> None:
        selected_goal = goal(axes=[axis("x")])
        matching = record_from_rows(selected_goal, [{"x": "1"}, {"x": "2"}], "matching")
        mismatched = record_from_rows(goal(axes=[axis("x", resolution=0.5)]), [{"x": "1"}], "mismatched")
        coverage = app.build_global_bin_counts([matching, mismatched], selected_goal)
        self.assertEqual(coverage["coverageEligibleClusterCount"], 2)
        self.assertEqual(coverage["coverageGridSignatureExcludedClusterCount"], 0)
        self.assertEqual(coverage["rowLevelObservationCount"], 3)

    def test_selected_axis_order_uses_canonical_grid_signature_and_bin_key(self) -> None:
        unsorted_goal = goal(axes=[axis("y"), axis("x")])
        selected_xy = app.goal_subset(unsorted_goal, ["x", "y"])
        selected_yx = app.goal_subset(unsorted_goal, ["y", "x"])
        self.assertEqual([axis["name"] for axis in selected_xy["axes"]], ["x", "y"])
        self.assertEqual([axis["name"] for axis in selected_yx["axes"]], ["x", "y"])
        self.assertEqual(
            storage.grid_signature_from_axes(selected_xy["axes"]),
            storage.grid_signature_from_axes(selected_yx["axes"]),
        )
        _vector, meta = app.build_dataset_summary([{"x": "2", "y": "7"}], {"x": "x", "y": "y"}, selected_yx)
        self.assertEqual(meta["bin_occupancy"], {"[2,7]": 1})

    def test_legacy_cluster_without_bin_occupancy_is_excluded_from_density(self) -> None:
        selected_goal = goal(axes=[axis("x")])
        eligible = record_from_rows(selected_goal, [{"x": "1"}], "eligible")
        legacy = {
            "id": "legacy",
            "goalId": selected_goal["id"],
            "axisNames": ["x"],
            "values": [1.0],
            "rowCount": 1,
            "gridSignature": storage.grid_signature_from_axes(selected_goal["axes"]),
        }
        coverage = app.build_global_bin_counts([eligible, legacy], selected_goal)
        self.assertEqual(coverage["coverageEligibleClusterCount"], 1)
        self.assertEqual(coverage["coverageLegacyExcludedClusterCount"], 1)

    def test_density_analysis_does_not_require_vector_peer_minimum(self) -> None:
        selected_goal = goal(axes=[axis("x"), axis("y")])
        peer = record_from_rows(selected_goal, [{"x": "1", "y": "1"}], "peer")
        payload = {
            "goalId": selected_goal["id"],
            "selectedAxes": ["x", "y"],
            "axisMapping": {"x": "x", "y": "y"},
            "rows": [{"x": "1", "y": "1"}, {"x": "5", "y": "5"}],
        }
        with (
            patch("app.load_goal_store", return_value=[selected_goal]),
            patch("storage.load_cluster_store", return_value=[peer]),
            patch("app.load_cluster_store", return_value=[peer]),
            patch("storage.save_cluster_store"),
        ):
            response = app.analyze_request_v2(payload)
        self.assertEqual(response["result"]["engine"], "density_grid")
        self.assertEqual(response["result"]["peer_observation_count"], 1)
        for key in [
            "specificity_score",
            "mean_rarity",
            "unseen_bin_rate",
            "rare_bin_rate",
            "observation_support_S",
            "coverage_C",
            "equitability_E",
            "confidence",
        ]:
            self.assertIn(key, response["result"])

    def test_same_mean_and_row_count_with_different_bin_occupancy_is_not_duplicate(self) -> None:
        selected_goal = goal(axes=[axis("x")])
        record_a = record_from_rows(selected_goal, [{"x": str(value)} for value in [2, 4, 6, 8]], "a")
        record_b = record_from_rows(selected_goal, [{"x": str(value)} for value in [1, 4, 7, 8]], "b")
        self.assertNotEqual(record_a["binOccupancy"], record_b["binOccupancy"])
        self.assertNotEqual(storage.cluster_fingerprint(record_a), storage.cluster_fingerprint(record_b))

    def test_projection_explorer_generates_all_axis_pairs_for_four_axes(self) -> None:
        selected_goal = goal(axes=[axis("d"), axis("b"), axis("a"), axis("c")])
        explorer = app.build_projection_explorer(selected_goal, {"binCounts": {}}, [])
        self.assertEqual(explorer["axisOrder"], ["a", "b", "c", "d"])
        self.assertEqual(len(explorer["axisPairs"]), 6)
        self.assertIn(["a", "b"], explorer["axisPairs"])
        self.assertIn(["c", "d"], explorer["axisPairs"])

    def test_peer_bin_occupancy_projects_to_2d_matrix(self) -> None:
        axes = [axis("a", domain_max=5), axis("b", domain_max=5), axis("c", domain_max=5)]
        axis_order = ["a", "b", "c"]
        axis_meta = app.projection_axis_meta(axes)
        projection = app.build_pair_projection_from_bin_counts(
            {"[1,2,3]": 5, "[1,4,0]": 2, "[9,0,0]": 99, "bad": 4},
            axis_order,
            axis_meta,
            "a",
            "c",
        )
        self.assertEqual(projection["counts"][3][1], 5)
        self.assertEqual(projection["counts"][0][1], 2)
        self.assertEqual(projection["maxCount"], 5)

    def test_target_row_tuples_project_to_2d_matrix(self) -> None:
        axes = [axis("a", domain_max=5), axis("b", domain_max=5), axis("c", domain_max=5)]
        axis_order = ["a", "b", "c"]
        axis_meta = app.projection_axis_meta(axes)
        projection = app.build_pair_projection_from_row_tuples(
            [[1, 2, 3], [1, 2, 3], [0, 2, 1], [8, 0, 0]],
            axis_order,
            axis_meta,
            "a",
            "c",
        )
        self.assertEqual(projection["counts"][3][1], 2)
        self.assertEqual(projection["counts"][1][0], 1)
        self.assertEqual(projection["maxCount"], 2)

    def test_projection_selection_filters_matching_target_tuples(self) -> None:
        selected = app.filter_row_tuples_for_axis_pair(
            [[1, 2, 3], [1, 4, 3], [1, 2, 0], [0, 2, 3]],
            ["a", "b", "c"],
            "a",
            "c",
            1,
            3,
        )
        self.assertEqual(selected, [[1, 2, 3], [1, 4, 3]])

    def test_crosshair_markers_only_target_pairs_with_selected_axes(self) -> None:
        markers = app.crosshair_markers_for_selection(
            [["a", "b"], ["a", "c"], ["b", "c"], ["c", "d"]],
            {"binsByAxis": {"a": 1, "b": 2}},
        )
        self.assertEqual(markers["a|b"], {"xBin": 1, "yBin": 2})
        self.assertEqual(markers["a|c"], {"xBin": 1})
        self.assertEqual(markers["b|c"], {"xBin": 2})
        self.assertNotIn("c|d", markers)

    def test_target_row_bin_tuples_are_visualization_only_not_stored(self) -> None:
        selected_goal = goal(axes=[axis("x"), axis("y")])
        vector, meta = app.build_dataset_summary([{"x": "2", "y": "3"}], {"x": "x", "y": "y"}, selected_goal)
        self.assertEqual(meta["row_bin_tuples"], [[2, 3]])
        record = app.make_cluster_record(selected_goal, selected_goal, vector, meta)
        self.assertNotIn("row_bin_tuples", record)
        self.assertNotIn("rowBinTuples", record)

    def test_row_level_vectors_are_stored_and_recompute_original_bins(self) -> None:
        selected_goal = goal(axes=[axis("x"), axis("y")])
        rows = [{"x": "2", "y": "3"}, {"x": "4", "y": "5"}]
        vector, meta = app.build_dataset_summary(rows, {"x": "x", "y": "y"}, selected_goal)
        record = app.make_cluster_record(selected_goal, selected_goal, vector, meta)
        self.assertEqual(record["rowLevelVectors"], [[2.0, 3.0], [4.0, 5.0]])
        recomputed = app.recompute_bin_occupancy_from_row_vectors(
            record["rowLevelVectors"],
            record["rowLevelVectorAxisOrder"],
            selected_goal,
        )
        self.assertEqual(recomputed["bin_occupancy"], record["binOccupancy"])
        self.assertEqual(recomputed["bin_occupancy_meta"]["validMultidimensionalRowCount"], 2)

    def test_widening_domain_brings_out_of_domain_vector_into_valid_bin(self) -> None:
        narrow = goal(axes=[axis("x", domain_max=10)])
        wide = goal(axes=[axis("x", domain_max=20)])
        vectors = [[12.0]]
        narrow_bins = app.recompute_bin_occupancy_from_row_vectors(vectors, ["x"], narrow)
        wide_bins = app.recompute_bin_occupancy_from_row_vectors(vectors, ["x"], wide)
        self.assertEqual(narrow_bins["bin_occupancy_meta"]["outOfDomainRowCount"], 1)
        self.assertEqual(narrow_bins["bin_occupancy_meta"]["validMultidimensionalRowCount"], 0)
        self.assertEqual(wide_bins["bin_occupancy"], {"[12]": 1})
        self.assertEqual(wide_bins["bin_occupancy_meta"]["validMultidimensionalRowCount"], 1)

    def test_resolution_change_changes_grid_signature_and_bin_occupancy(self) -> None:
        coarse = goal(axes=[axis("x", resolution=1)])
        fine = goal(axes=[axis("x", resolution=0.5)])
        vectors = [[1.2], [1.7]]
        coarse_bins = app.recompute_bin_occupancy_from_row_vectors(vectors, ["x"], coarse)
        fine_bins = app.recompute_bin_occupancy_from_row_vectors(vectors, ["x"], fine)
        self.assertNotEqual(storage.grid_signature_from_axes(coarse["axes"]), storage.grid_signature_from_axes(fine["axes"]))
        self.assertNotEqual(coarse_bins["bin_occupancy"], fine_bins["bin_occupancy"])

    def test_preview_metrics_recompute_from_row_level_vectors(self) -> None:
        selected_goal = goal(axes=[axis("x", domain_max=10)])
        peer = record_from_rows(selected_goal, [{"x": "1"}, {"x": "2"}, {"x": "8"}], "peer")
        coverage = app.build_global_bin_counts([peer], selected_goal)
        metrics = app.density_preview_metrics_from_coverage(selected_goal, coverage)
        self.assertEqual(metrics["peerValidRows"], 3)
        self.assertEqual(metrics["occupiedBins"], 3)
        self.assertGreater(metrics["observationSupportZ"], 0)
        self.assertIn("confidence", metrics)

    def test_apply_goal_grid_defaults_updates_record_grid_hashes(self) -> None:
        original_goal = goal(axes=[axis("x", domain_max=10, resolution=1)])
        record = record_from_rows(original_goal, [{"x": "1.2"}, {"x": "1.7"}], "record")
        old_grid = record["gridSignature"]
        old_hash = record["binOccupancyHash"]
        saved_goals: list[list[dict]] = []
        saved_clusters: list[list[dict]] = []

        def capture_goals(items: list[dict]) -> None:
            saved_goals.append(items)

        def capture_clusters(items: list[dict]) -> None:
            saved_clusters.append(items)

        with (
            patch("app.load_goal_store", return_value=[original_goal]),
            patch("app.load_cluster_store", return_value=[record]),
            patch("storage.load_cluster_store", return_value=[record]),
            patch("app.save_goal_store", side_effect=capture_goals),
            patch("app.save_cluster_store", side_effect=capture_clusters),
        ):
            response = app.apply_goal_grid_defaults_request(
                {
                    "goalId": original_goal["id"],
                    "selectedAxes": ["x"],
                    "previewAxes": [{"name": "x", "domainMin": 0, "domainMax": 10, "resolution": 0.5}],
                }
            )
        self.assertEqual(response["updatedRecordCount"], 1)
        updated_record = saved_clusters[0][0]
        self.assertNotEqual(updated_record["gridSignature"], old_grid)
        self.assertNotEqual(updated_record["binOccupancyHash"], old_hash)

    def test_report_template_does_not_render_coverage_axis_distribution(self) -> None:
        template = app.TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("${renderCoverageAxisDistributions(report.visualizations)}", template)

    def test_projection_zoom_ui_handlers_are_present(self) -> None:
        template = app.TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertIn("projectionZooms", template)
        self.assertIn("event.ctrlKey", template)
        self.assertIn("addEventListener('wheel'", template)
        self.assertIn("addEventListener('dblclick'", template)


if __name__ == "__main__":
    unittest.main()
