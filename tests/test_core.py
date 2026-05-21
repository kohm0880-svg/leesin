from __future__ import annotations

import json
import re
import unittest
from unittest.mock import patch

import numpy as np

import app
import storage
from feasible_mask import (
    compile_focused_2d_mask_rule,
    compile_gui_feasible_rule,
    compute_valid_bin_mask_for_axes,
    evaluate_feasible_expression_on_arrays,
    filter_bin_counts_by_feasible_domain,
    validate_feasible_expression,
)
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

    def test_render_page_uses_lightweight_bootstrap_without_storage_load(self) -> None:
        with (
            patch("app.load_goal_store", side_effect=AssertionError("GET / must not load goals")),
            patch("app.load_cluster_store", side_effect=AssertionError("GET / must not load clusters")),
        ):
            html = app.render_page(admin_allowed=False)
        match = re.search(r"const bootstrap = (.*?);", html)
        self.assertIsNotNone(match)
        payload = json.loads(match.group(1))
        self.assertTrue(payload["deferBootstrap"])
        self.assertEqual(payload["goals"], [])
        self.assertEqual(payload["clusters"], [])

    def test_cluster_summary_omits_heavy_density_payload_by_default(self) -> None:
        selected_goal = goal(axes=[axis("x")])
        record = record_from_rows(selected_goal, [{"x": "1"}, {"x": "2"}], "record_a")
        record["analysisAtUpload"] = {"specificityScore": 0.2, "feasibleMaskEnabled": True}
        with patch("storage.load_cluster_store", return_value=[record]):
            summary = storage.list_cluster_summaries()[0]
        self.assertNotIn("binOccupancy", summary)
        self.assertNotIn("axisBinOccupancy", summary)
        self.assertNotIn("binOccupancyMeta", summary)
        self.assertNotIn("analysisAtUpload", summary)
        self.assertNotIn("rowLevelVectors", summary)
        self.assertEqual(summary["rowLevelVectorCount"], 2)
        self.assertTrue(summary["feasibleMaskEnabledAtUpload"])

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
        analyzer = DensityGridAnalyzer(ExperimentConfig(["x"], [(0, 30)], [1]))
        analyzer.set_peer_bin_counts({f"[{index}]": index + 1 for index in range(21)})
        common = analyzer.diagnose({"[20]": 3}, {"validMultidimensionalRowCount": 3, "totalRows": 3})
        rare = analyzer.diagnose({"[0]": 3}, {"validMultidimensionalRowCount": 3, "totalRows": 3})
        self.assertEqual(common.rare_bin_rate, 0.0)
        self.assertEqual(rare.rare_bin_rate, 1.0)
        self.assertEqual(rare.extreme_specificity_rate, 1.0)

    def test_ecdf_specificity_unseen_bin_is_exact_one(self) -> None:
        analyzer = DensityGridAnalyzer(ExperimentConfig(["x"], [(0, 20)], [1]))
        analyzer.set_peer_bin_counts({"[0]": 1, "[1]": 1, "[2]": 2, "[3]": 5, "[4]": 10})
        result = analyzer.diagnose({"[9]": 1}, {"validMultidimensionalRowCount": 1, "totalRows": 1})
        self.assertEqual(result.specificity_score, 1.0)
        self.assertEqual(result.max_specificity, 1.0)
        self.assertEqual(result.unseen_bin_rate, 1.0)

    def test_ecdf_specificity_count_one_uses_occupied_rank(self) -> None:
        analyzer = DensityGridAnalyzer(ExperimentConfig(["x"], [(0, 20)], [1]))
        analyzer.set_peer_bin_counts({"[0]": 1, "[1]": 1, "[2]": 2, "[3]": 5, "[4]": 10})
        result = analyzer.diagnose({"[0]": 1}, {"validMultidimensionalRowCount": 1, "totalRows": 1})
        self.assertAlmostEqual(result.specificity_score, 0.6)
        self.assertEqual(result.rare_bin_rate, 0.0)

    def test_ecdf_specificity_dense_bin_is_zero(self) -> None:
        analyzer = DensityGridAnalyzer(ExperimentConfig(["x"], [(0, 20)], [1]))
        analyzer.set_peer_bin_counts({"[0]": 1, "[1]": 1, "[2]": 2, "[3]": 5, "[4]": 10})
        result = analyzer.diagnose({"[4]": 1}, {"validMultidimensionalRowCount": 1, "totalRows": 1})
        self.assertEqual(result.specificity_score, 0.0)
        self.assertEqual(result.max_specificity, 0.0)

    def test_ecdf_specificity_weighted_across_target_bins(self) -> None:
        analyzer = DensityGridAnalyzer(ExperimentConfig(["x"], [(0, 20)], [1]))
        analyzer.set_peer_bin_counts({"[0]": 1, "[1]": 1, "[2]": 2, "[3]": 5, "[4]": 10})
        result = analyzer.diagnose({"[9]": 3, "[0]": 7}, {"validMultidimensionalRowCount": 10, "totalRows": 10})
        self.assertAlmostEqual(result.specificity_score, 0.72)
        self.assertAlmostEqual(result.mean_bin_specificity, 0.72)
        self.assertEqual(result.max_specificity, 1.0)
        self.assertEqual(result.extreme_specificity_rate, 0.3)

    def test_empty_peer_density_raises_clear_error(self) -> None:
        analyzer = DensityGridAnalyzer(ExperimentConfig(["x"], [(0, 10)], [1]))
        with self.assertRaisesRegex(ValueError, "target-included reference row-level bin occupancy observation"):
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
        self.assertEqual(response["result"]["peer_observation_count"], 3)
        self.assertEqual(response["result"]["externalPeerObservationCount"], 1)
        self.assertEqual(response["result"]["referenceObservationCount"], 3)
        self.assertTrue(response["result"]["targetIncludedInReference"])
        for key in [
            "specificity_score",
            "specificity_method",
            "mean_bin_specificity",
            "max_specificity",
            "extreme_specificity_rate",
            "mean_rarity",
            "unseen_bin_rate",
            "rare_bin_rate",
            "observation_support_S",
            "coverage_C",
            "equitability_E",
            "confidence",
            "valid_bins",
            "masked_bins",
            "feasible_mask_enabled",
            "masked_out_target_rows",
            "reference_observation_count",
            "external_peer_observation_count",
            "target_included_in_reference",
        ]:
            self.assertIn(key, response["result"])

    def test_single_target_without_stored_peer_still_analyzes_internal_density(self) -> None:
        selected_goal = goal(axes=[axis("x")])
        payload = {
            "goalId": selected_goal["id"],
            "selectedAxes": ["x"],
            "axisMapping": {"x": "x"},
            "rows": [{"x": "1"}, {"x": "1"}, {"x": "2"}],
        }
        with (
            patch("app.load_goal_store", return_value=[selected_goal]),
            patch("storage.load_cluster_store", return_value=[]),
            patch("app.load_cluster_store", return_value=[]),
            patch("storage.save_cluster_store"),
        ):
            response = app.analyze_request_v2(payload)
        self.assertEqual(response["result"]["engine"], "density_grid")
        self.assertEqual(response["result"]["externalPeerRecordCount"], 0)
        self.assertEqual(response["result"]["externalPeerObservationCount"], 0)
        self.assertEqual(response["result"]["referenceObservationCount"], 3)
        self.assertGreaterEqual(response["result"]["specificity_score"], 0.0)
        self.assertGreaterEqual(response["result"]["confidence"], 0.0)
        self.assertTrue(response["result"]["internalDensityMode"])
        self.assertTrue(response["result"]["selfContainedReferenceWarning"])
        self.assertTrue(any("업로드된 데이터 내부" in message for message in response["summary"]))

    def test_delete_impact_reference_excludes_saved_target_before_readding_it(self) -> None:
        selected_goal = goal(axes=[axis("x")])
        target = record_from_rows(selected_goal, [{"x": "1"}, {"x": "1"}], "target")
        external = record_from_rows(selected_goal, [{"x": "2"}], "external")
        with (
            patch("storage.load_cluster_store", return_value=[target, external]),
            patch("app.load_cluster_store", return_value=[target, external]),
        ):
            response = app.impact_result_payload(selected_goal, selected_goal, target, None)
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["externalPeerRecordCount"], 1)
        self.assertEqual(response["result"]["externalPeerObservationCount"], 1)
        self.assertEqual(response["result"]["referenceObservationCount"], 3)
        self.assertEqual(response["result"]["referenceDensityPolicy"], "target_included")

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

    def test_large_projection_matrix_is_skipped(self) -> None:
        axis_order = ["x", "y"]
        axis_meta = {"x": {"totalBins": 1000}, "y": {"totalBins": 1000}}
        projection = app.build_pair_projection_from_bin_counts({"[1,2]": 5}, axis_order, axis_meta, "x", "y")
        self.assertTrue(projection["projectionSkipped"])
        self.assertEqual(projection["counts"], [])
        self.assertIn("MAX_PROJECTION_CELLS", projection["reason"])

    def test_projection_pairs_are_truncated_for_browser_payload(self) -> None:
        selected_goal = goal(axes=[axis(f"a{i}", domain_max=2, resolution=1) for i in range(10)])
        explorer = app.build_projection_explorer(selected_goal, {"binCounts": {}}, [])
        self.assertEqual(explorer["allAxisPairCount"], 45)
        self.assertTrue(explorer["projectionPairTruncated"])
        self.assertEqual(len(explorer["axisPairs"]), app.MAX_PROJECTION_PAIRS)

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

    def test_admin_token_panel_uses_local_storage_and_password_input(self) -> None:
        template = app.TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertIn("leesinAdminToken", template)
        self.assertIn('id="admin-token-input" type="password"', template)
        self.assertIn('id="admin-token-save"', template)
        self.assertIn('id="admin-token-clear"', template)
        self.assertIn("localStorage.setItem", template)
        self.assertIn("localStorage.removeItem", template)

    def test_admin_headers_attach_saved_token_to_admin_requests(self) -> None:
        template = app.TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertIn("headers['X-Admin-Token'] = token", template)
        self.assertIn("fetchJson('/api/admin/goals'", template)
        self.assertIn("headers:adminHeaders()", template)

    def test_admin_token_403_has_friendly_render_message(self) -> None:
        template = app.TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertIn("관리자 작업이 차단되었습니다", template)
        self.assertIn("Render Environment Variables의 ADMIN_TOKEN", template)
        self.assertIn("Admin Token missing or invalid", template)

    def test_feasible_mask_temperature_expression_reduces_valid_bins(self) -> None:
        axes = [axis("temperature", domain_max=100, resolution=10), axis("pressure", domain_max=10, resolution=1)]
        info = compute_valid_bin_mask_for_axes(axes, ["temperature <= 50"])
        self.assertEqual(info["totalBins"], 100)
        self.assertEqual(info["validBins"], 50)
        self.assertEqual(info["maskedBins"], 50)
        self.assertTrue(info["feasibleMaskEnabled"])

    def test_feasible_expression_evaluator_returns_boolean_array(self) -> None:
        temperature = np.array([0.0, 10.0, 20.0])
        pressure = np.array([0.0, 8.0, 15.0])
        mask = evaluate_feasible_expression_on_arrays(
            "pressure <= 0.05 * temperature ** 2",
            {"temperature": temperature, "pressure": pressure},
            ["temperature", "pressure"],
        )
        np.testing.assert_array_equal(mask, np.array([True, False, True]))

    def test_feasible_expression_rejects_import_call(self) -> None:
        with self.assertRaises(ValueError):
            validate_feasible_expression('__import__("os").system("echo hacked")', ["temperature"])

    def test_feasible_expression_rejects_unknown_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown axis name"):
            validate_feasible_expression("unknown_axis > 3", ["temperature"])

    def test_full_feasible_mask_eval_skips_before_large_meshgrid(self) -> None:
        axes = [axis("x", domain_max=1000, resolution=1), axis("y", domain_max=1000, resolution=1)]
        info = compute_valid_bin_mask_for_axes(axes, ["x >= 0"], max_bins=250000)
        self.assertEqual(info["totalBins"], 1_000_000)
        self.assertIsNone(info["validBins"])
        self.assertIsNone(info["maskedBins"])
        self.assertIsNone(info["validDomainRatio"])
        self.assertTrue(info["feasibleMaskEvaluationSkipped"])

    def test_validate_goal_allows_large_grid_when_expression_is_valid(self) -> None:
        payload = goal(axes=[axis("x", domain_max=1000, resolution=1), axis("y", domain_max=1000, resolution=1)])
        payload["feasibleDomainAdvancedExpressions"] = ["x >= 0"]
        normalized = storage.validate_goal(payload)
        self.assertEqual(normalized["feasibleDomainExpressions"], ["x >= 0"])

    def test_validate_expression_mask_info_reports_large_grid_skip(self) -> None:
        payload = goal(axes=[axis("x", domain_max=1000, resolution=1), axis("y", domain_max=1000, resolution=1)])
        payload["feasibleDomainAdvancedExpressions"] = ["x >= 0"]
        normalized = storage.validate_goal(payload)
        mask_info = app.mask_info_for_display(app.feasible_mask_info_for_goal(normalized))
        self.assertTrue(mask_info["feasibleMaskEvaluationSkipped"])
        self.assertEqual(mask_info["totalBins"], 1_000_000)
        self.assertIsNone(mask_info["validBins"])
        self.assertIsNone(mask_info["maskedBins"])
        self.assertIsNone(mask_info["validDomainRatio"])

    def test_large_grid_analysis_falls_back_to_rectangular_coverage_denominator(self) -> None:
        selected_goal = goal(axes=[axis("x", domain_max=1000, resolution=1), axis("y", domain_max=1000, resolution=1)])
        selected_goal["feasibleDomainAdvancedExpressions"] = ["x <= 500"]
        normalized = storage.validate_goal(selected_goal)
        peer = record_from_rows(normalized, [{"x": "10", "y": "10"}], "peer")
        coverage = app.build_global_bin_counts([peer], normalized)
        self.assertTrue(coverage["feasibleMaskEvaluationSkipped"])
        self.assertEqual(coverage["validBins"], 1_000_000)
        self.assertEqual(coverage["maskedBins"], 0)
        analyzer = DensityGridAnalyzer(app.experiment_config_from_goal(normalized))
        analyzer.set_peer_bin_counts(coverage["binCounts"])
        analyzer.set_feasible_domain(coverage["validBins"], coverage["maskedBins"], coverage["feasibleMaskEnabled"])
        result = analyzer.diagnose({"[10,10]": 1}, {"validMultidimensionalRowCount": 1, "totalRows": 1})
        self.assertAlmostEqual(result.coverage_C, 1 / 1_000_000)

    def test_grid_preview_request_skips_huge_preview_recalculation(self) -> None:
        selected_goal = goal(axes=[axis("x", domain_max=1000, resolution=1), axis("y", domain_max=1000, resolution=1)])
        selected_goal["feasibleDomainAdvancedExpressions"] = ["x <= 500"]
        normalized = storage.validate_goal(selected_goal)
        payload = {
            "goalId": normalized["id"],
            "selectedAxes": ["x", "y"],
            "previewAxes": normalized["axes"],
            "targetRowVectors": [[10, 10]],
            "targetRowVectorAxisOrder": ["x", "y"],
        }
        with patch("app.find_goal", return_value=normalized):
            response = app.grid_preview_request(payload)
        self.assertTrue(response["metrics"]["gridPreviewSkipped"])
        self.assertTrue(response["metrics"]["feasibleMaskEvaluationSkipped"])
        self.assertIsNone(response["metrics"]["validBins"])
        self.assertIsNone(response["result"])

    def test_target_row_inside_domain_but_outside_feasible_mask_is_masked_out(self) -> None:
        selected_goal = goal(
            axes=[axis("temperature", domain_max=100, resolution=10), axis("pressure", domain_max=10, resolution=1)]
        )
        selected_goal["feasibleDomainExpressions"] = ["temperature <= 50"]
        recomputed = app.recompute_bin_occupancy_from_row_vectors(
            [[60.0, 5.0], [40.0, 5.0]],
            ["temperature", "pressure"],
            selected_goal,
        )
        self.assertEqual(recomputed["bin_occupancy_meta"]["validMultidimensionalRowCount"], 1)
        self.assertEqual(recomputed["bin_occupancy_meta"]["outOfDomainRowCount"], 0)
        self.assertEqual(recomputed["bin_occupancy_meta"]["maskedOutRowCount"], 1)
        self.assertEqual(recomputed["bin_occupancy"], {"[5,4]": 1})

    def test_coverage_uses_feasible_valid_bins_denominator(self) -> None:
        selected_goal = goal(
            axes=[axis("temperature", domain_max=100, resolution=10), axis("pressure", domain_max=10, resolution=1)]
        )
        selected_goal["feasibleDomainExpressions"] = ["temperature <= 50"]
        peer = record_from_rows(selected_goal, [{"temperature": "40", "pressure": "5"}], "peer")
        coverage = app.build_global_bin_counts([peer], selected_goal)
        analyzer = DensityGridAnalyzer(app.experiment_config_from_goal(selected_goal))
        analyzer.set_peer_bin_counts(coverage["binCounts"])
        analyzer.set_feasible_domain(
            valid_bins=coverage["validBins"],
            masked_bins=coverage["maskedBins"],
            feasible_mask_enabled=coverage["feasibleMaskEnabled"],
        )
        result = analyzer.diagnose({"[4,5]": 1}, {"validMultidimensionalRowCount": 1, "totalRows": 1})
        self.assertEqual(coverage["totalBins"], 100)
        self.assertEqual(coverage["validBins"], 50)
        self.assertAlmostEqual(result.coverage_C, 1 / 50)
        self.assertTrue(result.feasible_mask_enabled)

    def test_gui_feasible_rule_compiles_to_expression(self) -> None:
        rule = {
            "guiSpec": {
                "type": "conditional_range",
                "if": {"axis": "temperature", "op": ">", "value": 80},
                "then": {"axis": "pressure", "min": 2, "max": 5},
            }
        }
        expression = compile_gui_feasible_rule(rule, ["temperature", "pressure"])
        self.assertEqual(expression, "not (temperature > 80 and (pressure < 2 or pressure > 5))")

    def test_focused_2d_mask_compiles_without_all_scope(self) -> None:
        rule = {
            "sourceType": "focused_2d_mask",
            "guiSpec": {
                "type": "focused_2d_mask",
                "xAxis": "temperature",
                "yAxis": "pressure",
                "xMin": 80,
                "xMax": 100,
                "yMin": 0,
                "yMax": 2,
                "scope": {"concentration": {"mode": "all"}},
            },
        }
        expression = compile_focused_2d_mask_rule(rule, ["temperature", "pressure", "concentration"])
        self.assertEqual(
            expression,
            "not (temperature >= 80 and temperature <= 100 and pressure >= 0 and pressure <= 2)",
        )

    def test_focused_2d_mask_compiles_range_scope(self) -> None:
        rule = {
            "sourceType": "focused_2d_mask",
            "guiSpec": {
                "type": "focused_2d_mask",
                "xAxis": "temperature",
                "yAxis": "pressure",
                "xMin": 80,
                "xMax": 100,
                "yMin": 0,
                "yMax": 2,
                "scope": {"concentration": {"mode": "range", "min": 70, "max": 100}},
            },
        }
        expression = compile_focused_2d_mask_rule(rule, ["temperature", "pressure", "concentration"])
        self.assertIn("concentration >= 70", expression)
        self.assertIn("concentration <= 100", expression)

    def test_focused_2d_mask_rejects_same_axis(self) -> None:
        rule = {
            "sourceType": "focused_2d_mask",
            "guiSpec": {
                "type": "focused_2d_mask",
                "xAxis": "temperature",
                "yAxis": "temperature",
                "xMin": 80,
                "xMax": 100,
                "yMin": 0,
                "yMax": 2,
                "scope": {},
            },
        }
        with self.assertRaisesRegex(ValueError, "different X and Y"):
            compile_focused_2d_mask_rule(rule, ["temperature", "pressure"])

    def test_focused_2d_mask_rejects_bad_range(self) -> None:
        rule = {
            "sourceType": "focused_2d_mask",
            "guiSpec": {
                "type": "focused_2d_mask",
                "xAxis": "temperature",
                "yAxis": "pressure",
                "xMin": 100,
                "xMax": 80,
                "yMin": 0,
                "yMax": 2,
                "scope": {},
            },
        }
        with self.assertRaisesRegex(ValueError, "X minimum"):
            compile_focused_2d_mask_rule(rule, ["temperature", "pressure"])

    def test_focused_2d_mask_reduces_valid_bins_and_coverage_denominator(self) -> None:
        selected_goal = goal(
            axes=[
                axis("temperature", domain_max=100, resolution=10),
                axis("pressure", domain_max=10, resolution=1),
                axis("concentration", domain_max=100, resolution=10),
            ]
        )
        selected_goal["feasibleDomainRules"] = [
            {
                "sourceType": "focused_2d_mask",
                "editableMode": "gui",
                "enabled": True,
                "guiSpec": {
                    "type": "focused_2d_mask",
                    "xAxis": "temperature",
                    "yAxis": "pressure",
                    "xMin": 80,
                    "xMax": 100,
                    "yMin": 0,
                    "yMax": 2,
                    "scope": {"concentration": {"mode": "all"}},
                },
            }
        ]
        normalized = storage.validate_goal(selected_goal)
        mask_info = compute_valid_bin_mask_for_axes(normalized["axes"], normalized["feasibleDomainExpressions"])
        self.assertEqual(mask_info["totalBins"], 1000)
        self.assertEqual(mask_info["maskedBins"], 40)
        self.assertEqual(mask_info["validBins"], 960)
        peer = record_from_rows(normalized, [{"temperature": "40", "pressure": "5", "concentration": "50"}], "peer")
        coverage = app.build_global_bin_counts([peer], normalized)
        analyzer = DensityGridAnalyzer(app.experiment_config_from_goal(normalized))
        analyzer.set_peer_bin_counts(coverage["binCounts"])
        analyzer.set_feasible_domain(coverage["validBins"], coverage["maskedBins"], coverage["feasibleMaskEnabled"])
        result = analyzer.diagnose(coverage["binCounts"], {"validMultidimensionalRowCount": 1, "totalRows": 1})
        self.assertAlmostEqual(result.coverage_C, 1 / 960)

    def test_focused_2d_mask_marks_target_inside_domain_as_masked_out(self) -> None:
        selected_goal = goal(
            axes=[
                axis("temperature", domain_max=100, resolution=10),
                axis("pressure", domain_max=10, resolution=1),
                axis("concentration", domain_max=100, resolution=10),
            ]
        )
        selected_goal["feasibleDomainRules"] = [
            {
                "sourceType": "focused_2d_mask",
                "editableMode": "gui",
                "enabled": True,
                "guiSpec": {
                    "type": "focused_2d_mask",
                    "xAxis": "temperature",
                    "yAxis": "pressure",
                    "xMin": 80,
                    "xMax": 100,
                    "yMin": 0,
                    "yMax": 2,
                    "scope": {"concentration": {"mode": "all"}},
                },
            }
        ]
        normalized = storage.validate_goal(selected_goal)
        recomputed = app.recompute_bin_occupancy_from_row_vectors(
            [[90.0, 1.0, 50.0], [40.0, 5.0, 50.0]],
            ["temperature", "pressure", "concentration"],
            normalized,
        )
        self.assertEqual(recomputed["bin_occupancy_meta"]["outOfDomainRowCount"], 0)
        self.assertEqual(recomputed["bin_occupancy_meta"]["maskedOutRowCount"], 1)
        self.assertEqual(recomputed["bin_occupancy_meta"]["validMultidimensionalRowCount"], 1)

    def test_validate_goal_merges_enabled_gui_rules_and_advanced_expressions(self) -> None:
        payload = goal(axes=[axis("temperature", domain_max=100, resolution=10), axis("pressure", domain_max=10, resolution=1)])
        payload["feasibleDomainRules"] = [
            {
                "id": "rule_test",
                "sourceType": "gui_conditional",
                "editableMode": "gui",
                "enabled": True,
                "guiSpec": {
                    "type": "conditional_range",
                    "if": {"axis": "temperature", "op": ">", "value": 80},
                    "then": {"axis": "pressure", "min": 2, "max": 5},
                },
            }
        ]
        payload["feasibleDomainAdvancedExpressions"] = ["pressure >= 0"]
        normalized = storage.validate_goal(payload)
        self.assertEqual(len(normalized["feasibleDomainRules"]), 1)
        self.assertEqual(
            normalized["feasibleDomainExpressions"],
            ["not (temperature > 80 and (pressure < 2 or pressure > 5))", "pressure >= 0"],
        )

    def test_validate_goal_merges_focused_mask_rules(self) -> None:
        payload = goal(axes=[axis("temperature", domain_max=100, resolution=10), axis("pressure", domain_max=10, resolution=1)])
        payload["feasibleDomainRules"] = [
            {
                "id": "rule_mask",
                "sourceType": "focused_2d_mask",
                "editableMode": "gui",
                "enabled": True,
                "guiSpec": {
                    "type": "focused_2d_mask",
                    "xAxis": "temperature",
                    "yAxis": "pressure",
                    "xMin": 80,
                    "xMax": 100,
                    "yMin": 0,
                    "yMax": 2,
                    "scope": {},
                },
            }
        ]
        normalized = storage.validate_goal(payload)
        self.assertEqual(len(normalized["feasibleDomainRules"]), 1)
        self.assertEqual(normalized["feasibleDomainRules"][0]["sourceType"], "focused_2d_mask")
        self.assertEqual(
            normalized["feasibleDomainExpressions"],
            ["not (temperature >= 80 and temperature <= 100 and pressure >= 0 and pressure <= 2)"],
        )

    def test_existing_feasible_expressions_without_rules_remain_advanced(self) -> None:
        payload = goal(axes=[axis("temperature", domain_max=100, resolution=10), axis("pressure", domain_max=10, resolution=1)])
        payload["feasibleDomainExpressions"] = ["temperature <= 50"]
        normalized = storage.validate_goal(payload)
        self.assertEqual(normalized["feasibleDomainRules"], [])
        self.assertEqual(normalized["feasibleDomainAdvancedExpressions"], ["temperature <= 50"])
        self.assertEqual(normalized["feasibleDomainExpressions"], ["temperature <= 50"])

    def test_gui_rule_rejects_bad_range(self) -> None:
        rule = {
            "guiSpec": {
                "type": "conditional_range",
                "if": {"axis": "temperature", "op": ">", "value": 80},
                "then": {"axis": "pressure", "min": 5, "max": 2},
            }
        }
        with self.assertRaisesRegex(ValueError, "minimum"):
            compile_gui_feasible_rule(rule, ["temperature", "pressure"])

    def test_rule_builder_ui_controls_are_present(self) -> None:
        template = app.TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertIn("Feasible Rule Builder", template)
        self.assertIn("data-rule-convert", template)
        self.assertIn("Convert to Advanced Expression", template)
        self.assertIn("feasibleDomainAdvancedExpressions", template)
        self.assertIn("Focused 2D Mask Builder", template)
        self.assertIn("Add Focused 2D Mask Rule", template)
        self.assertIn("Reset Focused Mask Builder", template)
        self.assertIn("Focused 2D Mask requires at least 2 axes.", template)
        self.assertIn("data-admin-focused-field=\"xAxis\"", template)
        self.assertIn("data-admin-focused-scope-field=\"mode\"", template)
        self.assertIn("loadAdminFocusedRuleIntoBuilder", template)
        self.assertIn("data-rule-edit", template)

    def test_focused_mask_side_panel_ui_controls_are_present(self) -> None:
        template = app.TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertIn("Focused Mask Tool Settings", template)
        self.assertIn("leesinFocusedMaskToolSettings", template)
        self.assertIn("sourceType:'focused_2d_mask'", template)
        self.assertIn("data-focused-edit", template)
        self.assertIn("data-focused-convert", template)
        self.assertIn('id="focused-mask-x-axis"', template)
        self.assertIn('id="focused-mask-y-axis"', template)
        self.assertIn('id="focused-mask-add"', template)
        self.assertIn('id="focused-mask-save"', template)
        self.assertIn("data-projection-card", template)
        self.assertIn("setFocusedToolActivePair", template)
        self.assertIn("projection skipped: too many cells", template)
        self.assertIn("Request timed out. The grid may be too large; try coarser resolution.", template)
        self.assertIn("validBinsText", template)

    def test_filter_bin_counts_by_feasible_domain_reports_exclusions(self) -> None:
        axes = [axis("temperature", domain_max=100, resolution=10), axis("pressure", domain_max=10, resolution=1)]
        filtered = filter_bin_counts_by_feasible_domain(
            {"[5,4]": 2, "[5,6]": 3},
            axes,
            ["temperature <= 50"],
        )
        self.assertEqual(filtered["binCounts"], {"[5,4]": 2})
        self.assertEqual(filtered["infeasibleRows"], 3)
        self.assertEqual(filtered["infeasibleBins"], 1)

    def test_density_payload_includes_valid_domain_ratio(self) -> None:
        analyzer = DensityGridAnalyzer(ExperimentConfig(["x"], [(0, 10)], [1]))
        analyzer.set_peer_bin_counts({"[0]": 1})
        analyzer.set_feasible_domain(valid_bins=5, masked_bins=5, feasible_mask_enabled=True)
        result = analyzer.diagnose({"[0]": 1}, {"validMultidimensionalRowCount": 1, "totalRows": 1})
        payload = result.to_payload(["x"])
        self.assertEqual(payload["valid_domain_ratio"], 0.5)

    def test_projection_explorer_includes_feasible_valid_domain_ratio(self) -> None:
        selected_goal = goal(axes=[axis("x"), axis("y")])
        explorer = app.build_projection_explorer(
            selected_goal,
            {
                "binCounts": {},
                "feasibleMaskEnabled": True,
                "validBins": 20,
                "maskedBins": 80,
                "validDomainRatio": 0.2,
                "feasibleExpressions": ["x <= 5"],
            },
            [],
        )
        self.assertTrue(explorer["feasibleMaskEnabled"])
        self.assertEqual(explorer["validDomainRatio"], 0.2)
        self.assertEqual(explorer["feasibleExpressions"], ["x <= 5"])

    def test_csv_export_includes_feasible_mask_fields(self) -> None:
        response = app.export_report_request(
            {
                "format": "csv",
                "report": {
                    "meta": {"experiment_goal": "A", "goal_id": "goal_a", "axis_names": ["x"]},
                    "result": {
                        "engine": "density_grid",
                        "feasible_mask_enabled": True,
                        "feasibleExpressions": ["x <= 5"],
                        "valid_bins": 5,
                        "masked_bins": 5,
                        "valid_domain_ratio": 0.5,
                        "infeasible_target_rows": 2,
                        "infeasiblePeerRows": 3,
                        "infeasiblePeerBins": 1,
                    },
                    "visualizations": {},
                    "confidenceReasons": [],
                    "summary": [],
                },
            }
        )
        content = response["content"]
        self.assertIn("feasible_expressions", content)
        self.assertIn("valid_domain_ratio", content)
        self.assertIn("infeasible_target_rows", content)
        self.assertIn("infeasible_peer_rows", content)
        self.assertIn("feasible_mask_evaluation_skipped", content)
        self.assertIn("feasible_mask_warning", content)
        self.assertIn("coverage_warning", content)
        self.assertIn("reference_density_policy", content)
        self.assertIn("self_contained_reference_warning", content)


if __name__ == "__main__":
    unittest.main()
