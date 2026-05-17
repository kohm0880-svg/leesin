# Leesin

Leesin certifies CSV datasets against saved peer observations for each Experiment Goal.

The main engine is now a row-level density grid specificity detector. Leesin keeps each Goal's Axis, Domain Range, and Resolution settings, maps CSV rows into the multidimensional grid, accumulates saved peer row-level bin occupancy into a density map, and scores how rare the target rows are on that peer density map.

## Core Model

- CSV file = one saved record.
- CSV rows = row-level observations inside that record.
- Raw rows, filenames, unmapped columns, and personal data are not stored.
- Saved records keep selected-axis sanitized numeric row vectors in `rowLevelVectors`.
- Saved records also keep compatibility axis summaries, `binOccupancy`, `axisBinOccupancy`, `binOccupancyMeta`, `binOccupancyHash`, and `gridSignature`.
- A mean vector is still stored for compatibility and display, but it is not the main analysis signal.

Storage and deletion happen at the record level. Leesin's density calculation does not treat a record mean vector as the sample; it re-bins the record's row-level sanitized axis vectors into the current density grid. When Domain Range or Resolution changes, saved row-level vectors can be used to recompute `binOccupancy` and `gridSignature`.

## Specificity Engine

For each occupied target bin `g`, Leesin estimates peer density with Jeffreys-style smoothing:

```text
p_hat_g = (peer_bin_count[g] + 0.5) / (peer_valid_rows + 0.5 * total_bins)
rarity_g = -ln(p_hat_g)
```

Target bin counts weight the row-level rarity:

```text
mean_rarity = sum(target_count[g] * rarity_g) / target_valid_rows
specificity_score = 1 - exp(-mean_rarity / log(peer_valid_rows + total_bins + 1))
```

Additional result fields include `max_rarity`, `unseen_bin_rate`, `rare_bin_rate`, and `out_of_domain_rate`.

Specificity Score is an engineering score that compresses peer-density rarity into a 0-1 range. It is not a p-value, posterior probability, or statistical probability.

If `peer_valid_rows == 0`, density analysis is limited and Leesin returns a clear error. Legacy records without row-level vectors are excluded from density scoring.

## Confidence

Confidence is engineering confidence, not a p-value. It combines:

- `observation_support_S = peer_valid_rows / (peer_valid_rows + K_density)`
- `coverage_C = occupied_bins / total_bins`
- `equitability_E = -sum(p_i ln(p_i)) / ln(occupied_bins)`

`K_density` currently reuses each Goal's `K_m` value so the code can rename it cleanly later.

```text
confidence = (observation_support_S * coverage_C * equitability_E) ** (1/3)
```

## Grid Signature

`axisSignature` only tracks axis names. To guard stored grid metadata, each saved record also stores `gridSignature`.

`gridSignature` hashes canonical JSON containing each selected axis's normalized name, `domainMin`, `domainMax`, and `resolution`. The canonical axes list is sorted by normalized axis name, and bin occupancy keys are generated in that same canonical order. Density analysis only combines saved records whose `gridSignature` matches the current selected Goal grid.

The analysis payload reports:

- `coverageEligibleClusterCount`
- `coverageLegacyExcludedClusterCount`
- `coverageGridSignatureExcludedClusterCount`
- `rowLevelObservationCount`

## Projection Explorer

Leesin visualizes high-dimensional density grids through linked 2D axis-pair projections.

- Peer density is collapsed from saved multidimensional `binOccupancy` into each 2D projection.
- Target rows are represented only as bin-index tuples during the analysis response; raw row values are not stored or sent.
- Clicking a bin in one projection selects the target row subset in that 2D bin and highlights where that subset appears in all other projections.
- Crosshair markers show the selected axis/bin location on every projection that includes the selected axes.
- Modes include combined, peer only, target only, and selected subset only.
- `Ctrl + wheel` zooms each heatmap independently, drag pans the zoomed view, and double-click or Reset zoom restores the projection.

This helps users inspect high-dimensional specificity without directly visualizing a 4D or higher-dimensional object. Selection A/B persistence is intentionally not part of this workflow.

## Grid Preview Editor

The report view includes a Grid Preview Editor for Domain Range and Resolution.

- Preview changes are recalculated in memory and are not stored.
- Preview metrics include Z / Observation Support, C / Coverage, E / Equitability, Confidence, total bins, occupied bins, peer valid rows, eligible records, and legacy excluded records.
- Preview recalculation uses saved `rowLevelVectors`, not stale stored bin occupancy.
- Apply as Goal Default asks for confirmation, then updates the Goal grid and recomputes saved records that have row-level vectors.
- Legacy records without row-level vectors remain stored but are excluded from density calculation and preview recalculation.

## Stored Record Notes

New saved records include:

- `rowLevelVectors`: selected-axis sanitized numeric row vectors
- `rowLevelVectorAxisOrder`: canonical axis order for those vectors
- `rowLevelVectorCount`: number of sanitized numeric rows
- `rowLevelVectorBasis`: currently `valid_multidimensional_numeric_rows`
- `hasRowLevelVectors`: whether the record can be re-binned for new grids
- `binOccupancy`: multidimensional bin key to row count
- `axisBinOccupancy`: 1D bin counts per axis for visualization
- `binOccupancyMeta`: valid, invalid, out-of-domain, and total row counts
- `binOccupancyHash`: duplicate detection support
- `gridSignature`: density grid compatibility guard

Duplicate detection still includes `binOccupancyHash`, so records with the same mean vector but different row-level occupancy are treated as distinct.

## API Shape

Analysis results include density fields such as:

- `engine = "density_grid"`
- `specificity_score`
- `mean_rarity`
- `max_rarity`
- `unseen_bin_rate`
- `rare_bin_rate`
- `out_of_domain_rate`
- `observation_support_S`
- `coverage_C`
- `equitability_E`
- `confidence`
- `peer_observation_count`
- `valid_target_rows`
- `invalid_target_rows`
- `out_of_domain_rows`
- `total_bins`
- `occupied_bins`

Report visualizations include `projectionExplorer`, which carries canonical `axisOrder`, `axisMeta`, all 2D `axisPairs`, peer/target projection matrices, and target row bin-index tuples for in-session linked highlighting.

## Admin Token UX

Admin endpoints check the `X-Admin-Token` header when remote admin authentication is required. In Render deployments, users can enter the `ADMIN_TOKEN` value in the Admin Token panel. The token is stored only in the current browser's `localStorage` under `leesinAdminToken`; it is not stored on the server. Saved tokens are automatically attached to `/api/admin/*` requests, and 403 responses highlight the Admin Token panel with a Render-specific help message.

## Local Run

```powershell
.\run_app.ps1
```

Run tests:

```powershell
.\Lee_sin.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
