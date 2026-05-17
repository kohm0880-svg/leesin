# Leesin

Leesin certifies CSV datasets against saved peer observations for each Experiment Goal.

The main engine is now a row-level density grid specificity detector. Leesin keeps each Goal's Axis, Domain Range, and Resolution settings, maps CSV rows into the multidimensional grid, accumulates saved peer row-level bin occupancy into a density map, and scores how rare the target rows are on that peer density map.

## Core Model

- CSV file = one saved data cluster.
- CSV rows = repeated observations inside that cluster.
- Raw rows, filenames, unmapped columns, and personal data are not stored.
- Saved records keep sanitized axis summaries, `binOccupancy`, `axisBinOccupancy`, `binOccupancyMeta`, `binOccupancyHash`, and `gridSignature`.
- The representative cluster vector is still stored for compatibility and display, but it is not the main analysis signal.

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

If `peer_valid_rows == 0`, density analysis is limited and Leesin returns a clear error. Legacy clusters without row-level bin occupancy or grid signature are excluded from density scoring.

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

`axisSignature` only tracks axis names. To prevent mixing incompatible grids, each saved cluster also stores `gridSignature`.

`gridSignature` hashes canonical JSON containing each selected axis's normalized name, `domainMin`, `domainMax`, and `resolution`. Density analysis only combines saved clusters whose `gridSignature` matches the current selected Goal grid.

The analysis payload reports:

- `coverageEligibleClusterCount`
- `coverageLegacyExcludedClusterCount`
- `coverageGridSignatureExcludedClusterCount`
- `rowLevelObservationCount`

## Stored Cluster Notes

New saved clusters include:

- `binOccupancy`: multidimensional bin key to row count
- `axisBinOccupancy`: 1D bin counts per axis for visualization
- `binOccupancyMeta`: valid, invalid, out-of-domain, and total row counts
- `binOccupancyHash`: duplicate detection support
- `gridSignature`: density grid compatibility guard

Duplicate detection still includes `binOccupancyHash`, so clusters with the same mean vector but different row-level occupancy are treated as distinct.

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

## Local Run

```powershell
.\run_app.ps1
```

Run tests:

```powershell
.\Lee_sin.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
