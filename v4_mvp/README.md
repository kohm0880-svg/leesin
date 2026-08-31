# Leesin_V4 MVP

This folder is an isolated MVP for the V4 direction. It intentionally does **not** replace the V1/V2 application yet.

## What this MVP tests

The UI follows the V4 workflow:

1. choose a **Project**
2. add one or more **Data Clusters**
3. choose the **Question** (the information you want)
4. select which clusters should be analyzed (`Select all` is provided)
5. run the analysis
6. review **Preview / Result / Assumptions / Limits / Next**
7. every analysis is automatically persisted as an immutable-style snapshot
8. if the module can justify a next observation, a **Proposal** is created and can be used to start the next experiment

The first real module is a `SingleBoundaryModule` for a computational experiment comparing:

- incremental trial division using already-found primes
- Sieve of Eratosthenes

The module asks where performance advantage changes under one declared execution environment.

It is deliberately able to **stop** rather than manufacture a result:

- `ASSUMPTION_FAILED` when winner order crosses more than once (A → B → A)
- `ANALYSIS STOPPED` on Protocol/Context mismatch
- `INSUFFICIENT INFORMATION` when no crossover is bracketed
- `INSUFFICIENT INFORMATION` for the intentionally broad question “which algorithm is generally faster?” because the target environment population is undefined

This is the core V4 distinction being tested: the system manages the link

`Question → selected Data → Assumptions → Information / Limits → Next Observation`

instead of treating every available numeric value as sufficient to answer every question.

## Run the MVP

From the repository root:

```bash
python -m v4_mvp.app
```

Open `http://127.0.0.1:8765`.

No login exists yet. The app seeds one project named `Prime Algorithm Benchmark`.

Runtime state is written to `v4_mvp/runtime/store.json`.

Override it with:

```bash
LEESIN_V4_STORE=/path/to/store.json python -m v4_mvp.app
```

## Generate benchmark CSV data

Choose the repeat count as part of your own protocol rather than letting Leesin infer a “sufficient” value:

```bash
python -m v4_mvp.benchmark_prime \
  --n 10000 1000000 \
  --repeats 5 \
  --warmup 1 \
  --out initial.csv
```

Then add `initial.csv` as a cluster in the web UI.

Required columns are `N,algorithm,runtime_ms`; `repeat` is optional.

Recognized algorithm labels include `trial`, `trial_division`, `incremental_trial_division`, `sieve`, `eratosthenes`, and `sieve_of_eratosthenes`.

For the same analysis, use the same Protocol label and Context text for clusters that are meant to be directly comparable.

## Suggested first experiment

1. Benchmark two distant N values.
2. Upload the CSV as Cluster 001.
3. Choose `Prime algorithm performance crossover`.
4. Select the cluster and analyze.
5. If a crossover is bracketed, Leesin proposes the integer midpoint.
6. Click `Start next experiment`.
7. Run the displayed benchmark command with the same protocol.
8. Upload that CSV as the next cluster.
9. Re-run the question with `Select all`.

The analysis history is auto-saved; a later cluster does not rewrite earlier analyses.

## Tests

```bash
python -m unittest tests.test_v4_mvp
```

The tests cover normal crossover bracket + midpoint proposal, multiple crossover assumption failure, protocol mismatch, and refusal to answer the undefined general-performance question.

## MVP boundaries

Not implemented yet: authentication/accounts, public/private projects, external references/fork, schema editor, generalized plugin registry, probabilistic timing uncertainty, automatic environment discovery, upload compatibility preview before analysis, and deletion/edit revision UI.
