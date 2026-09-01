# Leesin_V4 MVP

This folder is an isolated MVP for the V4 direction. It intentionally does **not** replace the V1/V2 application yet.

## What this MVP tests

The UI follows the V4 workflow:

1. choose a **Project**
2. add or generate one or more **Data Clusters**
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

## Fastest test: run the MVP experiment inside Leesin

The prime benchmark executor is intentionally isolated under `v4_mvp/mvp_adapters/` so it can be deleted after this MVP. It is not part of the general Leesin core.

1. Open the seeded `Prime Algorithm Benchmark` project.
2. Click **⚗ Run MVP experiment**.
3. Enter two or more `N` values separated by commas, for example:

   `1000,100000`

4. Leesin runs both prime algorithms locally, records 7 repetitions after 1 warmup, and automatically creates a new Data Cluster. The Protocol and execution Context are filled consistently by the temporary adapter.
5. Choose **Prime algorithm performance crossover**.
6. Use **Select all** and click **Analyze selected data**.
7. Inspect **Preview / Result / Assumptions / Limits / Next**.
8. If a crossover is bracketed, click **Start next experiment**. For this MVP only, that button executes the proposed `N` directly and creates the next Data Cluster automatically.
9. Return to the Question, use **Select all**, and analyze again.

This closes the loop inside the MVP:

`Analysis → Proposal → experiment execution → new Cluster → Analysis`

without making built-in experiment execution a permanent Leesin assumption.

### Reset the experiment

Stop the server and delete:

```text
v4_mvp/runtime/store.json
```

The next launch will seed a fresh project.

## Manual benchmark path

The benchmark can still be run outside the UI when needed:

```bash
python -m v4_mvp.benchmark_prime \
  --n 10000 1000000 \
  --repeats 7 \
  --warmup 1 \
  --out initial.csv
```

Then add `initial.csv` as a cluster in the web UI.

Required columns are `N,algorithm,runtime_ms`; `repeat` is optional.

Recognized algorithm labels include `trial`, `trial_division`, `incremental_trial_division`, `sieve`, `eratosthenes`, and `sieve_of_eratosthenes`.

For the same analysis, use the same Protocol label and Context text for clusters that are meant to be directly comparable.

## Tests

```bash
python -m unittest tests.test_v4_mvp
```

The tests cover normal crossover bracket + midpoint proposal, multiple crossover assumption failure, protocol mismatch, and refusal to answer the undefined general-performance question.

## Removing the built-in MVP experiment later

Delete `v4_mvp/mvp_adapters/` and remove the small adapter import, `/mvp-prime-ui.js` route, HTML injection line, and `/mvp/prime-benchmark` endpoint from `v4_mvp/app.py`. The Question/Cluster/Analysis/Proposal core remains independent.

## MVP boundaries

Not implemented yet: authentication/accounts, public/private projects, external references/fork, schema editor, generalized plugin registry, probabilistic timing uncertainty, upload compatibility preview before analysis, and deletion/edit revision UI.
