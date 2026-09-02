# Leesin_V4 MVP

This folder is an isolated MVP for the V4 direction. It intentionally does **not** replace the V1/V2 application yet.

## What this MVP tests

The original project flow still tests:

1. choose a **Project**
2. add or generate one or more **Data Clusters**
3. choose a **Question**
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

## Module Workshop: Paste / Drop → Map → Run → Save

V4 now also contains a separate `Module Workshop` prototype. Its purpose is to test a different product direction: Leesin does not infer the meaning of an arbitrary research question on its own. Instead, the executable analysis logic is explicit in an Analysis Module, while a human-readable Question can remain optional semantic context.

Open **Modules** in the top bar and:

1. paste an existing Python function, or drop/browse a `.py` file
2. paste CSV/TSV/Excel clipboard data, or drop/browse a `.csv` / `.tsv` file
3. the same Workshop input area accepts a `.py` and a data file together and routes each to the correct editor
4. Leesin detects top-level functions, function parameters, table columns, and suggests parameter → column mappings
5. correct only ambiguous mappings
6. click **Run**
7. click **Save Module**; Description / Question / Assumptions / Limits are optional
8. use a saved Module again with different data, or **Copy JSON** / **Paste Module JSON** to transfer the Module by copy-paste

A simple example:

```python
def average(values):
    return sum(values) / len(values)
```

Paste data such as:

```text
values
1
2
3
```

The Workshop maps `values ← values`, executes the function, and returns `2.0`.

The current runner is deliberately narrow. It executes a restricted Python subset in a separate process with a timeout and supports common builtins plus curated `math` / `statistics` imports. It is **not a complete security sandbox**, so only trusted local code should be pasted. NumPy/Pandas compatibility, dependency environments, public accounts/registry, signing, and server-grade sandboxing are later work.

Saved Workshop modules are written separately to:

```text
v4_mvp/runtime/custom_modules.json
```

Override it with `LEESIN_V4_MODULE_STORE`.

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
3. Use the default initial values `5,100` unless you want to choose a different starting bracket.
4. Leesin runs both prime algorithms locally, records 101 repetitions after 3 warmups, and automatically creates a new Data Cluster. This repeat count is only a declared test protocol chosen because the crossover occurs at tiny runtimes; it is not a confidence score or a claim of statistical sufficiency.
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
  --n 5 100 \
  --repeats 101 \
  --warmup 3 \
  --out initial.csv
```

Then add `initial.csv` as a cluster in the web UI.

Required columns are `N,algorithm,runtime_ms`; `repeat` is optional.

Recognized algorithm labels include `trial`, `trial_division`, `incremental_trial_division`, `sieve`, `eratosthenes`, and `sieve_of_eratosthenes`.

For the same analysis, use the same Protocol label and Context text for clusters that are meant to be directly comparable.

## Tests

```bash
python -m unittest tests.test_v4_mvp tests.test_module_workshop
```

The Workshop tests cover signature inspection, Excel-style TSV parsing, automatic mapping, whole-table input, restricted execution, curated imports, and module persistence.

## Removing the built-in MVP experiment later

Delete `v4_mvp/mvp_adapters/` and remove the small adapter import, `/mvp-prime-ui.js` route, HTML injection line, and `/mvp/prime-benchmark` endpoint from `v4_mvp/app.py`. The Project/Cluster/Analysis/Proposal core and Module Workshop remain independent.

## MVP boundaries

Not implemented yet: authentication/accounts, public/private module registry, dependency environments, robust server sandboxing, external references/fork graph, generalized Module version publication, probabilistic timing uncertainty, upload compatibility preview before analysis, and deletion/edit revision UI.
