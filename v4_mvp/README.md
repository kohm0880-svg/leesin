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

The first real module is a `SingleBoundaryModule` for a computational experiment comparing incremental trial division and Sieve of Eratosthenes.

## Three-pane workspace prototype

The current V4 UI now follows the layout:

`Projects / Files | Core workspace | Modules`

The left panel is a file-explorer-style Project workspace. It adds real expand/collapse behavior, project folders, original-file storage, Ctrl/Cmd multi-select, Shift range-select, batch Trash, Restore, permanent delete, rename, and drag-to-folder movement. Project files are stored as their original bytes (base64 in the local MVP store) plus optional text preview metadata; Leesin's parsed/derived representation is not treated as the original file.

The center panel keeps the current work area and adds a visible Throughout step bar:

`Data → Module → Mapping → Analysis → Result → Next`

At this stage `Data` and `Module` are navigable; destructive rollback of downstream Core state is intentionally left for the next Core-state-machine iteration rather than faked in the UI.

The right panel is a persistent Module shelf. It currently shows local **Favorites** and **My Modules**, allows quick Use/New actions, and provides simple local filtering. Natural-language/GPT discovery is intentionally deferred until a real Module Registry exists.

Both side panels can be collapsed and resized.

Workspace state uses the same local JSON store as the MVP but is kept under separate `workspace_files`, `workspace_folders`, and `workspace_trash` keys so the original V4 experiment objects can remain independent. Deleting an existing Cluster/Analysis/Proposal from the explorer moves the original object into workspace Trash rather than immediately destroying it.

## Module Workshop: Paste / Drop → Map → Run → Save

V4 also contains a `Module Workshop` prototype. Leesin does not infer the meaning of an arbitrary research question on its own. The executable analysis logic is explicit in an Analysis Module, while a human-readable Question can remain optional semantic context.

Open **Modules** and:

1. paste an existing Python function, or drop/browse a `.py` file
2. paste CSV/TSV/Excel clipboard data, or drop/browse `.csv` / `.tsv`
3. Leesin detects functions, parameters and table columns and suggests mappings
4. correct only ambiguous mappings
5. click **Run**
6. click **Save Module**; Description / Question / Assumptions / Limits are optional
7. reuse it with different data, or **Copy JSON** / **Paste Module JSON**

A simple example:

```python
def average(values):
    return sum(values) / len(values)
```

with

```text
values
1
2
3
```

maps `values ← values` and returns `2.0`.

The current runner executes a restricted Python subset in a separate process with a timeout and supports common builtins plus curated `math` / `statistics` imports. It is **not a complete security sandbox**, so only trusted local code should be used.

Saved Workshop modules are written separately to `v4_mvp/runtime/custom_modules.json` unless `LEESIN_V4_MODULE_STORE` overrides it.

## Run the MVP

From the repository root:

```bash
python -m v4_mvp.app
```

Open `http://127.0.0.1:8765`.

No login exists yet. Runtime state is written to `v4_mvp/runtime/store.json` unless `LEESIN_V4_STORE` overrides it.

## Fastest prime-loop test

The prime benchmark executor remains intentionally isolated under `v4_mvp/mvp_adapters/`.

1. Open the seeded `Prime Algorithm Benchmark` project.
2. Click **Run MVP experiment**.
3. Use the initial values `5,100` unless you want another starting bracket.
4. Leesin runs both prime algorithms locally, records 101 repetitions after 3 warmups, and creates a Data Cluster. This is a protocol choice, not a confidence score or statistical sufficiency claim.
5. Choose **Prime algorithm performance crossover**.
6. Use **Select all** and Analyze.
7. If a crossover is bracketed, click **Start next experiment** and analyze all clusters again.

This closes the test loop:

`Analysis → Proposal → experiment execution → new Cluster → Analysis`

without making built-in experiment execution a permanent Leesin assumption.

## Tests

```bash
python -m unittest tests.test_v4_mvp tests.test_module_workshop tests.test_workspace_store
```

The Workspace tests cover original-byte preservation, Trash/Restore, and removing legacy Analysis/Proposal objects from the active Core project when they are moved to Trash.

## MVP boundaries

Not implemented yet: authentication/accounts, public/private Module Registry, real module authorship and server-side Favorites, natural-language/GPT module discovery, dependency environments, robust server sandboxing, external references/fork graph, generalized published Module versioning, destructive downstream Core rollback/revision UI, and probabilistic timing uncertainty.
