# Leesin_V4 MVP

This folder is an isolated prototype for the V4 direction. It does **not** replace the V1/V2 application yet.

## Current product structure

The interface follows a three-pane model:

`Projects / Files | Core workflow | Modules`

### Left: Project Explorer

The left panel behaves like a small file explorer rather than a decorative tree.

- real expand/collapse controls
- folders and original-file storage
- Ctrl/Cmd multi-select and Shift range-select
- batch move to Trash
- Restore and permanent delete
- file/folder rename
- drag files and folders into folders
- existing Clusters, Analyses and Proposals can also be moved to Trash

Uploaded project files retain their original bytes in the local MVP store. Parsed tables, previews and mappings are derived views; they do not replace the source file.

### Center: stateful Core workflow

The current experimental cycle is represented as:

`Data → Module → Mapping → Analysis → Result → Next`

`Question` is no longer the executable selector. A Question is optional human-readable context attached to the chosen Module application. The Module contains the actual analysis function and its declared conditions.

The editable previous stages are clickable. Returning to an earlier stage removes downstream state from the active Run. Existing backend Analysis/Proposal records are moved to workspace Trash rather than immediately destroyed, and an Undo action restores the previous snapshot. `Analysis` itself is a transient execution state; return to `Mapping` to rerun it.

When a next observation is executed by the prime MVP adapter, the completed cycle is archived and the Core returns to `Data` with the new Cluster included.

### Right: Module shelf

The right panel is the canonical Module location.

- **Browse**: public/built-in registry examples plus locally saved Modules
- **My**: locally owned Modules
- **Favorites**: user-specific starred Modules
- author, version, visibility, description and tags
- click **Use** or drag a Module into the center Module slot
- deterministic search across title, description, author, tags, inputs, outputs, assumptions and example Questions
- compatibility hints based on the currently selected Data columns
- public examples can be Forked into My Modules

The built-in registry entries are currently local demo metadata, not a deployed server registry. GPT/Sites discovery is intentionally deferred until the registry has a stable API.

## Data lens

CSV/TSV data can be viewed through three interchangeable lenses in the same location:

- **Heat**: a grid in which numeric magnitude is encoded within each column
- **Table**: the original cell values in a normal grid
- **Raw**: the underlying delimited text

The Heat view is only a visual encoding. It does not alter the values or claim that columns are directly comparable to one another.

## Local account prototype

The top-right account control provides a local identity used for Module authorship, Project ownership, Favorites and public/private metadata.

This is **not** Google/GitHub OAuth. A local server has no hosted callback or credentials, so the current screen deliberately identifies itself as a local MVP profile. Real authentication remains a deployment task.

## Module Workshop

Module creation remains a center-screen workflow:

`Paste / Drop Python → Paste / Drop Data → Auto-map → Run → Save`

The Workshop can:

- inspect top-level Python functions with `ast`
- detect parameters, defaults, annotations and docstrings
- accept pasted CSV, TSV and Excel clipboard tables
- accept `.py` plus `.csv`/`.tsv` by drop or file picker
- suggest parameter-to-column mappings
- run a deliberately restricted Python subset in a separate process with a timeout
- save the function as a reusable Module
- transfer a Module by JSON copy/paste

The runner is **not a complete security sandbox**. Only trusted local code should be executed. Dependency environments, NumPy/Pandas packaging and server-grade isolation remain future work.

Saved Workshop Modules are written to:

```text
v4_mvp/runtime/custom_modules.json
```

unless `LEESIN_V4_MODULE_STORE` overrides it.

## Prime boundary MVP

The first concrete Core/Module experiment compares incremental trial division with Sieve of Eratosthenes.

The built-in `Single Boundary` Module:

- uses the declared `N`, `algorithm`, `runtime_ms` binding
- checks its single-crossover assumption
- returns Result / Assumptions / Limits / Diagnostics
- creates a Proposal when another integer `N` can narrow the bracket
- can execute that Proposal through the temporary prime adapter

The adapter remains isolated under `v4_mvp/mvp_adapters/` because built-in experiment execution is not assumed to be a permanent Core responsibility.

## Run

From the repository root:

```bash
python -m v4_mvp.app
```

Open:

```text
http://127.0.0.1:8765
```

After pulling a UI update, restart the Python process and use `Ctrl+F5` in the browser.

Server-side MVP state is stored at:

```text
v4_mvp/runtime/store.json
```

unless `LEESIN_V4_STORE` overrides it. Core cycle state, local identity, Favorites and registry metadata are currently browser-local (`localStorage`).

## Backend tests

```bash
python -m unittest tests.test_v4_mvp tests.test_module_workshop tests.test_workspace_store
```

These tests cover the prime module, Module Workshop execution and workspace storage. The browser state-machine and visual interactions still require a manual browser smoke test.

## Deliberately deferred

- real Google/GitHub OAuth and server accounts
- persistent server-side ownership/Favorites
- hosted public/private Module Registry API
- GPT/Sites Module discovery and reranking
- dependency environments and robust remote-code sandboxing
- signed Modules and trust policy
- published version/fork graph
- generalized experiment adapters
- probabilistic timing uncertainty
