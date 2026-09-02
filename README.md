# Leesin

Leesin is a local web-based research workflow prototype built around one question:

> **What information can the current data justify, and what should be observed next when it cannot?**

The current V4 prototype treats analysis as one step inside a repeatable experimental cycle rather than as the endpoint of the system.

```text
Data → Module → Mapping → Analysis → Result → Next
```

The repository also preserves the earlier V1–V3 implementations that led to this redesign.

---

## Current V4 prototype

The V4 interface is organized as:

```text
Projects / Files | Core workflow | Modules
```

### Core workflow

Each Project keeps one active experimental cycle.

- **Data** — select saved Data Clusters or original CSV/TSV project files.
- **Module** — attach an Analysis Module by clicking **Use** or dragging it from the Module shelf.
- **Mapping** — explicitly connect Module inputs to Data columns.
- **Analysis** — execute only the declared Module logic.
- **Result** — show the result together with declared assumptions, limits and diagnostics.
- **Next** — optionally continue to another observation/experiment and return to a new Data cycle.

`Question` is optional human-readable context. It is not an executable inference object; the actual analysis rule lives in the Module.

Previous editable stages can be revisited. Moving backward removes downstream state from the active Run, while backend Analysis/Proposal records move to Trash and the prior Run snapshot remains undoable.

### Analysis Modules

A reusable Module is treated as more than a bare Python function.

```text
Analysis Module = Function + Input Contract + Assumptions + Output/Limit metadata
```

The right-side Module shelf provides:

- **Browse / My / Favorites**
- author and version
- description and visibility
- tags and example Questions
- deterministic search across title, description, tags, inputs, outputs, assumptions and examples
- compatibility hints from the currently selected Data columns
- **Use** and drag-and-drop attachment
- local Fork / metadata editing for prototype Modules

### Module Workshop

Existing Python functions can be wrapped with a minimal workflow:

```text
Paste / Drop Python → Paste / Drop Data → Auto-map → Run → Save
```

The Workshop currently:

- inspects top-level functions with `ast`
- detects parameters, defaults, annotations and docstrings
- accepts pasted CSV/TSV/Excel clipboard tables
- accepts `.py` plus `.csv`/`.tsv` through the same file area
- suggests parameter-to-column mappings
- runs a restricted Python subset in a separate process with a timeout
- saves the function as a reusable Module

The runner is **not a security sandbox** and should only execute trusted local code.

### Project workspace

The left panel behaves like a small file explorer.

- folders and original-file storage
- original uploaded bytes preserved separately from parsed views
- rename, drag/move and multi-select
- Trash / Restore / permanent delete
- Project settings and Project deletion from the Projects sidebar

### Data lens

CSV/TSV data can be viewed in the same location through three interchangeable lenses:

- **Heat** — per-column numeric magnitude visualization
- **Table** — ordinary rows and columns
- **Raw** — the original delimited text

The Heat view is visual only. It does not modify values or claim cross-column comparability.

---

## V4 generality checks

Two small manual checks were used to test the separation between Core and Module.

### Same Module, different experiments

The same `Descriptive Summary` Module was applied without changing its code to:

1. Monte Carlo π estimation data — `abs_error → values`
2. Prime algorithm benchmark data — `runtime_ms → values`

Observed results included:

```text
Monte Carlo π
count = 100
mean abs_error = 0.056524000000000005

Prime benchmark
count = 404
mean runtime_ms = 0.002865346534653465
```

Only the Data and Mapping changed; the Module did not.

### Same Core, different Modules

The Monte Carlo π Project was then analyzed with `Pearson Correlation` using:

```text
sample_size → x
abs_error   → y
```

The Module returned:

```text
-0.4160325607005631
```

The analysis function changed, but the Core workflow remained:

```text
Data → Module → Mapping → Analysis → Result
```

These are prototype checks, not a proof of universal applicability.

---

## How Leesin reached V4

### V1 — anomaly detection and confidence separation

The first version asked whether an anomaly score and the reliability of that judgment should be treated separately. It used saved peer observations and several distance-based engines.

### V2 — row-level density and experiment-space coverage

V2 moved from record-level distance toward row-level density grids, introduced Domain Range / Resolution / Feasible Domain Mask, separated Input and Output axes, and used Grid Preview to expose unobserved regions.

### V3 — limits of confidence inference

V3 exposed a deeper problem: observed Data and the Information that can be justified from that Data are not the same thing. Some desired claims cannot be determined from the current observations without additional assumptions or additional experiments.

### V4 — experiment / analysis / next-observation loop

V4 therefore reframed Leesin around a reusable Core and explicit Analysis Modules. Analysis is no longer the endpoint; it can return a limit or a next observation that feeds a new experimental cycle.

Earlier V1/V2 code remains in the repository as part of this development history. The current V4 prototype is isolated under `v4_mvp/`.

---

## Run V4 locally

From the repository root:

```bash
python -m v4_mvp.app
```

Open:

```text
http://127.0.0.1:8765
```

After pulling a UI update, restart the Python process and use `Ctrl+F5` in the browser.

V4 server-side state is stored under `v4_mvp/runtime/`. Core cycle state, local identity, Favorites and prototype Registry metadata are currently browser-local.

---

## Verification

GitHub Actions checks the V4 branch with:

```bash
node --check v4_mvp/mvp_adapters/prime_ui.js
node --check v4_mvp/module_workshop_ui.js
node --check v4_mvp/module_file_input_ui.js
node --check v4_mvp/workspace_ui.js
node --check v4_mvp/ux_polish_ui.js
node --check v4_mvp/project_controls_ui.js
python -m compileall -q v4_mvp
python -m unittest tests.test_v4_mvp tests.test_module_workshop tests.test_workspace_store
```

Browser drag/drop, rollback/Undo and visual interactions are also smoke-tested manually.

---

## Scope boundary

Leesin V4 is intentionally finished as a **local service prototype**, not a deployed public service.

Not implemented as production infrastructure:

- real Google/GitHub OAuth
- server-side multi-user authorization
- hosted public/private Module Registry
- GPT/Sites Module discovery
- dependency environments
- robust sandboxing for arbitrary user Python
- signed Modules / trust policy
- generalized remote experiment adapters

Public deployment would require a separate security design for authentication, authorization, untrusted code execution and uploaded data. That work is outside the scope of the current prototype.