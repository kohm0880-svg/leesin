from __future__ import annotations

import csv
import io
import platform
from typing import Any

from v4_mvp.benchmark_prime import benchmark


ADAPTER_ID = "prime-benchmark-mvp-v1"
# The first crossover for these pure-Python implementations occurs at very small
# N, so individual timings are tiny. 101 repetitions are a declared experimental
# protocol choice to make the median less jumpy; they are NOT a confidence score
# or a claim that 101 samples are statistically sufficient.
DEFAULT_REPEATS = 101
DEFAULT_WARMUP = 3


def execution_context() -> str:
    """Return a stable declared context for comparability checks.

    Do not include timestamps or volatile measurements: clusters generated on the
    same machine/runtime should carry exactly the same context string.
    """
    return "; ".join(
        [
            f"system={platform.system()} {platform.release()}",
            f"machine={platform.machine() or '(unknown)'}",
            f"processor={platform.processor() or '(unspecified)'}",
            f"python={platform.python_implementation()} {platform.python_version()}",
        ]
    )


def protocol_label(repeats: int = DEFAULT_REPEATS, warmup: int = DEFAULT_WARMUP) -> str:
    return f"{ADAPTER_ID}; repeats={repeats}; warmup={warmup}; statistic=median"


def generate_cluster_payload(
    n_values: list[int],
    *,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
    origin_proposal_id: str | None = None,
) -> dict[str, Any]:
    """Execute the temporary MVP benchmark and return an add_cluster payload."""
    if not n_values:
        raise ValueError("nValues must contain at least one integer.")
    normalized = sorted({int(value) for value in n_values})
    if any(value < 2 for value in normalized):
        raise ValueError("Every N must be at least 2.")

    rows = benchmark(normalized, repeats=repeats, warmup=warmup)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["N", "algorithm", "runtime_ms", "repeat"])
    writer.writeheader()
    writer.writerows(rows)

    if len(normalized) == 1:
        cluster_name = f"MVP benchmark N={normalized[0]}"
        filename = f"prime_benchmark_N{normalized[0]}.csv"
    else:
        cluster_name = f"MVP benchmark N={normalized[0]}..{normalized[-1]}"
        filename = f"prime_benchmark_{normalized[0]}_{normalized[-1]}.csv"

    return {
        "name": cluster_name,
        "filename": filename,
        "csv_text": buffer.getvalue(),
        "protocol": protocol_label(repeats, warmup),
        "context": execution_context(),
        "origin_proposal_id": origin_proposal_id,
        "experiment": {
            "adapter": ADAPTER_ID,
            "nValues": normalized,
            "repeats": repeats,
            "warmup": warmup,
        },
    }
