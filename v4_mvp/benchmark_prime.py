from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def incremental_trial_primes(limit: int) -> list[int]:
    if limit < 2:
        return []
    primes: list[int] = []
    for candidate in range(2, limit + 1):
        is_prime = True
        for prime in primes:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
    return primes


def sieve_primes(limit: int) -> list[int]:
    if limit < 2:
        return []
    flags = bytearray(b"\x01") * (limit + 1)
    flags[0:2] = b"\x00\x00"
    stop = int(limit ** 0.5)
    for prime in range(2, stop + 1):
        if flags[prime]:
            start = prime * prime
            step_count = ((limit - start) // prime) + 1
            flags[start : limit + 1 : prime] = b"\x00" * step_count
    return [index for index, flag in enumerate(flags) if flag]


def _measure(function, n_value: int) -> tuple[float, list[int]]:
    started = time.perf_counter_ns()
    result = function(n_value)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return elapsed_ms, result


def benchmark(n_values: list[int], repeats: int, warmup: int) -> list[dict[str, object]]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    rows: list[dict[str, object]] = []
    for n_value in n_values:
        if n_value < 2:
            raise ValueError("N must be at least 2")

        for _ in range(warmup):
            trial = incremental_trial_primes(n_value)
            sieve = sieve_primes(n_value)
            if trial != sieve:
                raise RuntimeError("Prime algorithms disagree during warmup.")

        for repeat in range(1, repeats + 1):
            trial_ms, trial = _measure(incremental_trial_primes, n_value)
            sieve_ms, sieve = _measure(sieve_primes, n_value)
            if trial != sieve:
                raise RuntimeError(f"Prime algorithms disagree at N={n_value}.")
            rows.append({"N": n_value, "algorithm": "trial", "runtime_ms": f"{trial_ms:.6f}", "repeat": repeat})
            rows.append({"N": n_value, "algorithm": "sieve", "runtime_ms": f"{sieve_ms:.6f}", "repeat": repeat})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prime-search benchmark CSV data for the Leesin_V4 MVP.")
    parser.add_argument("--n", nargs="+", type=int, required=True, help="One or more N values.")
    parser.add_argument("--repeats", type=int, required=True, help="Recorded repetitions per algorithm and N. This is part of the experiment protocol.")
    parser.add_argument("--warmup", type=int, default=1, help="Unrecorded warmup repetitions per N (default: 1). Record this choice in Protocol.")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    args = parser.parse_args()

    rows = benchmark(args.n, args.repeats, args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["N", "algorithm", "runtime_ms", "repeat"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
