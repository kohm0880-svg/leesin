from __future__ import annotations

import csv
import io
import statistics
from dataclasses import dataclass, asdict
from typing import Any


MODULE_VERSION = "0.1.0"

QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "prime_crossover",
        "name": "Prime algorithm performance crossover",
        "description": (
            "현재 명시된 실행환경에서 incremental trial division과 "
            "Sieve of Eratosthenes의 성능 우위가 한 번 전환된다고 가정할 때 "
            "그 경계 범위를 찾습니다."
        ),
        "requiredColumns": ["N", "algorithm", "runtime_ms"],
        "module": "SingleBoundaryModule",
        "supportsProposal": True,
    },
    {
        "id": "general_prime_speed",
        "name": "Which prime algorithm is generally faster?",
        "description": (
            "한 실행환경의 benchmark만으로 '일반적으로 더 빠르다'는 결론을 "
            "정당화할 수 있는지 확인하는 정보 범위 검증 질문입니다."
        ),
        "requiredColumns": ["N", "algorithm", "runtime_ms"],
        "module": "ScopeBoundaryDiagnostic",
        "supportsProposal": False,
    },
]


@dataclass
class Outcome:
    status: str
    status_label: str
    title: str
    summary: str
    preview: list[dict[str, Any]]
    assumptions: list[str]
    limits: list[str]
    diagnostics: list[str]
    proposal: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def question_registry() -> list[dict[str, Any]]:
    return [dict(item) for item in QUESTIONS]


def get_question(question_id: str) -> dict[str, Any]:
    for item in QUESTIONS:
        if item["id"] == question_id:
            return dict(item)
    raise ValueError(f"Unknown question: {question_id}")


def _normalize_header(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def _normalize_algorithm(value: str) -> str | None:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    trial_aliases = {
        "trial",
        "trial_division",
        "incremental_trial",
        "incremental_trial_division",
        "prime_trial",
    }
    sieve_aliases = {
        "sieve",
        "eratosthenes",
        "sieve_of_eratosthenes",
        "eratosthenes_sieve",
    }
    if token in trial_aliases:
        return "trial"
    if token in sieve_aliases:
        return "sieve"
    return None


def _parse_cluster_csv(cluster: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    text = str(cluster.get("csvText") or "")
    if not text.strip():
        return [], [f"{cluster.get('name', 'cluster')}: CSV content is empty."]

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return [], [f"{cluster.get('name', 'cluster')}: CSV header is missing."]

    header_map = {_normalize_header(name): name for name in reader.fieldnames if name is not None}
    n_key = header_map.get("n")
    algorithm_key = header_map.get("algorithm") or header_map.get("method")
    runtime_key = (
        header_map.get("runtime_ms")
        or header_map.get("time_ms")
        or header_map.get("elapsed_ms")
        or header_map.get("runtime")
    )
    missing: list[str] = []
    if not n_key:
        missing.append("N")
    if not algorithm_key:
        missing.append("algorithm")
    if not runtime_key:
        missing.append("runtime_ms")
    if missing:
        return [], [
            f"{cluster.get('name', 'cluster')}: missing required column(s): {', '.join(missing)}."
        ]

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for row_number, row in enumerate(reader, start=2):
        try:
            n_value = int(str(row.get(n_key, "")).strip())
            runtime_ms = float(str(row.get(runtime_key, "")).strip())
        except (TypeError, ValueError):
            errors.append(
                f"{cluster.get('name', 'cluster')}: row {row_number} has invalid N/runtime."
            )
            continue
        if n_value < 2 or runtime_ms < 0:
            errors.append(
                f"{cluster.get('name', 'cluster')}: row {row_number} has out-of-range N/runtime."
            )
            continue
        algorithm = _normalize_algorithm(str(row.get(algorithm_key, "")))
        if algorithm is None:
            errors.append(
                f"{cluster.get('name', 'cluster')}: row {row_number} uses an unknown algorithm label."
            )
            continue
        rows.append(
            {
                "N": n_value,
                "algorithm": algorithm,
                "runtime_ms": runtime_ms,
                "clusterId": cluster.get("id"),
                "clusterName": cluster.get("name"),
            }
        )
    return rows, errors


def _environment_signature(cluster: dict[str, Any]) -> tuple[str, str]:
    protocol = str(cluster.get("protocol") or "").strip() or "(unspecified)"
    context = str(cluster.get("context") or "").strip() or "(unspecified)"
    return protocol, context


def _aggregate_preview(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["N"]), {"trial": [], "sieve": []})
        grouped[int(row["N"])][str(row["algorithm"])].append(float(row["runtime_ms"]))

    preview: list[dict[str, Any]] = []
    for n_value in sorted(grouped):
        trial_values = grouped[n_value]["trial"]
        sieve_values = grouped[n_value]["sieve"]
        trial_median = statistics.median(trial_values) if trial_values else None
        sieve_median = statistics.median(sieve_values) if sieve_values else None
        winner: str | None = None
        if trial_median is not None and sieve_median is not None:
            if trial_median < sieve_median:
                winner = "trial"
            elif sieve_median < trial_median:
                winner = "sieve"
            else:
                winner = "tie"
        preview.append(
            {
                "N": n_value,
                "trialMedianMs": trial_median,
                "sieveMedianMs": sieve_median,
                "trialRepeats": len(trial_values),
                "sieveRepeats": len(sieve_values),
                "winner": winner,
            }
        )
    return preview


def _analysis_input(clusters: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for cluster in clusters:
        parsed, cluster_errors = _parse_cluster_csv(cluster)
        rows.extend(parsed)
        errors.extend(cluster_errors)
    return rows, errors


def analyze_prime_crossover(clusters: list[dict[str, Any]]) -> Outcome:
    if not clusters:
        return Outcome(
            status="insufficient_information",
            status_label="INSUFFICIENT INFORMATION",
            title="No data selected",
            summary="분석에 사용할 군집을 하나 이상 선택해야 합니다.",
            preview=[],
            assumptions=[],
            limits=["선택된 데이터가 없습니다."],
            diagnostics=[],
        )

    signatures = {_environment_signature(cluster) for cluster in clusters}
    if len(signatures) > 1:
        return Outcome(
            status="protocol_mismatch",
            status_label="ANALYSIS STOPPED",
            title="Selected clusters do not share one declared environment",
            summary=(
                "서로 다른 Protocol 또는 Context가 선언된 군집을 자동으로 하나의 "
                "성능 reference로 합치지 않았습니다."
            ),
            preview=[],
            assumptions=["선택된 군집은 동일한 실행조건에서 비교 가능해야 합니다."],
            limits=[
                "현재 선택만으로는 서로 다른 Protocol/Context의 성능값을 동일한 분포로 취급할 근거가 없습니다."
            ],
            diagnostics=[
                "Declared environments: "
                + "; ".join(f"protocol={p}, context={c}" for p, c in sorted(signatures))
            ],
        )

    rows, parse_errors = _analysis_input(clusters)
    preview = _aggregate_preview(rows)
    if parse_errors:
        return Outcome(
            status="invalid_data",
            status_label="ANALYSIS STOPPED",
            title="Some selected data cannot be interpreted",
            summary="필수 열 또는 값 형식을 확인한 뒤 다시 분석해야 합니다.",
            preview=preview,
            assumptions=[],
            limits=["유효하지 않은 행을 임의로 보정하거나 추정하지 않았습니다."],
            diagnostics=parse_errors,
        )

    comparable = [item for item in preview if item["winner"] is not None]
    if len(comparable) < 2:
        return Outcome(
            status="insufficient_information",
            status_label="INSUFFICIENT INFORMATION",
            title="At least two comparable N values are required",
            summary=(
                "같은 N에서 두 알고리즘의 runtime이 모두 측정된 지점이 최소 두 개 필요합니다."
            ),
            preview=preview,
            assumptions=["각 비교 지점에는 trial과 sieve 측정이 모두 존재해야 합니다."],
            limits=["현재 데이터만으로 성능 우위 전환 구간을 만들 수 없습니다."],
            diagnostics=[],
        )

    ties = [item["N"] for item in comparable if item["winner"] == "tie"]
    if ties:
        return Outcome(
            status="insufficient_information",
            status_label="INSUFFICIENT INFORMATION",
            title="An observed comparison is tied",
            summary=(
                "동일한 median runtime이 관측된 N이 있어 binary winner로 처리하지 않았습니다."
            ),
            preview=preview,
            assumptions=["Single Boundary Module은 각 비교 지점의 우위가 결정되어 있어야 합니다."],
            limits=["tie를 임의의 알고리즘 승리로 바꾸지 않았습니다."],
            diagnostics=[f"Tied N: {', '.join(map(str, ties))}"],
        )

    ordered = sorted(comparable, key=lambda item: int(item["N"]))
    transitions: list[int] = []
    for idx in range(1, len(ordered)):
        if ordered[idx - 1]["winner"] != ordered[idx]["winner"]:
            transitions.append(idx)

    if len(transitions) > 1:
        sequence = " → ".join(f"{item['N']}:{item['winner']}" for item in ordered)
        return Outcome(
            status="assumption_failed",
            status_label="ANALYSIS STOPPED",
            title="Single-crossover assumption conflicts with the selected data",
            summary=(
                "성능 우위가 두 번 이상 뒤집혀 하나의 경계만 존재한다는 가정 아래에서 "
                "결과나 다음 실험을 계산하지 않았습니다."
            ),
            preview=preview,
            assumptions=["탐색 구간에서 성능 우위는 최대 한 번만 전환된다."],
            limits=[
                "현재 Module로 하나의 crossover boundary를 정당하게 결정할 수 없습니다.",
                "현재 Module로 다음 N을 제안하지 않습니다.",
            ],
            diagnostics=[f"Observed winner sequence: {sequence}"],
        )

    if len(transitions) == 0:
        winner = ordered[0]["winner"]
        return Outcome(
            status="insufficient_information",
            status_label="INSUFFICIENT INFORMATION",
            title="No crossover is bracketed by the selected data",
            summary=(
                f"선택된 비교 지점에서는 모두 {winner}가 더 빨랐습니다. "
                "현재 관측 범위만으로 경계가 존재하는 위치를 정할 수 없습니다."
            ),
            preview=preview,
            assumptions=["탐색 구간에서 성능 우위가 한 번 전환될 수 있다."],
            limits=[
                "관측 범위 밖의 어느 N을 다음에 선택해야 하는지는 별도의 탐색 Domain 없이는 정당하게 결정되지 않습니다."
            ],
            diagnostics=[],
        )

    idx = transitions[0]
    left = ordered[idx - 1]
    right = ordered[idx]
    lower = int(left["N"])
    upper = int(right["N"])
    left_winner = str(left["winner"])
    right_winner = str(right["winner"])

    midpoint = (lower + upper) // 2
    proposal: dict[str, Any] | None = None
    if lower < midpoint < upper:
        proposal = {
            "type": "next_observation",
            "input": {"N": midpoint},
            "reason": (
                "현재 관측된 crossover bracket의 정수 중점을 측정하면 "
                "Single Boundary 가정 아래에서 후보 구간의 한쪽을 제거할 수 있습니다."
            ),
            "repeatInstruction": "현재 선택한 군집과 동일한 Protocol/Context로 측정하세요.",
            "commandTemplate": (
                f"python -m v4_mvp.benchmark_prime --n {midpoint} "
                "--repeats <protocol에_정한_반복횟수> --out next.csv"
            ),
        }

    return Outcome(
        status="ok",
        status_label="RESULT",
        title="A single performance crossover is bracketed",
        summary=(
            f"현재 선택한 데이터와 Single Boundary 가정 아래에서 성능 우위 전환은 "
            f"{lower} < N* ≤ {upper} 범위에 있습니다. "
            f"왼쪽에서는 {left_winner}, 오른쪽에서는 {right_winner}가 더 빨랐습니다."
        ),
        preview=preview,
        assumptions=[
            "선택된 군집의 Protocol/Context 표기가 동일하다.",
            "탐색 구간에서 성능 우위가 한 번만 전환된다.",
            "각 N의 비교에는 관측된 runtime의 median을 사용한다.",
        ],
        limits=[
            "runtime의 sampling uncertainty를 확률적 confidence로 환산하지 않았습니다.",
            "결론은 선택된 구현과 실행환경에 한정됩니다.",
            "관측하지 않은 실행환경에 대한 일반적인 성능 우위는 결정하지 않습니다.",
        ],
        diagnostics=[
            f"Selected clusters: {len(clusters)}",
            f"Comparable N values: {len(ordered)}",
            f"Module version: {MODULE_VERSION}",
        ],
        proposal=proposal,
    )


def analyze_general_prime_speed(clusters: list[dict[str, Any]]) -> Outcome:
    rows, parse_errors = _analysis_input(clusters)
    preview = _aggregate_preview(rows)
    if parse_errors:
        return Outcome(
            status="invalid_data",
            status_label="ANALYSIS STOPPED",
            title="Some selected data cannot be interpreted",
            summary="필수 열 또는 값 형식을 확인한 뒤 다시 분석해야 합니다.",
            preview=preview,
            assumptions=[],
            limits=["유효하지 않은 행을 임의로 보정하지 않았습니다."],
            diagnostics=parse_errors,
        )

    return Outcome(
        status="insufficient_scope",
        status_label="INSUFFICIENT INFORMATION",
        title="The requested information is broader than the declared data scope",
        summary=(
            "'일반적으로 어느 알고리즘이 더 빠른가'라는 질문은 대상 실행환경의 모집범위가 "
            "먼저 정의되어야 합니다. 한 Project 안의 benchmark 값이 존재한다는 사실만으로 "
            "그 범위를 임의로 만들어 결론내리지 않습니다."
        ),
        preview=preview,
        assumptions=[],
        limits=[
            "현재 Question에서 'general'이 의미하는 CPU, OS, Python 구현 및 기타 실행환경의 범위가 정의되지 않았습니다.",
            "따라서 현재 데이터만으로 일반적인 우위를 정당하게 도출할 수 없습니다.",
        ],
        diagnostics=[
            "Next step: 먼저 추론하려는 execution-environment domain을 명시한 뒤 필요한 관측 설계를 별도로 정의해야 합니다."
        ],
        proposal=None,
    )


def analyze_question(question_id: str, clusters: list[dict[str, Any]]) -> dict[str, Any]:
    get_question(question_id)
    if question_id == "prime_crossover":
        return analyze_prime_crossover(clusters).to_dict()
    if question_id == "general_prime_speed":
        return analyze_general_prime_speed(clusters).to_dict()
    raise ValueError(f"No module implementation for question: {question_id}")
