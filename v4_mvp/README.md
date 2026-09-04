# Leesin_V4 — code guide

이 문서는 V4를 **직접 읽고 이해하기 위한 개발자용 지도**입니다. 프로젝트 전체 설명과 V1→V4 탐구 흐름은 루트 `README.md`를 먼저 보세요.

V4의 핵심 구조는 다음 두 줄입니다.

```text
Projects / Files | Core workflow | Modules
Data → Module → Mapping → Analysis → Result → Next
```

V4는 현재 Leesin의 주 구현입니다. 루트의 V1/V2 파일들은 개발 역사를 보존하기 위해 별도로 남아 있습니다.

---

## 가장 먼저 읽을 파일

처음부터 JavaScript 전체를 읽지 말고 아래 순서로 보는 것이 좋습니다.

### 1. `app.py` — HTTP/API 연결

로컬 HTTP 서버의 진입점입니다.

여기서 확인할 것:

- 어떤 URL이 어떤 backend 함수에 연결되는지
- Project / Module Workshop / Workspace / Analysis가 서로 어떻게 분리되어 있는지
- V4 Core가 특정 분석 수학을 직접 가지지 않고 다른 모듈에 위임하는 방식

`app.py`는 분석 알고리즘 파일이 아니라 **배선(wiring) 계층**에 가깝습니다.

### 2. `store.py` — Core 데이터 모델

Project, Cluster, Analysis, Proposal을 JSON에 저장합니다.

중요한 설계:

- 원본 Data와 Analysis 결과를 별도 객체로 저장
- 모든 정상/중단 Analysis를 기록 가능
- Proposal을 Analysis와 연결
- 파일 쓰기는 temporary file을 만든 뒤 `os.replace`하는 방식으로 교체

### 3. `workspace_store.py` — Project 파일 탐색기

`store.py`의 같은 상태 파일 위에 File / Folder / Trash 기능을 얹습니다.

이 파일이 `core_store._read_state()` 등을 직접 사용하는 이유는 현재 MVP가 **하나의 로컬 저장소를 공유**하기 때문입니다. production 구조라면 public repository/service interface로 분리하는 것이 더 적절합니다.

### 4. `modules.py` — built-in 분석 로직

첫 MVP의 Single Boundary와 정보 범위 진단 등이 있습니다.

읽을 때는 계산식보다 다음을 보세요.

- 어떤 Input을 요구하는가
- 어떤 Assumption을 검사하는가
- 어떤 경우 Result 대신 분석을 중단하는가
- 무엇을 Limit으로 반환하는가
- Next observation을 언제 제안하는가

V4의 핵심 철학이 가장 직접적으로 코드가 된 부분입니다.

### 5. `module_workshop.py` — 기존 Python 함수를 Module로 바꾸기

대략 다음 순서로 동작합니다.

```text
Python code
→ AST inspection
→ Data parsing
→ Mapping suggestion
→ restricted execution
→ Module save
```

중요: 이 파일의 AST validator와 subprocess timeout은 **위험을 줄이는 MVP 장치**이지 보안 sandbox가 아닙니다.

### 6. `benchmark_prime.py` + `mvp_adapters/prime_benchmark.py`

`benchmark_prime.py`는 계산 실험 자체이고, adapter는 그 실험을 Leesin의 Data Cluster 형식으로 연결합니다.

이 둘을 분리한 이유는 **실험 실행기가 Core의 영구 책임이라고 가정하지 않기 위해서**입니다.

### 7. UI 파일

- `templates/index.html`: 기본 화면 뼈대
- `workspace_ui.js`: Project workspace / file explorer
- `module_workshop_ui.js`: Module Workshop 화면
- `module_file_input_ui.js`: Workshop 파일 입력
- `project_controls_ui.js`: Project 설정
- `mvp_adapters/prime_ui.js`: prime MVP 전용 UI 연결
- `ux_polish_ui.js`: 최종 Core/Module shelf 및 여러 UI 보정

현재 UI JavaScript는 MVP를 빠르게 반복 개발한 흔적이 남아 있어 backend보다 읽기 어렵습니다. 기능이 안정된 뒤 component/module 단위로 다시 나누는 것이 장기적인 정리 방향입니다. **현재 제출본에서는 동작을 바꾸는 대규모 UI refactor보다 재현성을 우선합니다.**

---

## 데이터가 지나가는 길

### 일반 분석

```text
사용자가 Data 선택
→ Module 선택
→ Input ↔ column Mapping
→ Module 실행
→ Result / Assumptions / Limits / Diagnostics
→ 필요하면 Next
```

### Prime MVP의 다음 실험

```text
Single Boundary Result
→ Proposal(N)
→ prime MVP adapter
→ benchmark 실행
→ 새 Cluster 저장
→ 새 Cycle의 Data
```

`prime_benchmark.py`가 `mvp_adapters/` 안에 있는 이유가 여기 있습니다. Core는 Proposal을 다루지만, 특정 실험을 어떻게 실제로 수행할지는 adapter가 담당합니다.

---

## 저장 위치

기본 server-side 상태:

```text
v4_mvp/runtime/store.json
```

Workshop에서 저장한 custom Module:

```text
v4_mvp/runtime/custom_modules.json
```

환경변수로 각각 `LEESIN_V4_STORE`, `LEESIN_V4_MODULE_STORE`를 지정할 수 있습니다.

`runtime/`의 실제 실행 데이터는 Git에 포함하지 않습니다.

---

## 테스트

```bash
python -m unittest tests.test_v4_mvp tests.test_module_workshop tests.test_workspace_store
```

GitHub Actions에서는 여기에 Python compile과 각 JavaScript 파일의 `node --check`도 수행합니다.

테스트가 다루는 것:

- prime Single Boundary 분석
- Module Workshop 준비/실행
- workspace 저장 및 Trash 계열 동작

테스트가 충분히 다루지 못하는 것:

- 브라우저 drag/drop
- 실제 레이아웃
- Core rollback/Undo의 시각적 상호작용
- Data Lens 표현

이 부분은 수동 browser smoke test가 필요합니다.

---

## 코드를 정리할 때 지키는 기준

이 저장소는 결과물인 동시에 V1→V4 사고과정의 기록입니다. 따라서 정리는 다음 우선순위를 따릅니다.

1. **동작과 실험 재현성을 깨지 않는다.**
2. generated/cache/runtime 파일은 Git에서 제거한다.
3. 중복은 의미가 같고 테스트 가능한 경우에만 합친다.
4. 주석은 코드 자체를 번역하기보다 **왜 그렇게 설계했는지** 설명한다.
5. V1/V2의 역사적 구현을 V4 스타일로 억지로 다시 쓰지 않는다.
6. 대규모 UI refactor는 별도 작업으로 분리한다.

이 기준은 나중에 코드를 학습할 때도 중요합니다. 모든 낡은 부분을 즉시 없애기보다, **왜 그 구조가 생겼고 V4에서 무엇이 달라졌는지 비교하는 것**이 Leesin을 이해하는 데 더 도움이 됩니다.
