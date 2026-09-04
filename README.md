# Leesin

**Leesin: From Data to Information**

Leesin은 실험 데이터에서 곧바로 결론을 만들어내기보다,

> **현재 데이터로 무엇을 알 수 있는가? 그리고 원하는 정보를 얻기 위해 무엇을 더 관측해야 하는가?**

를 명시적으로 다루기 위해 만든 연구용 프로토타입입니다.

현재 구현의 중심은 **Leesin_V4**입니다. V1~V3 코드는 이전 설계와 탐구 과정을 보존하기 위해 저장소에 함께 남겨두었습니다.

---

## 한눈에 보기

V4는 분석을 실험의 끝이 아니라 다음 관측으로 이어질 수 있는 한 단계로 봅니다.

```text
Experiment → Data → Analysis → Information → Experiment → ...
```

프로그램 안의 Core 흐름은 다음과 같습니다.

```text
Data → Module → Mapping → Analysis → Result → Next
```

화면은 세 영역으로 나뉩니다.

```text
Projects / Files | Core workflow | Modules
```

- **Projects / Files**: 원본 파일, Data, Analysis, Proposal, Trash 관리
- **Core**: 현재 분석 Cycle의 실행 흐름
- **Modules**: 재사용 가능한 분석 방법 탐색·선택·제작

`Question`은 사용자가 분석 목적을 기록하기 위한 선택적 문맥입니다. 실제 계산 방법, Input, Assumption, Limit은 Analysis Module이 가집니다.

---

## 빠르게 실행하기

Python 환경에서 저장소 루트 기준:

```bash
python -m v4_mvp.app
```

브라우저에서:

```text
http://127.0.0.1:8765
```

UI 파일을 수정하거나 새 코드를 pull한 뒤에는 Python 프로세스를 다시 시작하고 브라우저에서 `Ctrl+F5`를 권장합니다.

> V4는 **로컬 웹 서비스 프로토타입**입니다. Module Workshop의 Python 실행기는 제한된 실행 환경일 뿐 완전한 보안 sandbox가 아닙니다. 신뢰할 수 있는 로컬 코드만 실행해야 합니다.

---

## V4에서 할 수 있는 것

### 1. Project와 원본 Data 관리

- Project 생성·이름 변경·설명 수정·삭제
- 파일/폴더 생성, 이동, 다중 선택, Trash/Restore
- 업로드한 원본 bytes와 Leesin이 만든 파생 정보를 분리해 보존
- CSV/TSV를 Heat / Table / Raw 세 Lens로 확인

### 2. Analysis Module 사용

Module은 단순 Python 함수보다 넓은 단위입니다.

```text
Analysis Module
= Function
+ Input Contract
+ Assumptions
+ Output / Limit metadata
```

Module shelf에서는 Browse / My / Favorites, metadata 검색, Input compatibility hint, Use/drag-and-drop 등을 제공합니다.

### 3. Module Workshop

기존 Python 함수를 가능한 한 그대로 가져와 Module로 만들기 위한 흐름입니다.

```text
Paste / Drop Python
→ Paste / Drop Data
→ Auto-map
→ Run
→ Save
```

Workshop은 AST를 이용해 top-level 함수, parameter, default value, annotation, docstring을 읽고 Data column과의 Mapping을 제안합니다. 최종 Mapping은 사용자가 확인합니다.

### 4. Result 이후의 Next

Module이 추가 관측을 정당하게 제안할 수 있는 경우 `Next`를 반환할 수 있습니다. 첫 MVP에서는 prime algorithm benchmark의 crossover 범위를 좁히기 위한 다음 `N`을 제안하도록 구현했습니다.

---

## 범용성 확인에 사용한 작은 실험

V4의 목표는 모든 분석을 하나의 수식으로 통합하는 것이 아니라, **Core와 Analysis Module을 분리해 재사용할 수 있는지** 확인하는 것입니다.

### 같은 Module, 다른 실험

`Descriptive Summary`를 코드 수정 없이 다음 두 Data에 적용했습니다.

- Monte Carlo π: `abs_error → values`
- Prime benchmark: `runtime_ms → values`

### 같은 Core, 다른 Module

Monte Carlo π Project에서 `Pearson Correlation`을 실행했습니다.

```text
sample_size → x
abs_error   → y
```

결과는 `r = -0.4160325607005631`이었습니다.

이 실험들은 모든 연구에 대한 범용성을 증명하기 위한 것이 아니라, 현재 MVP에서 **Module 재사용성과 Core 분리 구조가 실제로 작동하는지** 확인하기 위한 것입니다.

---

## 저장소 구조

처음 보는 경우 아래처럼 읽는 것이 가장 쉽습니다.

```text
README.md                    ← 프로젝트 전체 설명
v4_mvp/
  README.md                  ← V4 코드 구조와 읽는 순서
  app.py                     ← 로컬 HTTP 서버/API 진입점
  store.py                   ← Project/Cluster/Analysis/Proposal 저장
  workspace_store.py         ← File/Folder/Trash 작업
  modules.py                 ← built-in 분석 로직
  module_workshop.py         ← Python 함수 검사·Mapping·실행·저장
  benchmark_prime.py         ← prime benchmark 자체
  mvp_adapters/              ← MVP 전용 실험 실행 adapter
  templates/index.html       ← V4 기본 화면
  *_ui.js                    ← 각 UI 기능
  runtime/                   ← 로컬 실행 상태(대부분 git에서 제외)
tests/
  test_v4_mvp.py
  test_module_workshop.py
  test_workspace_store.py

# 아래 루트 파일들은 주로 V1/V2 역사적 구현
app.py
stats_engine.py
storage.py
feasible_mask.py
feasible_box_counter.py
templates/index.html
```

V4를 공부할 목적이라면 **루트의 거대한 V1/V2 `app.py`부터 읽지 말고 `v4_mvp/README.md`의 순서대로 보는 것**을 권장합니다.

---

## Leesin이 V4까지 온 과정

### V1 — 이상치 점수와 신뢰도의 분리

저장된 peer data를 기준으로 Mahalanobis / SSCM / Spatial Rank 계열 엔진을 검토하고, 이상치 정도와 그 판단의 신뢰도를 분리하려 했습니다.

### V2 — row-level density와 실험 공간

군집 평균 vector 압축 문제를 발견한 뒤 row-level density 구조로 전환했습니다. Domain Range, Resolution, Grid Preview, Feasible Domain Mask, eCDF Specificity, Input/Output axis 등을 도입했습니다.

### V3 — 신뢰도 추론의 한계

Sample Size를 임의의 포화함수로 두지 않기 위해 Stability를 검토했습니다. Distribution Stability와 Rank Stability는 명시된 sampling assumption 아래에서는 의미가 있었지만, 현재 Data만으로 모든 상황의 신뢰도를 하나의 일반적인 값으로 결정할 수 없다는 한계가 드러났습니다.

### V4 — Data에서 Information으로

문제를 다음 두 질문으로 확장했습니다.

- 현재 Data로 어떤 Information을 정당하게 얻을 수 있는가?
- 부족하다면 무엇을 더 관측해야 하는가?

그 결과 분석법은 재사용 가능한 **Analysis Module**로, 공통 실행 과정은 **Core**로 분리했습니다.

---

## 코드 읽기 / 학습 순서

이 저장소를 나중에 직접 이해하려면 다음 순서를 권장합니다.

1. `v4_mvp/README.md` — 전체 구조와 각 파일 역할
2. `v4_mvp/app.py` — UI와 backend가 어떤 API로 연결되는지
3. `v4_mvp/store.py` — V4의 기본 데이터 모델과 영속화
4. `v4_mvp/workspace_store.py` — Project file explorer가 Core store 위에 어떻게 얹히는지
5. `v4_mvp/modules.py` — 실제 분석 결과/Assumption/Limit/Proposal이 어떻게 만들어지는지
6. `v4_mvp/module_workshop.py` — AST 검사와 restricted runner
7. `v4_mvp/mvp_adapters/prime_benchmark.py` — Core 밖의 실험 adapter
8. `v4_mvp/templates/index.html`과 `*_ui.js` — 브라우저 상태와 화면 동작

코드 안의 주석은 **무엇을 하는지 그대로 번역하기보다, 왜 이 경계가 존재하는지와 어떤 설계 결정을 보존하는지**를 설명하는 방향으로 유지합니다.

---

## 검증

GitHub Actions는 V4에 대해 다음 검사를 수행합니다.

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

Browser drag/drop, rollback/Undo, Data Lens와 Core flow 같은 시각적 상호작용은 별도의 수동 smoke test가 필요합니다.

---

## 의도적으로 구현 범위 밖에 둔 것

현재 V4는 public production service가 아닙니다. 다음 항목은 별도 보안·배포 설계가 필요한 후속 범위입니다.

- 실제 Google/GitHub OAuth
- server-side multi-user authorization
- hosted Module Registry
- GPT/Sites 기반 Module discovery
- dependency environment 관리
- 임의 사용자 Python을 위한 robust sandbox
- signed Module / trust policy
- generalized remote experiment adapter

Leesin_V4는 이 문제들을 불완전하게 숨기기보다, **로컬에서 전체 연구 workflow를 검증 가능한 상태까지 구현하는 것**을 현재 종착점으로 삼았습니다.
