# Leesin Deployment Guide

Leesin은 Experiment Goal별로 저장된 실제 데이터 군집을 Peer Group으로 사용해 CSV 데이터 품질을 진단하는 앱입니다.

- CSV 파일 하나 = 하나의 데이터 군집(cluster)
- CSV 내부 row = 해당 군집의 반복 관측값
- 각 CSV는 axis별 대표값(mean)으로 요약되어 하나의 cluster vector가 됩니다.
- 단, Coverage와 Equitability는 cluster 대표값이 아니라 저장된 row-level bin occupancy summary 기준으로 계산됩니다.
- Peer Group은 같은 Experiment Goal과 호환되는 Axis 구성을 가진 저장 cluster record만 사용합니다.
- 시연용 peer 기준망은 제공하지 않습니다.
- 기존 Mardia test, Mahalanobis, Ledoit-Wolf, SSCM, spatial rank 기반 분석 로직은 유지됩니다.

## 1. Before You Push

- `Lee_sin.venv/`, `.vscode/`, runtime JSON store는 GitHub에 올리지 마세요.
- `goal_store.json`과 `data_cluster_store.json`은 런타임 데이터입니다.
- Remote admin writes require `ADMIN_TOKEN` on Render. Keep `ALLOW_REMOTE_ADMIN=false`.
- `SAVE_DATA_CLUSTERS=true`이면 분석/저장 과정에서 sanitized numeric cluster vector와 row-level bin count summary가 저장됩니다.

## 2. Peer Group Policy

Peer Group은 오직 저장된 실제 cluster record로 구성됩니다.

- Sample Size N = 저장된 cluster 개수입니다.
- Heterogeneity, engine selection, Mardia/Mahalanobis/SSCM/spatial rank 입력은 cluster 대표 vector 기준입니다.
- Coverage와 Equitability는 peer cluster들의 row-level bin occupancy 합산 기준입니다.
- 저장된 cluster가 없거나 수가 부족하면 분석은 제한되며 사용자에게 안내 메시지를 보여줍니다.
- 기본 goal template인 `DEFAULT_GOALS`는 유지되지만, 실제 peer 판단에는 저장 cluster만 쓰입니다.
- batch preview와 reevaluation, impact analysis도 저장 cluster만 기준으로 동작합니다.
- 같은 batch 안에서 새로 만든 cluster들은 업로드 당시 서로의 분석 기준에 포함되지 않습니다.
- batch save가 끝난 뒤 다음 분석부터 저장 cluster가 Peer Group 후보와 row-level coverage 후보에 포함됩니다.

## 3. Row-Level Bin Occupancy

CSV 저장 시 raw row 전체를 저장하지 않고, mapped numeric axis 값의 bin count summary만 저장합니다.

- `binOccupancy`: multidimensional bin key(JSON array string)별 observation count
- `axisBinOccupancy`: axis별 1D bin count
- `binOccupancyMeta`: valid multidimensional row count, invalid row count, out-of-domain row count, total rows

Bin index 계산은 `floor((value - domainMin) / resolution)`입니다. `value == domainMax`는 마지막 bin에 포함됩니다. NaN, 빈 문자열, 숫자 변환 불가 값은 invalid로 처리하고, domain 밖 값은 out-of-domain으로 처리합니다.

기존 legacy cluster에는 `binOccupancy`가 없을 수 있습니다. 이런 cluster는 heterogeneity peer로는 계속 사용할 수 있지만 Coverage/Equitability row-level 계산에서는 제외됩니다. 정확한 row-level coverage를 보려면 기존 cluster를 삭제하고 원본 CSV를 다시 업로드해야 할 수 있습니다.

Axis subset 분석에서는 privacy 정책상 raw row를 재투영할 수 없으므로, row-level coverage는 저장 당시 `axisSignature`가 현재 selected axes와 정확히 같은 cluster만 사용합니다. Axis 구성이 다른 cluster는 heterogeneity peer에는 사용 가능하지만 row-level coverage 계산에서는 제외됩니다.

## 4. Stored Cluster Schema

저장되는 `ClusterRecord`는 원본 CSV가 아니라 비식별 numeric axis vector와 bin count summary입니다.

- core: internal `id`, `goalId`, `goalName`, `axisNames`, `axisSignature`, internal peer key, `values`
- Welford summary: `valuesMean`, `valuesVariance`, `valuesStd`, `rowCount`, `summaryMethod`
- row-level summary: `binOccupancy`, `axisBinOccupancy`, `binOccupancyMeta`
- audit: `createdAt`, `uploadedAt`, `sourceBatchId`, internal fingerprint, `storagePolicy`
- upload snapshot: `analysisAtUpload` and aliases such as `peerGroupSizeAtUpload`, `confidenceAtUpload`, `coverageCAtUpload`, `equitabilityEAtUpload`

저장하지 않는 항목:

- 원본 CSV 파일
- 파일명
- unmapped column
- 개인정보 column
- raw row 전체

중복 방지는 기존 fingerprint 정책을 유지합니다. 같은 goal, axis, values, rowCount, summaryMethod 조합은 중복 저장하지 않습니다.

## 5. Saved Clusters UI

메인 화면에는 별도의 “저장된 군집” 섹션이 있습니다.

- Experiment Goal별 필터와 전체 보기 옵션 제공
- 최신순/오래된순 정렬과 검색 제공
- “저장된 군집 N개” 형태의 개수 표시
- 각 cluster는 카드형 grid로 표시
- 화면 표시명은 goal별 순번을 사용합니다. 예: `진공 유지 품질 인증 - 군집 1`
- 기본 화면에서는 `cluster_xxx`, internal peer key, fingerprint, raw JSON을 표시하지 않습니다.
- 카드에는 goal name, axis 목록, axis 대표값, rowCount, row-level valid count, occupied bin count, 업로드 날짜, upload 당시 요약값을 표시합니다.
- legacy cluster는 row-level coverage에서 제외된다는 안내를 카드에 표시합니다.
- 삭제, 재평가, 영향 분석 액션을 카드에서 바로 실행할 수 있습니다.

## 6. Batch Upload Policy

Batch upload는 여러 CSV/TSV/TXT 파일을 동시에 preview하고 저장할 수 있게 합니다.

- 각 파일은 독립적인 하나의 cluster로 처리됩니다.
- Batch preview에서는 row count, axis mapping, cluster vector, duplicate 여부, save 가능 여부, 분석 요약을 보여줍니다.
- 사용자가 체크한 cluster만 저장합니다.
- 같은 batch 안의 cluster들은 업로드 당시 서로의 분석 기준에 포함되지 않습니다.
- 저장 완료 후 다음 분석부터 Peer Group과 row-level coverage 계산에 포함됩니다.

## 7. Audit, Deletion, Reevaluation

저장된 군집 카드는 다음 액션을 제공합니다.

- 삭제: 확인 후 삭제하며 목록과 count를 자동 갱신합니다.
- 재평가: 업로드 당시 snapshot과 현재 Peer Group 기준 결과를 비교합니다.
- 영향 분석: 삭제 전후의 Peer Group N, confidence, coverage, equitability, heterogeneity, D2 변화를 계산합니다.

결과는 기본적으로 숫자를 보기 좋게 반올림해서 표시하고 raw JSON 덤프를 보여주지 않습니다.

## 8. UI/UX Improvements

화면은 상단 quick navigation, 분석/업로드, batch preview, 저장된 군집, 리포트, 설정/관리 섹션으로 나뉩니다. 저장된 군집은 responsive card grid로 표시됩니다.

주요 비동기 버튼에는 loading state가 적용되어 있습니다.

- 분석 실행
- batch preview
- batch save
- goal 저장/삭제
- cluster 삭제
- impact analysis
- reevaluate
- report export

요청 중에는 버튼이 비활성화되고 spinner와 진행 문구가 표시됩니다. 성공/실패 후에는 버튼 상태가 복구되어 중복 요청을 막습니다.

## 9. Analysis Extensions

- Contribution 계산은 `diff * (covariance_inv @ diff)` 기반 percent contribution을 유지합니다.
- Axis Ablation Sensitivity는 각 axis를 제외하고 재계산한 delta heterogeneity, confidence, D2를 표시합니다.
- p=1 또는 sample 부족은 `insufficient dimension/sample`로 안전하게 처리합니다.
- Target 또는 peer 값이 Domain Range 밖이면 out-of-domain warning을 표시합니다.

## 10. Report Export

분석 결과는 다음 형식으로 export할 수 있습니다.

- JSON: 전체 report 구조 보존
- CSV summary: 핵심 수치와 JSON-encoded detail fields
- HTML: standalone report dump

Export에는 goal, axis/domain/resolution, target vector, peer group size, engine, Mardia 결과, D2, p-value, heterogeneity, confidence, row-level coverage/equitability/sample size, contributions, axis ablation, out-of-domain warnings, confidence reasons, summary, timestamp가 포함됩니다.

## 11. Goal Editor Bin Preview

Goal editor는 숫자 입력과 slider를 함께 제공합니다.

- `domainMin`, `domainMax`, `resolution` 수정
- slider step은 resolution 단위
- axis별 bin count, occupied bins, row-level coverage 표시
- 전체 multidimensional bins, occupied bins, row-level coverage 표시
- coverage eligible clusters, excluded legacy clusters, row-level observations 표시
- `total bins > 100000`이면 resolution 과세분화 경고 표시

## 12. API

기존 `/api/analyze`는 유지됩니다.

- `POST /api/analyze-batch`
- `POST /api/admin/clusters/batch-save`
- `POST /api/admin/clusters/impact`
- `POST /api/admin/clusters/reevaluate`
- `POST /api/admin/clusters/delete`
- `POST /api/export/report`

Analysis payload에는 다음 coverage meta가 포함됩니다.

- `coverageBasis=row_level_bin_occupancy`
- `coverageEligibleClusterCount`
- `coverageLegacyExcludedClusterCount`
- `coverageAxisSignatureExcludedClusterCount`
- `rowLevelObservationCount`
- `occupiedBins`
- `totalBins`

## 13. Deploy On Render

This repo includes `render.yaml`.

- `runtime: python`
- `buildCommand: pip install -r requirements.txt`
- `startCommand: python Leesin.py --host 0.0.0.0`
- `healthCheckPath: /health`
- `LEESIN_STORE_DIR=/var/data`
- persistent disk mounted at `/var/data`

If `DATABASE_URL` is set, Leesin stores Goal/Cluster payloads in PostgreSQL via `psycopg2`; otherwise it writes JSON files under `LEESIN_STORE_DIR`.

## 14. Local Run

```powershell
.\run_app.ps1
```

Run tests:

```powershell
.\Lee_sin.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

## 15. Peer Group Debug Checklist

Use this when saved clusters exist but analysis reports a very small Peer Group N.

- Create a Goal with 4 axes.
- Batch upload and save 6 CSV files for that Goal and those 4 axes.
- Confirm the batch-save response reports `compatiblePeerCountForSelectedAxes=6`.
- Analyze a new target CSV and confirm `peer_group_size=6`.
- Confirm Coverage meta reports row-level observation counts from saved cluster `binOccupancy`.
- Change selected axis order and confirm Peer Group N remains stable.
- If coverage eligible count is lower than Peer Group N, check whether some clusters are legacy records or have a different `axisSignature`.
- If the count is unexpected, inspect `Peer filter diagnostics` in the analysis error message: it includes total clusters, same-goal clusters, compatible-axis clusters, selected axis keys, and excluded examples.
