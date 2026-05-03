# Leesin Deployment Guide

Leesin은 실험 목적(Experiment Goal)을 최우선 분류 기준으로 삼는 데이터 품질 인증 시스템입니다.

- CSV 파일 하나 = 하나의 데이터 군집(cluster)
- CSV 내부 row = 해당 군집의 반복 관측값
- 각 CSV는 axis별 대표값(mean)으로 하나의 cluster vector가 됨
- Peer Group은 반드시 같은 `goalId`와 같은 Axis 구성의 저장 군집만 사용
- 이질성(`heterogeneity`)과 신뢰도(`confidence`)는 분리 계산

## 1. Before You Push

- `Lee_sin.venv/`, `.vscode/`, runtime JSON store는 GitHub에 올리지 마세요.
- `goal_store.json`과 `data_cluster_store.json`은 런타임 데이터입니다.
- Remote admin writes require `ADMIN_TOKEN` on Render. Keep `ALLOW_REMOTE_ADMIN=false`.
- `USE_DEMO_PEER_GROUP` 기본값은 `false`입니다.

## 2. Demo Peer Group Policy

Demo Peer Group은 개발/시연용 기준망입니다.

- 기본값: `USE_DEMO_PEER_GROUP=false`
- 실제 인증용 분석에는 기본적으로 포함되지 않음
- 설정/리포트 meta에 demo peer group 포함 여부가 표시됨
- 시연이 필요할 때만 환경변수 `USE_DEMO_PEER_GROUP=true`로 켜세요.

## 3. Stored Cluster Schema

저장되는 `ClusterRecord`는 원본 CSV가 아니라 비식별 numeric axis vector입니다.

- core: `id`, `goalId`, `goalName`, `axisNames`, `axisSignature`, `peerGroupKey`, `values`
- Welford summary: `valuesMean`, `valuesVariance`, `valuesStd`, `rowCount`, `summaryMethod`
- audit: `createdAt`, `uploadedAt`, `sourceBatchId`, `fingerprint`, `storagePolicy`
- upload snapshot: `analysisAtUpload`와 flat aliases such as `peerGroupSizeAtUpload`, `confidenceAtUpload`, `D2AtUpload`

기존 `data_cluster_store.json`은 backward compatible하게 normalize됩니다. 누락된 새 필드는 `None` 또는 안전한 기본값으로 채워집니다.

저장하지 않는 항목:

- 원본 CSV 파일
- 파일명
- unmapped column
- 개인정보 column

## 4. Batch Upload Policy

Batch Upload는 여러 CSV/TSV/TXT 파일을 동시에 선택할 수 있게 합니다.

- 각 파일은 독립적인 하나의 cluster로 처리
- Batch preview에서 row count, axis mapping, cluster vector, fingerprint, duplicate 여부, save 가능 여부, 분석 요약을 표시
- 사용자가 체크한 cluster만 저장
- 저장 record에는 `sourceBatchId`가 추가됨
- 이번 batch 안의 cluster들은 서로의 분석 기준에 포함되지 않음
- 저장 완료 후 다음 분석부터 Peer Group에 포함됨

중복 방지는 기존 fingerprint 정책을 유지합니다. 같은 `goalId`, `peerGroupKey`, `axisNames`, `values`, `rowCount`, `summaryMethod`이면 중복 저장하지 않습니다.

## 5. Audit, Deletion, Reevaluation

Cluster audit snapshot은 업로드 당시 결과와 현재 기준 재평가를 분리해 보존합니다.

- 업로드 당시: `analysisAtUpload`
- 현재 재평가: 현재 누적 Peer Group 기준
- 자기 자신을 target으로 재평가할 때는 `exclude_cluster_id`로 자기 자신을 Peer Group에서 제외

삭제 전 영향 분석은 leave-one-out 방식으로 계산합니다.

- Peer Group N 변화
- `Δconfidence`, `Δcoverage`, `Δequitability`, `Δheterogeneity`, `ΔD2`
- `Δcenter_norm = ||center_all - center_without_cluster||`
- `bin_uniqueness`

## 6. Analysis Extensions

- 기존 contribution 계산은 유지됩니다: `diff * (covariance_inv @ diff)` 기반 percent contribution.
- Axis Ablation Sensitivity가 추가되었습니다. 각 axis를 하나씩 제외하고 재분석하여 `delta_heterogeneity`, `delta_confidence`, `delta_D2`를 표시합니다.
- p=1 또는 sample 부족 시 `insufficient dimension/sample`로 안전하게 처리합니다.
- Target 또는 peer 값이 Domain Range 밖이면 out-of-domain warning을 표시합니다. Bin 계산의 clipping 정책은 유지하되, clipping 사실과 axis/value/domain/source를 리포트에 노출합니다.

## 7. Report Export

분석 결과는 다음 형식으로 export할 수 있습니다.

- JSON: 전체 report 구조 보존
- CSV summary: 핵심 수치와 JSON-encoded detail fields
- HTML: standalone report dump

Export에는 goal, axis/domain/resolution, target vector, peer group size, engine, Mardia 결과, D2, p-value, heterogeneity, confidence, coverage/equitability/sample size, contributions, axis ablation, out-of-domain warnings, confidence reasons, summary, timestamp, demo peer 포함 여부가 포함됩니다.

## 8. Goal Editor Bin Preview

Goal editor는 숫자 입력과 slider를 함께 제공합니다.

- `domainMin`, `domainMax`, `resolution` 수정
- slider step은 resolution 단위
- axis별 bin count, occupied bins, coverage 표시
- 전체 multidimensional bins, occupied bins, estimated coverage 표시
- `total bins > 100000`이면 resolution 과세분화 경고 표시

## 9. Optimizations

- `app.build_cluster_vector`: CSV row를 Welford algorithm으로 streaming mean/variance/std 계산
- `stats_engine.BinGridTracker`: hashmap 기반 multidimensional bin tracking
- `stats_engine.regularized_sscm_inverse`: Sherman-Morrison update로 SSCM inverse 계산
- `stats_engine.DataQualityAnalyzer._compute_heterogeneity`: covariance/SSCM interaction을 반영한 contribution 유지

## 10. API

기존 `/api/analyze`는 유지됩니다.

- `POST /api/analyze-batch`
- `POST /api/admin/clusters/batch-save`
- `POST /api/admin/clusters/impact`
- `POST /api/admin/clusters/reevaluate`
- `POST /api/export/report`

## 11. Deploy On Render

This repo includes `render.yaml`.

- `runtime: python`
- `buildCommand: pip install -r requirements.txt`
- `startCommand: python Leesin.py --host 0.0.0.0`
- `healthCheckPath: /health`
- `LEESIN_STORE_DIR=/var/data`
- persistent disk mounted at `/var/data`

If `DATABASE_URL` is set, Leesin stores Goal/Cluster payloads in PostgreSQL via `psycopg2`; otherwise it writes JSON files under `LEESIN_STORE_DIR`.

## 12. Local Run

```powershell
.\run_app.ps1
```

Run tests:

```powershell
.\Lee_sin.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
