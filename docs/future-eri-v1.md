# FCFF Evidence → Future ERI V1

역사 공시 원천, LLM/human 품질 게이트, feature 봉인, ERI outcome, clustered 통계 및
downstream Value-neutral 단계는
[historical-future-eri-research-pipeline.md](historical-future-eri-research-pipeline.md)에 정리되어 있다.

## 목적

V1은 일반 FCFF 영업기업에서 오늘 확인된 사업 변화가 63거래일 뒤 시장의 미래 기대수정을 예측하는지 검증한다.

```text
Evidence_t → Future ERI_t+63
```

주가수익률을 직접 예측하거나 검증하지 않는다. PER+PBR 및 다른 Value 지표도 primary ranking에 사용하지 않는다.

## 사전 고정 계약

Evidence는 다음 6축만 사용한다.

1. `DEMAND`
2. `PRICE_MIX`
3. `BACKLOG`
4. `MARGIN`
5. `INVENTORY_MISMATCH`
6. `CAPACITY_CAPEX`

각 축은 현재 공시와 직전 비교 가능 공시의 상태 차이를 `-1`, `0`, `+1`로 자른다. Primary Evidence F-score는 여섯 방향값의 동일가중 합이다. Materiality는 `min(numerator / denominator, 1)`로 제한해 별도 진단 점수에 적용하며, 산식 basis와 분자·분모 source ID를 함께 보존한다. 한 축이라도 없으면 중립값 0으로 대체하지 않고 해당 행을 제외한다.

LLM을 사용할 경우 역할은 원문 span을 `WEAKENING`, `STABLE`, `IMPROVING`으로 분류하는 데 한정한다. 모든 관측치는 `source_span`, `statement_type`, `classification_rule_id`, `materiality_rule_id`, `confidence`, `source_published_at`, `available_at`, `signal_timestamp`를 보존한다. LLM 전망 예측은 허용하지 않는다.

## PIT 규칙

- 모든 시각은 timezone-aware여야 한다.
- `source_published_at <= available_at <= signal_timestamp`를 강제한다.
- Reverse DCF의 시장가격 timestamp는 signal timestamp와 같아야 하며 가격 source ID를 남긴다.
- 장 마감 뒤 공개된 자료는 다음 거래 세션 시가 전에는 사용할 수 없다.
- 거래일은 외부에서 명시적으로 전달한 세션 달력을 사용한다. 영업일을 거래일로 임의 대체하지 않는다.
- Feature 파일을 먼저 기록하고 SHA-256으로 봉인한 다음에만 outcome 파일을 연다.

## ERI 라벨

Signal 시점 Reverse DCF의 기대 경로를 `frozen_expectation_assumptions`로 저장한다. 63거래일 뒤 counterfactual은 다음 항목만 당시 실현 상태로 바꾼다.

- base revenue
- base NOPAT margin
- invested capital
- net debt
- diluted shares
- WACC

Growth, target margin, ROIIC, stable state, reinvestment method는 signal 시점 값을 유지한다. Economic DCF의 CAP가 정수 연도이므로 V1은 signal date의 1년 anniversary가 완전히 지난 경우에만 CAP 한 해를 차감한다.

```text
Future ERI_63 = log(Actual Price_t+63 / Counterfactual FCFF Value_t+63)
```

실현 재무정보는 target price timestamp 이전에 공개된 것만 사용할 수 있다. Target price와 변경 WACC에도 각각 source ID를 요구한다.

## 현재 데이터 준비 상태

기존 v7.3 산출물은 재무 프록시와 다음 분기 implied-driver revision을 담고 있지만, 동일 기업의 연속 공시를 이 V1의 6축 상태로 분류한 이력은 담고 있지 않다. 따라서 기존 행의 누락 축을 0으로 채워 실측 결과를 만들지 않는다. 실제 단조성 결과를 내려면 먼저 과거 공시 쌍에 대해 V1 `EvidenceObservation` 입력을 생성하고, 당시 Reverse DCF 경로와 63거래일 outcome을 연결해야 한다.

## 메커니즘 gate

F-score band는 사전에 다음처럼 고정한다.

| Band | F-score |
|---|---:|
| Q1 | -6 ~ -3 |
| Q2 | -2 ~ -1 |
| Q3 | 0 |
| Q4 | +1 ~ +2 |
| Q5 | +3 ~ +6 |

다음 조건을 모두 만족해야 gate가 열린다.

- 모든 band가 최소 표본 수를 만족한다.
- Q1부터 Q5까지 인접 mean ERI가 모두 비감소한다.
- Q5-Q1 mean ERI가 양수다.
- F-score와 ERI의 Spearman 상관이 0 이상이다.

Gate가 실패하면 ML은 금지된다. Gate가 통과해도 V1 실행기는 미래수익률을 읽거나 검사하지 않는다. Return test는 별도 사전등록 단계로만 진행한다.

## 입력과 실행

Feature 입력은 JSON array 또는 JSONL이며 각 행에 아래 필드가 필요하다.

```text
observation_id
current_observations[]
prior_observations[]
expectation_state
frozen_expectation_assumptions
```

Outcome 입력은 `FutureEriOutcomeInputV1` 구조를 사용한다. 거래 세션 파일은 ISO date JSON array 또는 첫 열이 date인 텍스트/CSV다.

```powershell
python scripts/run_future_eri_v1.py `
  --feature-input data/features.jsonl `
  --outcome-input data/outcomes.jsonl `
  --trading-sessions data/krx-sessions.csv `
  --output data-lake/experiments/future-eri-v1
```

실행 결과에는 `features-pre-label.jsonl`, `feature-seal.json`, `future-eri-labels.jsonl`, `mechanism-report.json`, `FINAL-RESULT.json`, `build-manifest.json`이 생성된다.
