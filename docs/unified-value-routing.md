# Unified Economic Value routing

Unified Value는 하나의 valuation model이 아닙니다. 가격을 보기 전에 경제구조에 맞는
모델을 선택하고, 모델별 intrinsic value gap을 같은 reference class 안에서
percentile로 변환하는 출력 계약입니다.

```text
PIT profile -> route -> method-specific assumptions -> actual engine
            -> trusted valuation -> gap -> route/archetype percentile
```

## Route와 실제 엔진

| 경제구조 | Route | 실행 엔진 |
| --- | --- | --- |
| 일반 사업회사 | `ECONOMIC_FCFF` | `CommonEconomicFcffEngine` 또는 감사된 `LegacyFcffCommonEngine` 호환 경로 |
| 강한 경기순환 | `NORMALIZED_FCFF` | `NormalizedFcffEngine` |
| 금융 | `RIM` | `CommonRimEngine` |
| 바이오/파이프라인 | `RNPV` | `CommonRnpvEngine` |
| 적자 성장기업 | `SCENARIO_DCF` | `ScenarioDcfEngine` |
| 구조적 레버리지 변화 | `APV` | `ApvEngine` |
| 자산가치 중심 | `NAV` | `NavEngine` |
| 복합/지주회사 | `SOTP` | `SotpEngine` |

`RoutedValuationExecutor`는 route와 입력 method, issuer, as-of를 모두 대조합니다.
전용 입력이 없거나 다른 method 입력이면 valuation을 만들지 않습니다. 다른 FCFF로
fallback하지 않습니다.

## 모델별 deterministic builder

- Normalized FCFF는 최소 5년(기본 7년)의 price-free 이력을 요구하고 winsorized median,
  포함·제외 회계연도, cycle phase를 입력에 고정합니다. 구조적 경기순환 기업은 당기
  EBIT가 음수여도 route를 유지합니다.
- Scenario DCF는 비경기순환·지속 적자이면서 양(+)의 unit economics 경로가 있는 경우만
  허용합니다. downside/base/upside를 모두 내되 순위에는 base만 사용하고 25/50/25는
  진단용 고정 weight입니다.
- rNPV는 파이프라인, 임상 단계·상태, 소유권을 서로 다른 evidence role로 검증하고
  point-in-time source date와 frozen phase-POS 정책을 저장합니다. Sponsor는 소유권
  증거로 간주하지 않습니다.
- RIM은 동일 owner-of-parent scope의 전기 FY, 당기 YTD, 전년 동기 YTD를 사용해
  `TTM = prior FY + current YTD - prior YTD`로 구축합니다.
- SOTP는 외부 계산값을 단순 합산하지 않습니다. 각 part의 실제 sub-model을 실행하고
  valuation basis, ownership 적용 여부, cash-flow/net-debt/NCI scope, input SHA-256와
  실행 상태를 강제합니다. 중복 scope나 재귀 SOTP는 오류로 닫습니다.

## 입력 계약

감사 스크립트의 `--valuation-input-root`에는
`<root>/<YYYY-MM-DD>/<ticker>.json`을 둡니다.

```json
{
  "schema_version": "routed-valuation-input/1",
  "issuer_id": "000001",
  "as_of": "2026-05-31",
  "method": "RIM",
  "assumptions": {},
  "source_refs": ["PIT:DART:2026-05-31:000001"]
}
```

`assumptions`는 method별 Pydantic 모델로 다시 검증됩니다. 현재 가격, forward return,
LLM 생성값은 이 계약에 들어갈 수 없습니다. Routing·valuation·normalization은 전부
결정론적 Python이며 LLM 호출 수는 0입니다.

Snapshot heuristic으로 표현할 수 없는 구조적 leverage/APV 같은 route는
`--valuation-profile-root`에 같은 날짜/티커 구조로 price-free `ValuationProfile` JSON을
제공합니다. Profile의 issuer/as-of/archetype/구조 flag가 맞지 않으면 감사 실행이
중단됩니다. 명시적 profile이 잘못됐을 때 heuristic profile로 fallback하지 않습니다.

## Trusted와 Unified score

`CheapSignal.from_valuation()`은 다음 중 하나라도 실패하면 fair value를 설명용으로
남기되 `rank_eligible=false`, `unified_value_score=null`로 처리합니다.

- model applicability가 `ELIGIBLE`이 아님
- fair value가 0 이하
- frozen screening 실패
- assumption confidence가 정책 하한 미만 또는 누락
- warning 수가 정책 상한 초과

`valuation-trust/2`에서는 모든 종목에 공통으로 붙는 가정 한계 안내문을
`disclosures`로 보존하고, 종목별 valuation 실패 가능성을 나타내는 `warnings`만
상한 3에 집계합니다. 상한·confidence·screening 기준은 낮추지 않았습니다.

통과한 신호만 `unified-value-normalization/2`의 return-blind hierarchy로
정규화합니다.

```text
valuation_method + economic_archetype
    -> valuation_method
    -> economic model family
raw_value_gap = supported_intrinsic_value / market_price - 1
unified_value_score = percentile(raw_value_gap | reference_class)
```

Model family는 수익률을 보지 않고 사전 정의합니다. FCFF·Normalized FCFF·Scenario
DCF·APV는 `OPERATING_CASH_FLOW`, RIM은 `RESIDUAL_INCOME`, rNPV는
`PIPELINE_PROBABILITY_WEIGHTED`, NAV·SOTP는 `ASSET_AND_SUM_OF_PARTS`입니다.
서로 다른 model family의 raw gap은 절대 합치지 않습니다.
`raw_value_gap`은 market-implied driver를 역산하는 Expectation Gap과 별개입니다.
각 단계의 trusted reference class가 20개 미만이면 다음 단계로 이동하고, model
family도 20개 미만이면 score를 만들지 않습니다. `reference_class_size`와 세 단계별
표본 수, 실제 fallback level을 모두 산출물에 기록합니다.

## 감사 산출물

`scripts/audit_expanded_valuation_signals.py` schema v5는 다음을 기록합니다.

- route별 종목 수, valuation 생성 수, rank-eligible 수와 trusted 비율
- route label과 별도의 `actual_engine`
- `valuation_input_source`, input provenance, execution status
- reference class와 `unified_value_score`
- `fallback_fcff_count=0`, `llm_call_count=0`
- 직전 route, route change reason, 전체·직전 method별 route stability
- `execution_rate=generated/eligible`, `trusted_generated_share=trusted/generated`,
  `score_coverage=ranked/routed`를 서로 분리한 coverage
- `INVALID/UNTRUSTED`의 status reason과 valuation warning 원문별 건수
- 설명용 disclosure와 trust warning의 별도 개수
- method+archetype, method, model-family 표본 수와 실제 normalization level

Architecture gate는 eligible route 전부의 실제 실행, 100% route-engine 호환,
fallback/LLM 0건, route stability 90% 이상을 요구합니다. Trusted reference class가
20개 미만이라 점수를 만들지 않는 것은 실행 실패가 아니라 의도된 `UNRANKABLE`입니다.

과거 v7.1 historical FCFF 결과는 이 routed framework의 성과검증이 아닙니다. 기존
기간은 architecture/calibration 용도로만 취급하고, 규칙을 freeze한 뒤 미래 OOS에서
PER+PBR과 비교해야 합니다.

PER+PBR은 현재 비교 baseline일 뿐 우선 랭킹으로 전환하지 않습니다. 이번 계약 변경은
기존 primary ranking policy를 변경하지 않습니다.
