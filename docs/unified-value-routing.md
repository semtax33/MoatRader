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
| 일반 사업회사 | `ECONOMIC_FCFF` | `CommonEconomicFcffEngine` |
| 강한 경기순환 | `NORMALIZED_FCFF` | `CommonEconomicFcffEngine` + 명시적 normalized assumptions |
| 금융 | `RIM` | `CommonRimEngine` |
| 바이오/파이프라인 | `RNPV` | `CommonRnpvEngine` |
| 적자 성장기업 | `SCENARIO_DCF` | `ScenarioDcfEngine` |
| 구조적 레버리지 변화 | `APV` | `ApvEngine` |
| 자산가치 중심 | `NAV` | `NavEngine` |
| 복합/지주회사 | `SOTP` | `SotpEngine` |

`RoutedValuationExecutor`는 route와 입력 method, issuer, as-of를 모두 대조합니다.
전용 입력이 없거나 다른 method 입력이면 valuation을 만들지 않습니다. 다른 FCFF로
fallback하지 않습니다.

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

통과한 신호만 다음 기준집단 안에서 정규화합니다.

```text
reference_class = valuation_method + economic_archetype
unified_value_score = percentile(raw_expectation_gap | reference_class)
```

Raw FCFF gap과 raw rNPV gap은 직접 섞어 순위화하지 않습니다.

## 감사 산출물

`scripts/audit_expanded_valuation_signals.py` schema v2는 다음을 기록합니다.

- route별 종목 수, valuation 생성 수, rank-eligible 수와 trusted 비율
- route label과 별도의 `actual_engine`
- `valuation_input_source`, input provenance, execution status
- reference class와 `unified_value_score`
- `fallback_fcff_count=0`, `llm_call_count=0`

과거 v7.1 historical FCFF 결과는 이 routed framework의 성과검증이 아닙니다. 기존
기간은 architecture/calibration 용도로만 취급하고, 규칙을 freeze한 뒤 미래 OOS에서
PER+PBR과 비교해야 합니다.
