# Expectation Analysis 계약

## 목적

주 질문은 “MOAT 점수가 높은가?”가 아니라 다음입니다.

```text
PIT 증거와 경제구조가 지지하는 Possible/Plausible/Probable 미래
− 현재 가격이 요구하는 성장·마진·ROIIC·CAP 조합
= Expectation Gap
```

과거 `economic_moat_score`와 legacy FCFF는 회귀·sanity-check용 진단 결과입니다. 기본 랭킹에는 쓰이지 않습니다.

## 두 개의 격리된 lane

1. Intrinsic lane은 공시 cutoff, 재무 기준점, valuation-driver evidence, reference class, CAP prior와 downside/central/upside 가정만 사용합니다.
2. Economic DCF는 `성장 → 재투자 → ROIIC → FCFF`를 연결하고 CAP 뒤에 ROIIC·성장·마진을 안정상태로 fade시킵니다.
3. 3P는 가정을 생성하는 공식이 아니라 Possible(물리·수학 경계), Plausible(PIT base rate와 회사의 역사적 경제성), Probable(정성 evidence) 검증기입니다.
4. Intrinsic valuation과 3P가 완성된 뒤에만 `MarketPriceInput`을 만들고 Reverse DCF를 실행합니다.
5. Reverse DCF는 하나의 “정답”을 역산하지 않습니다. 현재 가격 허용오차 안에 들어오는 성장·마진·ROIIC·CAP 조합 전체와 각 축의 범위를 보존합니다.

`ExpectationAnalysisRequest`에는 가격 필드가 없고 알 수 없는 필드는 거부됩니다. 요청의 timezone-aware `evidence_cutoff`는 run의 `--as-of`와 정확히 같아야 하며, `evidence_cutoff > price_as_of`도 거부됩니다.

## ValuationDriverEvidence

동결된 Evidence Sensor v1을 바꾸지 않고 별도의 valuation-only atomic selector와 분류 요청을 사용합니다. `MOAT_NONE`이라도 backlog, CAPA, ASP, product mix, 승인·pipeline, CAPEX, 재투자, 위험에 관한 미래 사실이면 보존합니다.

각 evidence는 다음 중 하나의 primary driver만 가집니다.

- `REVENUE_GROWTH`
- `TARGET_MARGIN`
- `REINVESTMENT_EFFICIENCY`
- `ROIIC`
- `CAP_FADE`
- `RISK`

관련 driver는 진단 관계일 뿐 숫자 조정 권한이 없습니다. LLM은 원문 인용, driver/role, 방향, 짧은 근거만 분류하며 성장률·마진·확률·CAP 연수·공정가치를 생성하지 않습니다.

## 숫자 가정

Manifest의 `expectation_assumptions`는 price-blind `ExpectationAnalysisRequest` JSON을 가리킵니다. 예시는 [`../examples/sample-expectation-assumptions.json`](../examples/sample-expectation-assumptions.json)에 있습니다.

세 시나리오는 다음을 명시합니다.

- TTM 또는 PIT 기준 매출·NOPAT margin·invested capital
- 성장률과 target NOPAT margin
- ROIIC 방식 또는 sales-to-capital 방식의 재투자
- competitive advantage period와 fade 기간
- stable growth, stable margin, stable ROIC, WACC
- net debt, diluted shares, failure probability
- 각 가정의 evidence ID·source·사전 정의된 assumption type

선택적인 `capital_periods`와 `intangible_adjustment_policy`를 주면 R&D와 지정 비율의 SG&A를 자본화해 보고/경제적 ROIC와 ROIIC를 병렬로 계산합니다. 중앙 ROIIC 가정은 이 역사 범위와 비교되지만 자동으로 덮어쓰지 않습니다.

Confidence는 중앙가치에 곱하지 않습니다. 낮은 evidence confidence는 downside/upside 간 거리를 확대하여 범위만 넓힙니다.

## CAP와 3P

CAP는 timestamp와 source를 가진 외부 PIT reference-class prior에서 시작합니다. 회사별 primary CAP evidence, erosion, 관찰된 persistence는 bounded range를 보강하거나 단축합니다. MOAT scalar를 CAP 연수로 변환하지 않으며 같은 장점을 growth, margin, WACC, terminal growth에 중복 반영하지 않습니다.

3P 상태는 다음과 같습니다.

- Possible: `PASS` 또는 `FAIL`
- Plausible: `IN_RANGE`, `OUTLIER`, `UNKNOWN`
- Probable: `SUPPORTED`, `MIXED`, `WEAK`, `CONTRADICTED`
- 종합: `PASS`, `REVIEW`, `FAIL`

Probable은 과거 calibration이 없는 v1 정성 상태이며 숫자 확률로 가장하지 않습니다.

## 실행과 산출물

```powershell
moatrader analyze run `
  --universe path\to\universe.csv `
  --as-of 2026-08-14T15:30:00+09:00 `
  --run-id expectation-20260814

moatrader screen expectations `
  --run-dir data-lake\gold\runs\expectation-20260814
```

회사 디렉터리에는 다음 감사 artifact가 추가됩니다.

- `evidence-sensor-manifest.json`
- `valuation-atomic-evidence-units.jsonl`
- `valuation-driver-requests.jsonl`
- `valuation-driver-evidence.json`
- `competitive-advantage-profile.json`
- `cap-assessment.json`
- `capital-allocation-profile.json` (capital periods supplied when present)
- `intrinsic-valuation.json`
- `three-p.json`
- `reverse-dcf-surface.json`
- `expectation-gap.json`
- `expectation-analysis.json`
- `expectation-analysis-manifest.json`

run 최상위의 `opportunities.csv`가 기본 screening 결과입니다. `ranking.csv`는 `--enable-legacy-moat-ranking`을 켠 진단 실행에서만 채워집니다.

## Model router와 fail-closed 경계

일반 비금융·성숙기업·플랫폼·distressed 회사는 경제적 FCFF 계열로 route됩니다. Pre-revenue biotech에는 별도 rNPV 엔진이 있으나 현재 universe expectation runner와 자동 결합하지 않습니다. 금융회사는 excess-return equity model로 route되지만 아직 구현되지 않아 fail closed 합니다. 한 모델을 모든 업종에 억지로 적용하지 않습니다.

## 다음 검증 순서

Evidence Sensor의 boss 9/full 30 회귀와 extraction-set reproducibility를 먼저 유지합니다. 새 driver/3P 규칙을 과거 수익률에 맞춰 조정하지 않습니다. 계약이 고정된 뒤 보지 않은 여러 날짜에서 Expectation Gap 및 Value interaction을 검증합니다.
