# Expectation GAP Alpha/Risk production contract

## Frozen architecture

- `Cheap` is the only production ranking signal: `primary fair value / PIT market price - 1`.
- Cross-company rank uses `signal date × valuation method × economic archetype` percentiles.
- `Improving` is thesis-confirmation diagnostics only.
- `Possible = FAIL` is a hard 3P exclusion. Plausible/Probable uncertainty remains visible and is not added to rank.
- Fragility and industry evidence form a risk overlay only. They may cap or exclude a position but never change its Cheap rank.
- The legacy weighted composite is retained only as a benchmark diagnostic.

The preregistered candidates are:

| Candidate | Rank | Gate/overlay |
|---|---|---|
| A | Cheap | none |
| B | Cheap | hard 3P validity gate |
| C | Cheap | hard 3P gate plus frozen fragility/industry risk overlay |

## Price-blind valuation routing

The router selects a primary method before market price is read. Supported production-contract methods are Economic FCFF, Normalized FCFF, RIM, rNPV, Scenario DCF, APV, NAV, and SOTP. Missing method-specific inputs fail closed; the pipeline does not switch models after seeing a valuation or market price.

SOTP parts carry explicit included/excluded cash-flow scopes so the same cash flow cannot be valued twice. All methods return the common `ValuationResult` contract. The primary method is never averaged with a cross-check method.

## Legacy return-blind 150 × 4 engineering audit

Inputs are the attached fixed 150-stock universe and the four development dates. Official historical KRX sector snapshots are required; a current-sector fallback is prohibited.

두 번의 schema v1 실행은 byte-identical했지만, 이후 route/engine 감사에서 다음 제한이
확인됐습니다.

| Metric | Result |
|---|---:|
| Rows | 600 |
| PIT sector rows | 600 |
| Router eligible | 485 |
| Valuation generated | 485 (legacy FCFF adapter 469, RIM 16) |
| Cheap rank eligible | 259 |
| Return data accessed | false |

이 실행은 multi-model 성과나 실행 완결성을 증명하지 않습니다. 생성 valuation의
96.7%가 legacy FCFF adapter였고, NAV/SOTP는 route만 선택된 채 valuation이 0건이었으며
rNPV/APV route도 0건이었습니다. 따라서 아래 v8 pair는 legacy architecture 진단용이며
새 Unified Value freeze 근거로 재사용할 수 없습니다.

Audit artifacts (the final repeat pair):

- `data-lake/experiments/valuation-routing-stability-20260818-v8-pit-a`
- `data-lake/experiments/valuation-routing-stability-20260818-v8-pit-b`

두 실행의 동일 hash는 결정론만 증명합니다. 모델 다양성·정확성은 증명하지 않습니다.

현재 schema v4 freeze gate는 fallback FCFF와 LLM 호출이 모두 0이어야 하며, 입력 계약을
충족해 eligible인 route는 100% 실제 valuation을 생성해야 합니다. Route와 actual engine의
호환도 100%, route stability는 90% 이상이어야 합니다. Trusted reference class가 20개
미만이라 percentile을 만들지 않는 것은 실행 실패와 분리해 기록합니다. 서로 다른 실제
엔진이 최소 3개 실행되고 단일 엔진 비중이 90%를 넘으면 freeze를 거부합니다.

## Current return-blind architecture calibration

Normalized FCFF, Scenario DCF, rNPV와 RIM은 method-specific deterministic builder와 실제
엔진으로 실행합니다. SOTP도 각 part의 실제 sub-model 실행 계약을 갖지만, 실제 part
evidence가 없는 historical route는 fail closed합니다. 따라서 architecture PASS는 typed
input, 실행 완결성, 엔진 일치, route 안정성과 결정론을 증명할 뿐 alpha나 Universal Value
ranking 우월성을 증명하지 않습니다. RIM의 현재 입력 생성 표본은 41개 issuer로 N≥20을
충족하지만, 이 4개 역사 signal-date 내부의 RIM reference class는 여전히 작아 올바르게
`UNRANKABLE`입니다.

최종 반복 실행은
`data-lake/experiments/unified-value-routing-calibration-20260820-v8`입니다. 600개 route 중
413개가 입력 계약을 충족했고 413개 모두 실제 엔진으로 실행됐습니다. A/B 산출물은
byte-identical, route-engine 호환은 100%, 전체 route stability는 97.1%, fallback FCFF와
LLM 호출은 각각 0건입니다. 실제 엔진별 실행은 Economic FCFF 298, Normalized FCFF 80,
Scenario DCF 16, RIM 16, rNPV 3건입니다. Scenario의 이전-route 기준 안정성은 82.9%로
전체보다 낮아 계속 별도 관찰합니다. 6개 이탈은 4개 persistent-loss 조건 해제와 2개
pipeline 구조 확인으로 reason code에 보존됐으며, 이 수치를 숨기거나 100%로 보정하지
않습니다. NAV/SOTP/APV는 실제 evidence가 없어 0건 실행됐고 이를 coverage 성공으로
간주하지 않습니다.

## Return-blind research adapter

`scripts/build_expectation_gap_research_inputs.py` converts each PIT company run into the 3P, fragility, industry-evidence, and source-reference fields required by the holdout contract. It reads no return data and fails closed when a company run or required valuation evidence is missing.

The development dry run for 2026-05-31 produced exactly 150 research rows and 150 normalized holdout-signal rows. Sixty rows were Cheap-rank eligible; Candidate A contained 60 names and Candidates B/C contained 55 after the frozen validity gate. The current development company runs are DART-only, so industry-evidence coverage was correctly reported as zero rather than synthesized. This dry run validates the adapter, not investment performance.

## Prospective holdout — 기존 v6 계약 폐기 필요

기존 `expectation-gap-production-candidate-v6/frozen-contract.json`은 schema v1 감사와
수정 전 source hash를 고정하므로 새 multi-model 코드에는 유효하지 않습니다. 아래
날짜를 사용하려면 schema v4 engineering audit A/B를 먼저 통과한 뒤 새 계약을
freeze해야 합니다. 기존 hash를 업데이트하거나 덮어쓰면 안 됩니다.

Holdout dates:

1. 2026-08-31
2. 2026-11-30
3. 2027-02-28
4. 2027-05-31

For each date:

1. Collect a KRX sector snapshot available on or before the signal date.
2. Run the one-date return-blind routing/valuation audit with `--expected-date-count 1`.
3. Produce a 150-row research JSON containing `risk`, `confirmation`, and PIT `source_references` with `scripts/build_expectation_gap_research_inputs.py`.
4. Build holdout signals with `scripts/build_expectation_gap_holdout_signals.py`.
5. Seal them with `scripts/seal_expectation_gap_holdout.py` before any forward return is accessed.

Run `scripts/preflight_expectation_gap_holdout.py` before and after these steps. Before the signal date it returns `WAIT`; on or after the date a missing input becomes `FAIL`. A `READY` result proves the contract/source hashes, frozen universe, 150-row PIT sector/valuation/research panels, normalized signal contract, and immutable seal all agree without reading returns.

The seal recalculates method/archetype percentiles, verifies all 150 tickers exactly match the frozen universe, rejects post-cutoff sources, verifies the contract's source hashes, and writes immutable signal/candidate hashes.

Only after the final date plus 77 days may `scripts/evaluate_frozen_expectation_gap_holdout.py` join returns. Candidate C succeeds only if its sector-neutral IC is no more than 0.05 below A and it improves either the top-portfolio worst decile by at least 3 percentage points or downside capture by at least 0.10.
