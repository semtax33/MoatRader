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

## Return-blind 150 × 4 engineering audit

Inputs are the attached fixed 150-stock universe and the four development dates. Official historical KRX sector snapshots are required; a current-sector fallback is prohibited.

Two clean executions produced identical artifacts:

| Metric | Result |
|---|---:|
| Rows | 600 |
| PIT sector rows | 600 |
| Router eligible | 485 |
| Valuation generated | 485 |
| Cheap rank eligible | 259 |
| Return data accessed | false |

The 259/600 rank coverage is a disclosed limitation. Routing coverage expanded, but NAV/SOTP/rNPV inputs and non-positive/invalid fair values still limit usable Cheap ranks. This must not be hidden by switching to another model after valuation.

Audit artifacts (the final repeat pair):

- `data-lake/experiments/valuation-routing-stability-20260818-v8-pit-a`
- `data-lake/experiments/valuation-routing-stability-20260818-v8-pit-b`

The two runs have identical routing and signal payload hashes. This audit is engineering/stability evidence only; it does not read or report forward returns.

## Return-blind research adapter

`scripts/build_expectation_gap_research_inputs.py` converts each PIT company run into the 3P, fragility, industry-evidence, and source-reference fields required by the holdout contract. It reads no return data and fails closed when a company run or required valuation evidence is missing.

The development dry run for 2026-05-31 produced exactly 150 research rows and 150 normalized holdout-signal rows. Sixty rows were Cheap-rank eligible; Candidate A contained 60 names and Candidates B/C contained 55 after the frozen validity gate. The current development company runs are DART-only, so industry-evidence coverage was correctly reported as zero rather than synthesized. This dry run validates the adapter, not investment performance.

## Prospective holdout

The final contract is `data-lake/experiments/expectation-gap-production-candidate-v6/frozen-contract.json` (canonical contract hash `6911d74f02e5be8cdc6e3f6f6ec5a1cb6579a40c105ef055106001a62fd6f5f0`). It freezes the universe, code hashes, router, Cheap definition, risk policy, evaluation thresholds, preflight, four dates, and a 77-calendar-day forward-return horizon.

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
