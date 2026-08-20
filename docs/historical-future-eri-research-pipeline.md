# Historical Evidence → Future ERI V1 research pipeline

## Frozen question

V1 tests only whether six-axis operating evidence observed at a regular filing precedes the
63-trading-session market-implied expectation revision (ERI). It does not use a Value factor,
future return, or machine learning to define or rank the feature.

- Freeze tag: `future-eri-v1-preoutcome`
- Regular filings: annual, half-year, and quarterly reports
- Amendments: excluded from regular pairs and reported separately
- Axes: demand, price/mix, backlog, margin, inventory mismatch, capacity/capex
- Primary feature: equal-weight sum of six comparable-period directions in `{-1, 0, +1}`
- Missing axis: exclude; never impute as zero
- Research period: 2020–2025 pseudo-OOS calibration evidence
- Live shadow start: 2026-08-20

## Read-only source policy

The builder uses both source systems:

1. Arcana `data-lake/bronze/dart/business-info` HTML plus its silver DART metadata.
2. MoatRader `data-lake` OpenDART `original-document.zip` / `original.zip` archives plus the
   exact archive metadata.

Source files are opened only for read/stat/hash. Derived data is written only to a new
MoatRader experiment directory. A pre-build SHA-256/size/mtime manifest is verified again
after packet generation. A run is complete only when `build-manifest.json` exists and
`source-integrity-after.json` says `PASS_NO_SOURCE_MUTATION`.

Receipt duplicates are deduplicated by receipt and archive hash. Blank, nonnumeric, and
all-zero tickers are rejected rather than mapped to `000000`.

## Stage gates

1. `build_historical_future_eri_evidence.py`
   - Discovers and hashes both source systems.
   - Builds consecutive regular-filing pairs.
   - Produces issuer/date/ticker/receipt-masked six-axis packets.
   - Does not open ERI outcomes or returns.
2. `classify_historical_future_eri_evidence.py`
   - The LLM is a paired fact parser only.
   - It must ground both states with verbatim spans or abstain.
   - Missing evidence is never classified as stable.
   - Request-only mode makes no external call.
3. `seal_historical_future_eri_features.py`
   - Compares LLM labels with the filled human-gold file.
   - Default gate: at least 20 reviewed packets per axis, at least 80% overall exact
     status/state-pair agreement, and at least 70% agreement for every axis.
   - Only after the gate passes does it open the private source map, build six-axis
     `EvidenceObservation` rows, save axis/band/coverage distributions, and seal features.
   - Qualitative evidence uses the explicit
     `QUALITATIVE_AXIS_STATE_EQUAL_UNIT_V1` materiality rule; it is not presented as a
     numeric revenue-scaled fact.
4. `run_historical_future_eri_outcomes.py`
   - Refuses to open expectation/outcome inputs unless feature quality, coverage, and band
     gates authorize it.
   - Adds contemporaneous reverse-DCF expectations and seals again before opening outcomes.
   - Builds exactly t+63-session ERI labels.
   - Stores an explicit Operating EV → net debt → equity → diluted-share bridge.
   - Keeps the frozen equity ERI as primary and enterprise ERI as a diagnostic.
5. Clustered mechanism validation
   - Primary endpoint remains fixed five-band monotonicity.
   - Monthly IC and outer-spread HAC plus issuer × calendar two-way clustered slope are
     secondary statistics for overlapping 63-session observations.
6. `run_future_eri_downstream_validation.py`
   - Analyst revision and future-fundamental inputs remain unopened until the ERI mechanism
     gate passes.
   - Return data additionally requires explicit return-stage authorization and a completed
     secondary validation.
   - Actual future ERI is never used as an investment signal; the signal is the feature-only
     predicted revision score (initially the frozen F-score).

## ERI label null fixtures

The label contract has regression fixtures for:

- unchanged price and expectations;
- a WACC-only move fully explained by DCF;
- expected fundamental realization and roll-forward;
- split, dividend, and rights-issue capital-structure changes.

In each case, an economically explained price produces ERI at zero within Decimal tolerance.

## Value-neutralization policy

Value appears only after the ERI mechanism and return gates pass. The primary neutralization
specification jointly controls all available metrics:

- PBR
- PER
- P/FCF
- PSR
- PCR
- EV/EBITDA
- RPR

Each metric is also reported individually, followed by a Value + Momentum + Analyst Revision
specification. `PER+PBR` is retained only as a comparator control. It is not a primary ranking,
feature definition, or preferred neutralization specification. Reports include the retained
F-score coefficient and the share of signal variation explained by Value controls.

## Completed feature-only source build (2026-08-21)

Canonical completed output:

`data-lake/experiments/future-eri-historical-evidence-v1-2020-2025-complete2-20260821`

Key source and packet results:

| Item | Count |
|---|---:|
| Arcana regular filings | 49,787 |
| Arcana amendments excluded | 5,675 |
| MoatRader valid original regular filings | 6,622 |
| MoatRader original metadata/archive directories inspected | 19,041 |
| MoatRader duplicate archive copies removed | 2,813 |
| Archive hash mismatches | 0 |
| Merged regular filings | 50,418 |
| Filings retaining both source systems | 5,947 |
| Consecutive filing pairs | 43,752 |
| Blinded axis packets | 262,512 |
| Six-axis candidate-complete pairs | 15,978 |
| Candidate coverage | 36.52% |
| Unique issuers | 2,716 |
| Unique signal months | 62 |
| Source integrity records verified unchanged | 63,032 |

Candidate-complete pairs by fiscal year:

| Fiscal year | Count |
|---|---:|
| 2020 | 1,425 |
| 2021 | 2,032 |
| 2022 | 2,333 |
| 2023 | 2,512 |
| 2024 | 3,034 |
| 2025 | 4,642 |

Candidate span availability by axis:

| Axis | Both periods | Missing pair |
|---|---:|---:|
| Demand | 43,645 | 107 |
| Price/Mix | 42,947 | 805 |
| Backlog | 42,227 | 1,525 |
| Margin | 40,249 | 3,503 |
| Inventory mismatch | 16,923 | 26,829 |
| Capacity/Capex | 42,034 | 1,718 |

Inventory mismatch is the binding candidate-coverage axis. Sector coverage is currently a
diagnostic limitation: the available current sector snapshot covers only 150 tickers, so most
historical pairs remain `UNMAPPED`. Sector is never a feature, and the output explicitly labels
this report as current/non-PIT scope diagnostics.

The completed build remains at `AWAITING_LLM_AND_HUMAN_LABEL_QUALITY`. This is intentional:
the human-gold template has 240 rows (40 per axis), no API credential is persisted, the feature
dataset is not sealed, and neither the outcome vault nor return data has been opened.

## Commands

Activate the supplied environment and set local imports:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\Activate.ps1'
$env:PYTHONPATH='D:\Programming\python_example\Arcana;D:\Programming\python_example\MoatRader\src;D:\Programming\python_example\MoatRader'
```

Prepare full blinded requests without an API call:

```powershell
python scripts\classify_historical_future_eri_evidence.py `
  --input-build data-lake\experiments\future-eri-historical-evidence-v1-2020-2025-complete2-20260821 `
  --output <new-request-output>
```

Execute classifications only after providing a hidden key:

```powershell
python scripts\classify_historical_future_eri_evidence.py `
  --input-build <completed-feature-source-build> `
  --output <new-classification-output> `
  --execute --prompt-api-key
```

Then fill `quality/human-gold-template.csv` and run the feature quality/seal stage. Outcome and
return runners will remain fail-closed until their preceding stage status explicitly authorizes
them.
