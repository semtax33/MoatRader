# Historical Evidence → Future ERI V1 tombstone and V2 sparse pipeline

## V1 tombstone

V1의 연구 상태는 다음으로 고정한다.

`INCONCLUSIVE_DUE_TO_COMPLETE_CASE_COVERAGE_COLLAPSE`

이는 가설 기각이 아니다. 여섯 축이 모두 `COMPLETE`인 행만 허용한 feature contract가
95,868개 분류 결과를 2개 관측치로 축소했으므로 ERI 결과를 열지 못한 것이다. V1을 다시
살리기 위해 confidence threshold를 낮추거나, 결측 축을 0으로 대체하거나, band당 최소
20행 계약을 완화하지 않는다.

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

1. Arcana `data-lake/bronze/dart`의 `business-info`, `finance-comment`,
   `finance-statement` HTML plus its silver DART metadata. 각 섹션은 별도 provenance로
   보존하며 발견·누락·추출 수를 `source-audit.json`에 기록한다.
2. MoatRader `data-lake` OpenDART `original-document.zip` / `original.zip` archives plus the
   exact archive metadata.

Source files are opened only for read/stat/hash. Derived data is written only to a new
MoatRader experiment directory. A pre-build SHA-256/size/mtime manifest is verified again
after packet generation. A run is complete only when `build-manifest.json` exists and
`source-integrity-after.json` says `PASS_NO_SOURCE_MUTATION`.

2026-08-21에 2020–2025 Arcana 후보를 read/stat/SHA-256으로 전수 감사한 결과는 다음과
같다. 후보 49,795건 중 기존 pair universe의 anchor인 non-empty business-info는 49,787건,
finance-comment와 finance-statement는 각각 49,770건이며 세 섹션이 모두 있는 공시는
49,762건이다. business-info가 없는 8건은 pair universe를 바꾸지 않도록 제외하고 audit에
남긴다. 이 확인 과정 뒤 기존 63,032개 source-integrity record도 다시
`PASS_NO_SOURCE_MUTATION`을 통과했다.

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
   - `prepare_historical_evidence_classification_subset.py` limits execution to DEV 120,
     single-use LOCKED_TEST 120, or the 95,868 six-axis candidate-complete packets.
   - `evaluate_historical_evidence_parser.py` evaluates DEV without private data, freezes the
     parser/prompt/model hashes after a DEV pass, and consumes LOCKED_TEST exactly once.
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
   - Production runs first use `audit_historical_future_eri_outcome_eligibility.py` with a
     value-free inventory to require exact t+63 calendar coverage, target-price metadata,
     PIT financial metadata, net debt, diluted shares, and WACC sources.
   - Adds contemporaneous reverse-DCF expectations and seals again before opening outcomes.
   - Builds exactly t+63-session ERI labels.
   - Stores an explicit Operating EV → net debt → equity → diluted-share bridge.
   - Re-runs six production-path ERI null fixtures before opening any outcome value.
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

이 결과는 V1 tombstone 재현용이며 Arcana `business-info`만 연결했던 과거 build다.
`finance-comment`와 `finance-statement`를 강제하는 V2 production source gate를 통과하지
못하므로 V2 sparse feature 입력으로 재사용하지 않는다.

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

## Human gold and parser split status (2026-08-21)

- Human-gold candidate: `quality/human-gold-codex-manual-v1.csv`
- Review provenance: `CODEX_MANUAL_REVIEW` using blinded packets only
- DEV: 120 packets, 20 per axis
- LOCKED_TEST: 120 packets, 20 per axis
- Candidate-complete execution input: 95,868 packets = 15,978 pairs × 6 axes
- Parser version: `historical-evidence-parser-v1.0.0`
- Prompt SHA-256: `ea535289abee6697ea3fca5beb6af596ce6896b6c70b3e02a44967dc5f4e5863`
- Current parser stage: DEV requests prepared; no external call made because no API credential
  is present in the execution environment
- Outcome vault and return data remain unopened.

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

The frozen execution order is:

1. Execute DEV with `--packet-input <parser-build>\inputs\dev-packets.jsonl`.
2. Evaluate DEV with `evaluate_historical_evidence_parser.py`; freeze parser, prompt, model,
   gold, and locked-input hashes only after a DEV pass.
3. Execute LOCKED_TEST with the frozen parser and consume it once. Failure records
   `EVIDENCE_PARSER_NOT_VALIDATED` and forbids outcome access.
4. After a locked pass, classify only
   `<parser-build>\inputs\candidate-complete-packets.jsonl` using resumable response caches
   and `--workers` concurrency.
5. Seal features using the candidate classification build plus the separate locked quality
   classification build.
6. Run the value-blind outcome eligibility audit and pass its `--eligibility-build` to the
   production outcome runner. Rows beyond calendar coverage remain
   `OUTCOME_WINDOW_INCOMPLETE`; the horizon is never shortened.

Use `quality/human-gold-codex-manual-v1.csv` for the frozen split-aware gate. Outcome and return
runners remain fail-closed until their preceding stage status explicitly authorizes them.

## V2 sparse feature contract

V2는 15,978개 six-axis-complete 후보 prefilter를 사용하지 않는다. 검증된 전체 43,752개
filing pair를 유지하고, 262,512개 pair-axis 중 이전·현재 evidence가 모두 있는 축만 독립적으로
LLM 분류 입력에 넣는다. 모든 feature row에는 여섯 축이 남지만 각 축 상태는 아래처럼
명시적으로 구분된다.

- grounded direction: `-1`, `0`, `+1`
- unavailable: `NA`
- outcome-blind applicability rule상 비적용: `NOT_APPLICABLE`

`NA`와 `NOT_APPLICABLE`은 neutral `0`이 아니다. primary sparse feature는
`SignedBreadth = (N_positive - N_negative) / N_observed`이고 `N_observed`, applicable-axis
count, coverage를 별도 필드로 보존한다. score와 coverage/confidence를 합성하지 않는다.

`historical_evidence_v2.py`와 V2 stage scripts는 다음 순서와 게이트를 구현한다. 현재 허용
범위는 feature-only calibration까지이며 ERI, Value, return은 닫혀 있다.

1. `freeze_historical_sparse_contract_v2.py`
   - `0=현재 근거로 확인된 무변화`, `NA=현재 근거 없음`,
     `NOT_APPLICABLE=경제적 비적용` 계약을 분리하며 자동 변환을 금지한다.
   - 먼저 git commit, feature/applicability/deterministic policy SHA, parser prompt,
     signal timestamp policy, `last_grounded_days=450`을 measurement contract에 고정한다.
     feature-only calibration seal은 여기에 dual LOCKED SHA, explicit min Nobs, band
     boundary와 coverage gate SHA를 더한 `pre-outcome-manifest.json`을 만든다.
   - last-grounded는 이전 비교 기준으로만 쓴다. 현재 공시에 grounded evidence가 없으면
     450일 이내 과거 상태가 있어도 현재 결과는 반드시 `NA`다.
2. `build_historical_deterministic_pit_evidence_v2.py`
   - 전체 43,752 pair에서 margin, inventory mismatch, backlog, capex/capacity 네 축을
     outcome-blind PIT 규칙으로 먼저 측정한다.
   - 축별 applicable, grounded, -1/0/+1, NA, N/A, stale, extraction failure와 reason/source
     분포를 저장한다.
   - provenance 우선순위는
     `DETERMINISTIC_NUMERIC > STRUCTURED_TABLE > LLM_NARRATIVE`이며 평균하지 않는다.
3. `prepare_historical_locked_sets_v2.py`와
   `evaluate_historical_evidence_parser_v2.py`
   - 기존 V1/DEV packet ID를 명시적으로 제외한 새 Natural-frequency LOCKED와 별도
     directional-balanced LOCKED를 만든다.
   - semantic parser 축 Demand, Price/Mix, Capacity/Capex 각각에 Balanced
     negative/neutral/positive/insufficient/ambiguous를 최소 5건씩 요구한다.
   - 두 세트는 상호 비중복 single-use이며 human neutral → machine bullish 편향도 gate한다.
4. `audit_historical_evidence_abstentions_v2.py`
   - full LLM 전 200~500개 blinded abstention을 층화해 사람이 판정한다.
   - reason은 `TRUE_NO_MENTION`, `ONE_PERIOD_ONLY`, `RETRIEVAL_MISS`,
     `TABLE_EXTRACTION_FAIL`, `PERIOD_MISMATCH`, `AMBIGUOUS_HUMAN_TOO`,
     `NOT_APPLICABLE`만 허용한다.
   - retrieval/table/period upstream failure rate를 별도 gate하여 sparse 설계 문제처럼
     숨기지 않는다.
5. `prepare_historical_semantic_packets_v2.py`
   - 모든 43,752×6 packet을 LLM에 보내지 않는다. 기본은 Demand, Price/Mix와
     deterministic Capacity/Capex가 NA인 fallback만 선택한다.
   - deterministic grounded 또는 N/A 결과는 semantic parser가 덮어쓰지 않는다.
6. `build_historical_sparse_features_v2.py`
   - 모든 filing pair를 보존하고 미분류/abstained 축은 `NA`로 둔다.
   - `Nobs`는 grounded -1/0/+1만 세며 coverage=`Nobs/N_applicable`이다. `Ndir`은
     `|s|=1` 축 수 진단이고 primary는 계속 SignedBreadth다.
   - Arcana 세 HTML 섹션과 MoatRader original regular disclosure를 모두 읽었다는
     source audit 및 before/after integrity 일치를 production gate로 요구한다.
7. `calibrate_historical_sparse_features_v2.py`
   - Nobs 0..6 exact report, Ndir, co-observation,
     `corr(|SignedBreadth|, Coverage)`, `corr(|SignedBreadth|, Nobs)`, band별 Nobs 분포를
     outcome 없이 저장한다.
   - min Nobs 2/3을 hard-code하지 않고 freeze 시 명시적으로 선택한다. [-1,+1] five-band
     boundary도 feature-only threshold 또는 quantile로 정한다.
   - band별 row뿐 아니라 unique issuer/month, 최대 issuer/year/evidence-source 집중도까지
     통과해야 pre-outcome seal이 가능하다. 이 seal도 outcome authorization을 열지 않으며,
     별도 V2 eligibility와 Reverse DCF 단계가 구현·통과된 뒤에만 t+63 ERI를 허용한다.

V2 selective semantic packet 준비 예시:

```powershell
python scripts\prepare_historical_semantic_packets_v2.py `
  --filing-pair-input <new-three-section-source-build>\private\filing-pairs.jsonl `
  --packet-input <new-three-section-source-build>\llm\blinded-packets.jsonl `
  --deterministic-evidence-input <pit-build>\deterministic-axis-evidence.jsonl `
  --applicability-input <pit-build>\axis-applicability.jsonl `
  --output <new-v2-semantic-input>
```

Outcome-blind coverage 진단 예시:

```powershell
python scripts\calibrate_historical_sparse_features_v2.py `
  --feature-build <new-v2-sparse-feature-build> `
  --output <new-v2-diagnostics>
```

진단을 검토한 뒤에만 명시적인 Nobs threshold로 freeze한다. 이 시점까지 ERI, return,
Value 데이터를 열지 않는다. Value 비교 순서는 ERI mechanism gate 이후이며 PBR, PER,
P/FCF, PSR, PCR, EV/EBITDA, RPR의 joint·individual neutralization을 보고한다.
`PER+PBR`은 comparator일 뿐 우선 ranking이 아니다.
