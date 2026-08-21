# Historical Evidence → Future ERI V1, V1R, and V2 pipeline

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

V1R와 V2 builder는 두 source system을 모두 사용한다. V1 재현 모드는 기존 계약대로
Arcana `business-info`와 MoatRader original disclosure만 사용한다.

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

## V1R source-corrected replication

V1R(V1.1)은 V1을 덮어쓰거나 되살리는 모델 변경이 아니다. V1의 six-axis complete-case
feature 가설, 방향 규칙, equal-weight F-score, 다섯 band를 그대로 두고 Arcana 정기공시의
세 HTML 섹션을 모두 읽도록 source coverage만 교정한 자연실험이다.

| Arm | Source contract | Feature contract | Purpose |
|---|---|---|---|
| A: V1 | Arcana business-info + Moat original | six-axis complete | 기존 tombstone 기준 |
| B: V1R | Arcana business-info + finance-comment + finance-statement + Moat original | V1과 동일 | source coverage 효과 |
| C: V2 | B와 동일 | sparse breadth | feature contract 효과 |

따라서 A→B는 source coverage 효과이고 B→C는 feature contract 효과다. V1 tag
`future-eri-v1-preoutcome`은 보존하며 V1R은 독립 tag
`future-eri-v1r-three-section-preoutcome`을 사용한다. V1R LOCKED_TEST도 packet 내용이
달라졌으므로 V1/DEV packet ID를 재사용하지 않고 다음 네 source stratum을 축별로 검증한다.

- business-info evidence
- finance-comment evidence
- finance-statement evidence
- multi-section evidence

Source-effect audit는 Arcana-only, Arcana+Moat overlap, 새 finance-comment 기여,
새 finance-statement 기여를 분리한다. Moat original archive와 Arcana가 겹치는 filing은
Moat archive 안에 이미 전체 공시가 있을 수 있으므로 단순히 "새 섹션"으로 집계하지 않는다.

V1R feature-only feasibility gate는 outcome, return, ERI, Value 데이터를 열지 않고 축별
`COMPLETE` 비율, six-axis complete 행 수, issuer/month 수, V1 다섯 band 분포만 본다.
각 band 최소 20행이면 향후 별도 outcome-eligibility 단계로 진행할 수 있고, 미달이면
complete-case 계약을 tombstone 처리한다. 어느 경우에도 PER+PBR은 feature나 우선 ranking이
아니며 이 단계에서는 PBR을 포함한 Value 지표를 전혀 사용하지 않는다.

V1R 실행 순서는 다음과 같다.

1. `build_historical_future_eri_evidence.py --research-variant V1R`로 새 three-section source
   build와 before/after source-integrity audit를 생성한다.
2. `prepare_historical_v1r_locked_set.py`로 V1/DEV와 비중복인 axis × source-stratum
   LOCKED_TEST를 만든다.
3. `evaluate_historical_evidence_parser_v1r.py freeze`, LOCKED classification,
   `evaluate` 순으로 parser를 single-use 검증한다.
4. clean commit에서 `freeze_historical_v1r_contract.py`를 실행하고 새 V1R tag를 고정한다.
   `--allow-dirty-for-dry-run`은 테스트 전용이며 production authorization이 아니다.
5. 전체 pair×6 분류 뒤 `build_historical_complete_features_v1r.py`로 V1과 같은 complete-case
   feature를 만들고 `audit_historical_v1r_feasibility.py`로 band당 최소 20행을 감사한다.
6. PASS하더라도 outcome runner는 봉인된 V1R manifest와 별도의 value-free
   outcome-eligibility build 없이는 expectation/outcome 입력을 열지 않는다.

예시 source build:

```powershell
python scripts\build_historical_future_eri_evidence.py `
  --research-variant V1R `
  --output data-lake\experiments\future-eri-v1r-three-section-source
```

기존 V1을 재현해야 할 때만 `--research-variant V1`을 명시한다. 두 모드는 항상 서로 다른
새 output directory에 기록한다.

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

V2 Value data is opened only after the Full Evidence Index → t+63 Future ERI evaluation is
complete and its feature/label hashes are verified. This diagnostic does not require or open
future returns. Every Value metric is an equal-status, same-sample sensitivity; there is no
preferred joint neutralizer and no Value-based ranking:

- PBR (B/M)
- PER (E/P)
- P/FCF (FCF/P)
- PSR (Sales/P)
- PCR (CFO/P)
- EV/EBITDA (EBITDA/EV)
- EV/EBIT, POR, PGPR
- RPR/PRR (R&D/P)
- retained earnings/P, assets/P, NCAV/P

For each signal month and metric, raw Full Evidence Index IC and residualized IC use exactly the
same complete-case observations. The report includes IC retention/attenuation, Value-exposure
R-squared, Newey-West inference, and moving-block bootstrap intervals. `PBR`, `PER`, and
`PER+PBR` receive no priority: the V2 diagnostic does not create a `PER+PBR` joint primary,
portfolio rank, or trading signal. Actual Future ERI remains the downstream outcome only.

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
`SignedBreadth = (N_positive - N_negative) / Nobs`이고 `Nobs`는 signed-score 대상 축의
grounded `-1/0/+1`만 센다. CAPEX는 이 분자·분모에서 모두 제외한다. applicable-axis
count와 coverage는 별도 필드로 보존하며 score에 coverage/confidence를 곱하지 않는다.

`historical_evidence_v2.py`와 V2 stage scripts는 다음 구성요소와 게이트를 구현한다.
production 실행 순서는 반드시 `semantic DEV 평가 → V2 parser freeze → measurement/code
contract freeze → Natural/Balanced single-use LOCKED 평가 → 87,204건 full semantic 분류 →
Full Evidence Index coverage seal → t+63 ERI open`이다. 검증용
`CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE` 결과는 sparse production 입력으로
승격할 수 없고, `FULL_HISTORICAL` authorization과 dual LOCKED SHA가 기록된 분류만 허용한다.
Full Index seal은 classifier, sparse builder, sealer, ERI runner, Value-neutral runner의 동결 SHA와 git commit도
재검증한다. 이 순서 전에는 ERI, Value, return이 닫혀 있다.

1. `freeze_historical_sparse_contract_v2.py`
   - `0=현재 근거로 확인된 무변화`, `NA=현재 근거 없음`,
     `NOT_APPLICABLE=경제적 비적용` 계약을 분리하며 자동 변환을 금지한다.
   - 먼저 git commit, feature/applicability/deterministic policy SHA, parser prompt,
     signal timestamp policy, `last_grounded_days=450`을 measurement contract에 고정한다.
     feature-only calibration seal은 여기에 dual LOCKED SHA, 고정 `min Nobs=2`, 고정
     경제 밴드 규칙과 coverage gate SHA를 더한 `pre-outcome-manifest.json`을 만든다.
   - last-grounded는 이전 비교 기준으로만 쓴다. 현재 공시에 grounded evidence가 없으면
     450일 이내 과거 상태가 있어도 현재 결과는 반드시 `NA`다.
2. `build_historical_deterministic_pit_evidence_v2.py`
   - 전체 43,752 pair에서 margin, inventory mismatch, backlog, capex/capacity 네 축을
     outcome-blind PIT 규칙으로 먼저 측정한다.
   - PIT 입력 추출은 Arcana `business-info`, `finance-comment`, `finance-statement`와
     MoatRader OpenDART original regular disclosure를 모두 읽고 원본 SHA-256을 검증한다.
     DART 표 계정명의 선행 행 번호와 후행 주석 참조는 표시 메타데이터로만 제거하며,
     계정 의미 자체를 넓혀 매칭하지 않는다.
   - 축별 applicable, grounded, -1/0/+1, NA, N/A, stale, extraction failure와 reason/source
     분포를 저장한다.
   - provenance 우선순위는
     `DETERMINISTIC_NUMERIC > STRUCTURED_TABLE > LLM_NARRATIVE`이며 평균하지 않는다.
   - deterministic primary SignedBreadth/Nobs는 `MARGIN`, `INVENTORY_MISMATCH`,
     `BACKLOG` 세 축만 사용한다. `CAPACITY_CAPEX`는 raw investment-direction 진단이고
     primary signed score에 포함하지 않는다.
   - 선택 2는 `prepare_historical_last_grounded_inputs_v2.py`로 현재 증거가 존재하는
     deterministic NA 행에 한해 가장 최근의 groundable 이전 공시를 최대 450일까지
     비교 기준으로 구성한다. 현재 증거를 과거 값으로 이월하지 않으며 outcome, return,
     ERI, Value를 열지 않는다.
3. `prepare_historical_locked_sets_v2.py`와
   `evaluate_historical_evidence_parser_v2.py`
   - 기존 V1/DEV packet ID를 명시적으로 제외한 새 Natural-frequency LOCKED와 별도
     directional-balanced LOCKED를 만든다.
   - semantic parser 축 Demand, Price/Mix 각각에 Balanced
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
   - 모든 43,752×6 packet을 LLM에 보내지 않는다. semantic 대상은 Demand와
     Price/Mix만 선택한다. Capacity/Capex는 deterministic raw 진단으로 유지한다.
   - deterministic grounded 또는 N/A 결과는 semantic parser가 덮어쓰지 않는다.
6. `build_historical_sparse_features_v2.py`
   - 모든 filing pair를 보존하고 미분류/abstained 축은 `NA`로 둔다.
   - `Nobs`는 CAPEX를 제외한 primary signed axes의 grounded -1/0/+1만 센다. `Ndir`은
     `|s|=1` primary 축 수 진단이고 primary는 계속 SignedBreadth다. coverage는 점수와
     별도로 저장하며 `SignedBreadth × coverage`를 만들지 않는다.
   - Arcana 세 HTML 섹션과 MoatRader original regular disclosure를 모두 읽었다는
     source audit 및 before/after integrity 일치를 production gate로 요구한다.
7. `calibrate_historical_sparse_features_v2.py`
   - primary Nobs 0..5 exact report, Ndir, co-observation,
     `corr(|SignedBreadth|, Coverage)`, `corr(|SignedBreadth|, Nobs)`, band별 Nobs 분포를
     outcome 없이 저장한다.
   - V2 freeze는 `min Nobs=2`로 고정한다. 표본 quantile이나 outcome을 보지 않고
     `Strong Bear=-1`, `Bear=(-1,0)`, `Neutral=0`, `Bull=(0,1)`,
     `Strong Bull=+1`의 경제적 다섯 밴드를 그대로 사용한다.
   - band별 row뿐 아니라 unique issuer/month, 최대 issuer/year/evidence-source 집중도까지
     통과해야 pre-outcome seal이 가능하다. 이 seal도 outcome authorization을 열지 않으며,
     별도 V2 eligibility와 Reverse DCF 단계가 구현·통과된 뒤에만 t+63 ERI를 허용한다.
8. `freeze_historical_evidence_index_v2.py`
   - 미래 primary를 Demand + Price/Mix + Margin + Inventory + Backlog의
     `FULL_EVIDENCE_SIGNED_BREADTH_V2`로 사전명세한다. Demand/Price-Mix semantic parser의
     dual LOCKED gate 전에는 Full Index를 물질화하지 않는다.
   - 현재 사용 가능한 secondary baseline은 Margin + Inventory + Backlog의
     `DETERMINISTIC_CORE_SIGNED_BREADTH_V2`다. CAPEX는 raw 방향 진단으로만 남는다.
   - 선택 2 deterministic evidence의 43,752개 pair를 모두 다시 검증한 결과, 고정
     `Nobs>=2` eligible은 37,014개, 2,321개 issuer, 59개 signal month다. 고정 밴드별
     row/issuer/month/year/Nobs/exact-score 분포와 집중도를 outcome 없이 저장한다.
   - 입력 gate는 Arcana `business-info`·`finance-comment`·`finance-statement` 세 섹션과
     MoatRader OpenDART 원천 정기공시를 모두 사용했는지, 예상 source path와 검증 path가
     일치하는지, 원본 write/hash mismatch가 0인지 다시 확인한다.
   - 이 동결 뒤에도 `outcome_stage_authorized=false`다. 다음 단계는 Demand/Price-Mix
     semantic parser LOCK이며 ERI, return, Value는 계속 닫혀 있다.

V2 selective semantic packet 준비 예시:

```powershell
python scripts\prepare_historical_semantic_packets_v2.py `
  --filing-pair-input <new-three-section-source-build>\private\filing-pairs.jsonl `
  --packet-input <new-three-section-source-build>\llm\blinded-packets.jsonl `
  --deterministic-evidence-input <pit-build>\deterministic-axis-evidence.jsonl `
  --applicability-input <pit-build>\axis-applicability.jsonl `
  --output <new-v2-semantic-input>
```

HUMAN 판정에서 축별 negative/positive 최소 5건이 부족하면 gate를 낮추지 않는다.
원래 Natural/Balanced와 V1/DEV packet ID를 모두 제외한 outcome-blind 보충 후보를 만들고,
보충 HUMAN 판정이 끝난 뒤 기존 결정과 합친다. `selection_hint`는 검토 라우팅일 뿐 gold가
아니며 보충 후보도 반드시 HUMAN이 판정한다.

```powershell
python scripts\prepare_historical_locked_sets_v2.py prepare-supplement `
  --packet-input <semantic-packets.jsonl> `
  --base-candidate-build <base-v2-candidates> `
  --prior-v1-input <v1-locked.jsonl> `
  --dev-input <dev-packets.jsonl> `
  --output <new-supplement-candidates>

python scripts\prepare_historical_locked_sets_v2.py extend-candidates `
  --base-candidate-build <base-v2-candidates> `
  --supplemental-candidate-build <new-supplement-candidates> `
  --output <new-extended-v2-candidates>

python scripts\merge_historical_human_review_decisions_v2.py `
  --input <base-human-decisions.json> `
  --input <supplement-human-decisions.json> `
  --output <new-merged-human-decisions.json>
```

세 명령은 입력 파일을 덮어쓰지 않고 새 output만 허용하며, packet 중복·해시 계보·
`outcome_vault_opened=false`·`return_data_opened=false`를 검증한다.

첫 Balanced LOCKED가 단일사용 평가에서 실패하면 그 결과와 consumption record를
그대로 보존한다. Retest 1은 실패 분류나 불일치 행을 읽지 않고, 이미 사용한
V1·DEV·Natural·Balanced·기타 retest ID를 명시적으로 제외한다. 방향성 후보는
outcome-blind 실현 문장 cue를 우선하고 부족분만 사전 정의된 광범위 텍스트 cue로
채운다. 기존 HUMAN strata는 비방향 후보 sampling에만 사용할 수 있으며 새 retest
gold로 재사용하지 않는다. 모든 후보는 selection hint를 보지 않은 HUMAN이 새로
판정해야 한다.

```powershell
python -m scripts.prepare_historical_balanced_retest_v2 `
  --packet-input <semantic-packets.jsonl> `
  --prior-v1-input <v1-input.jsonl> `
  --dev-input <semantic-dev.jsonl> `
  --prior-v2-locked-input <first-natural.jsonl> `
  --prior-v2-locked-input <first-balanced.jsonl> `
  --prior-human-gold <prior-human-gold.csv> `
  --prior-human-gold-materialization-manifest <prior-human-manifest.json> `
  --failed-balanced-evaluation-manifest <failed-balanced-stage.json> `
  --failed-balanced-consumption-record <balanced-consumption.json> `
  --parser-freeze-manifest <root-parser-freeze.json> `
  --output <new-balanced-retest-candidates>

python -m scripts.materialize_historical_balanced_retest_v2 materialize `
  --candidate-build <new-balanced-retest-candidates> `
  --review-decisions <fresh-human-review-decisions.json> `
  --output <new-balanced-retest-human-gold>

python -m scripts.materialize_historical_balanced_retest_v2 freeze `
  --parser-freeze-manifest <root-parser-freeze.json> `
  --candidate-build <new-balanced-retest-candidates> `
  --human-gold-build <new-balanced-retest-human-gold> `
  --output <new-balanced-retest-freeze.json>
```

Materialization은 축별 `COMPLETE_NEGATIVE`, `COMPLETE_NEUTRAL`,
`COMPLETE_POSITIVE`, `INSUFFICIENT_EVIDENCE`, `AMBIGUOUS`를 최소 5개씩
확보한 경우에만 50행 retest를 고정한다. `BALANCED_RETEST_1` 평가는 파생 freeze와
단일사용 consumption record를 요구하며 첫 Balanced 결과를 재점수화하거나
supersede하지 않는다. Dual manifest는 통과한 Natural/Natural Retest 1과
Balanced/Balanced Retest 1의 root parser-freeze 및 classification-stage SHA 계보가
정확히 일치할 때만 생성된다.

최종 Natural/Balanced LOCKED를 확정한 뒤 DEV도 같은 HUMAN 리뷰 풀에서 만들 수 있다.
이때 두 LOCKED의 packet ID를 모두 제외하고, COMPLETE·AMBIGUOUS를 최소 포함한 축별
30건을 outcome-blind하게 고정한다.

```powershell
python scripts\prepare_historical_semantic_dev_v2.py `
  --candidate-build <extended-v2-candidates> `
  --adjudicated-human-gold <materialized-human-gold.csv> `
  --human-gold-materialization-manifest <human-gold-materialization-manifest.json> `
  --locked-set-build <final-v2-locked-sets> `
  --output <new-v2-semantic-dev>
```

DEV 준비 manifest는 LOCKED overlap 0, HUMAN authority, 입력 해시, Value/return/outcome 미개방,
`per_pbr_role=NOT_USED`를 기록한다.

V2 semantic `--execute`는 실행 범위를 반드시 명시한다. DEV·LOCKED 검증은 최대
2,000 packet으로 제한되며, 전체 historical 실행은 통과한 dual LOCKED manifest와
사전 작성한 selection/cost manifest 없이는 transport를 열지 않는다.
Cost manifest도 실제 frozen V2 prompt/model로 실행한 Natural·Balanced 두 stage의
usage를 사용해야 한다. `pilot_prompt_differs_from_frozen_full_prompt=true`인 과거 V1
사용량 기반 운영 추정치는 참고용일 뿐 full-run authorization으로 사용할 수 없다.
최종 cost manifest는 passed dual LOCKED manifest가 봉인한 Natural·Balanced
classification-stage SHA-256 두 개와 입력 stage가 정확히 같아야 한다. DEV나 실패한
Natural stage를 두 번째 pilot으로 바꿔 끼울 수 없다. Dual gate 전에는 아래 prelock
preflight만 만들 수 있으며, status와 `full_historical_execution_authorized=false` 때문에
전체 실행 gate의 입력으로 사용할 수 없다.

```powershell
python scripts\classify_historical_future_eri_evidence.py `
  --input-build <source-build> `
  --packet-input <dev-or-locked-packets.jsonl> `
  --output <new-validation-classification> `
  --parser-profile DEMAND_PRICE_MIX_V2 `
  --semantic-execution-scope PILOT_OR_LOCKED_VALIDATION `
  --execute --prompt-api-key

python -m scripts.prepare_historical_semantic_cost_preflight_v2 `
  --semantic-packet-input <semantic-packets.jsonl> `
  --semantic-selection-manifest <semantic-packets.jsonl.manifest.json> `
  --observed-stage DEV=<completed-v2-dev-classification\stage-status.json> `
  --observed-stage NATURAL_LOCKED=<completed-v2-natural-classification\stage-status.json> `
  --pricing-checked-date <YYYY-MM-DD> `
  --output <new-prelock-cost-preflight.json>

python -m scripts.prepare_historical_semantic_cost_manifest_v2 `
  --semantic-packet-input <semantic-packets.jsonl> `
  --semantic-selection-manifest <semantic-packets.jsonl.manifest.json> `
  --dual-locked-manifest <passed-dual-locked-stage.json> `
  --pilot-stage-manifest <completed-v2-natural-classification\stage-status.json> `
  --pilot-stage-manifest <completed-v2-balanced-classification\stage-status.json> `
  --pricing-checked-date <YYYY-MM-DD> `
  --output <new-exact-v2-cost-manifest.json>

python scripts\classify_historical_future_eri_evidence.py `
  --input-build <source-build> `
  --packet-input <semantic-packets.jsonl> `
  --output <new-full-semantic-classification> `
  --parser-profile DEMAND_PRICE_MIX_V2 `
  --semantic-execution-scope FULL_HISTORICAL `
  --dual-locked-manifest <passed-dual-locked-stage.json> `
  --semantic-selection-manifest <semantic-packets.jsonl.manifest.json> `
  --semantic-cost-manifest <cost-manifest.json> `
  --execute --prompt-api-key
```

Full Index seal은 전체 classification stage에 기록된 위 세 authorization SHA도 다시
검증한다. 검증용 분류나 gate 이전 분류는 Full Index 입력으로 승격할 수 없다.

Outcome-blind coverage 진단 예시:

```powershell
python scripts\calibrate_historical_sparse_features_v2.py `
  --feature-build <new-v2-sparse-feature-build> `
  --output <new-v2-diagnostics>
```

고정 `Nobs=2`와 경제적 다섯 밴드의 coverage gate를 통과한 뒤에도 ERI, return,
Value 데이터를 열지 않는다. Value 비교는 Full Index → t+63 ERI 평가와 artifact hash
검증이 끝난 뒤에만 별도 실행한다. PBR, PER, P/FCF, PSR, PCR, EV/EBITDA,
EV/EBIT, POR, PGPR, RPR/PRR, retained earnings/P, assets/P, NCAV/P를 모두 동등한
individual sensitivity로 보고하며, joint primary 또는 Value ranking은 만들지 않는다.

```powershell
python scripts\run_historical_evidence_index_value_neutralization_v2.py `
  --eri-build <completed-v2-eri-build> `
  --value-input <pit-value-controls.jsonl> `
  --value-manifest <pit-value-controls-manifest.json> `
  --output <new-v2-value-neutralization-output>
```

Value manifest는 모든 관측치가 signal timestamp 이전에 이용 가능했음을 확인하고,
Future ERI로 Value control을 만들지 않았으며 원천파일 before/after 무결성이 통과했음을
명시해야 한다. 이 러너는 future return을 읽지 않고 순위를 출력하지 않는다.
