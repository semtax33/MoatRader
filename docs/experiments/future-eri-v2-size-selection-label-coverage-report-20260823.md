# Future ERI V2 Size·Selection·Label Coverage Diagnostic

기준일: 2026-08-23  
역할: 진단 및 라벨 커버리지 엔지니어링 결과. V2 primary retest나 랭킹 결과가 아니다.

## 결론

- Full Evidence V2와 `Nobs`는 Size proxy로 보이지 않는다. 다만 최종 ERI eligibility는 Size와 유의하게 연결된다.
- 같은 1,640개 complete-case에서 Size를 통제해도 월별 IC는 `0.104632 → 0.093368`로 89.23% 남았다.
- outcome-blind selection propensity weighting에서는 월별 IC가 `0.104632 → 0.049223`으로 약화됐다. 그러나 weighted ESS가 `380.39 / 1,640`(23.19%)에 불과해 severe weight concentration 진단이며 primary retest를 대체하지 않는다.
- 연간 DART 계정명 추출 보완으로 reverse expectation은 `1,834 → 4,931`, 잠재 t+63 label은 `1,673 → 4,484`로 확대됐다.
- 사전등록 coverage gate의 7개 조건 중 3개가 미달했다. 따라서 expanded Future ERI outcome은 열지 않았고, expanded-label 5-band replication도 실행하지 않았다.
- Evidence V2 행·점수·band·threshold, reverse-DCF solver, 가격 정책, 63거래일 horizon은 변경하지 않았다.

## 1. Size diagnostic

Size는 정확한 신호일의 `Open × listed shares`를 로그 변환해 사용했다. 같은 날 종가 시총은 사용하지 않았다.

| 항목 | 결과 |
|---|---:|
| baseline observations | 37,014 |
| log market cap complete | 34,410 |
| pooled rho: Evidence ↔ log Size | -0.001276 |
| monthly mean rho: Evidence ↔ log Size | -0.018697, bootstrap CI [-0.076953, 0.050658] |
| pooled rho: Nobs ↔ log Size | 0.006239 |
| monthly mean rho: Nobs ↔ log Size | 0.024808, bootstrap CI [-0.043250, 0.072628] |
| final ERI eligibility OR / 1 Size SD | 1.3353, issuer-clustered z = 5.618 |
| eligibility predicted probability | -1 SD 3.47%, mean 4.58%, +1 SD 6.02% |
| raw monthly IC | 0.104632 |
| Size-neutral monthly IC | 0.093368 |
| signed retention | 89.23% |
| delta IC | -0.011264, bootstrap CI [-0.033267, 0.000056] |

Size bucket별 월별 IC는 Small `0.044585`(CI가 0 포함), Mid `0.111090`, Large `0.107070`이다. 따라서 현재 근거는 “Size가 신호를 전부 설명한다”가 아니라 “신호 관계는 대체로 남지만, 라벨 eligibility와 Small 표본 안정성에는 Size 문제가 있다”이다.

## 2. Outcome-blind selection propensity weighting

37,014개 baseline에서 1,640개 최종 ERI eligible 여부를 issuer-grouped 5-fold cross-fit logistic으로 추정했다. Future ERI 값은 propensity rows와 seal을 먼저 기록한 뒤에만 열었다.

사용 변수는 log Size, Nobs, coverage/axis counts, semantic·deterministic source count, PIT annual snapshot/history/completeness, year, Size bucket, sector, listing-age proxy, financial archetype/security type, Evidence/annual source mode, market-data availability다. Sector는 2026 current KRX kind의 non-PIT sensitivity 변수이고, 실제 financial-statement type 대신 annual source mode와 metric completeness proxy를 사용했다.

| 항목 | 결과 |
|---|---:|
| OOF AUC | 0.931219 |
| OOF Brier | 0.035515 |
| raw monthly IC | 0.104632, CI [0.078754, 0.160159] |
| weighted monthly IC | 0.049223, CI [0.016233, 0.117239] |
| delta IC | -0.055409, CI [-0.087395, -0.006787] |
| weighted ESS | 380.388 / 1,640 = 23.19% |
| raw adjacent median nondecreasing | 3 / 4 |
| weighted adjacent median nondecreasing | 2 / 4 |

Weighted five-band medians는 Strong Bear `-0.070452`, Bear `-0.041546`, Neutral `0.020571`, Bull `-0.019881`, Strong Bull `-0.043049`다. Selection weighting은 IC를 크게 약화시키고 monotonicity를 고치지 않는다. 다만 극심한 ESS 손실 때문에 이를 인과 보정이나 확장 표본 replication으로 해석하면 안 된다.

생산본과 재현본의 propensity rows, propensity seal, monthly output, summary, stage 해시는 5/5 일치했다. Git commit만 후속 label-coverage 변경 때문에 다르다.

## 3. Expanded ERI label coverage engineering

사전등록 계약:

- frozen Full Evidence seal SHA-256: `6e9fe8e59de0d4a8a12be8d2ac481a001ebe241b5f49d3ecf91194409be00ab5`
- frozen eligible rows SHA-256: `cb98912c5cb65e9c8909db0e4fdf6fac9cc92e6e757470a1dc2bbc5965860f5c`
- horizon: 63 trading sessions
- reverse DCF: `TURBO_DRIVER_ONE_DIMENSIONAL_REVERSE_DCF_V2`
- 허용 변경: label-side PIT annual account-name extraction coverage only

발견한 주요 누락은 DART 표준 계정명 `수익(매출액)`, `영업이익(손실)`을 기존 exact-label set이 인식하지 못한 것이었다. 부채는 첫 대차대조표에서 확인된 명확한 유동/비유동 차입금·사채·리스부채 exact alias만 fallback에 추가했다. 총액 행이 있으면 구성항목을 더하지 않아 중복 합산을 막았다.

| 빌드 | reverse expectations | potential t+63 labels | complete target metadata |
|---|---:|---:|---:|
| 기존 sealed pre-outcome | 1,834 | 1,673 | 12,953 |
| label coverage v1 | 4,846 | 4,408 | 26,780 |
| label coverage v2 | 4,931 | 4,484 | 27,051 |

최종 v2 coverage gate:

| 조건 | 실제 | 기준 | 판정 |
|---|---:|---:|---|
| reverse expectations | 4,931 | ≥ 5,000 | FAIL (-69) |
| potential t+63 labels | 4,484 | ≥ 4,500 | FAIL (-16) |
| Small reverse expectations | 1,044 | ≥ 1,000 | PASS |
| Mid reverse expectations | 1,884 | ≥ 1,400 | PASS |
| unique reverse issuers | 789 | ≥ 600 | PASS |
| potential-label signal months | 23 | ≥ 24 | FAIL (-1) |
| Large share of potential labels | 40.95% | ≤ 45% | PASS |

추가 계정 alias만으로 해결되지 않은 가까운 월의 exclusion은 두 개 미만의 유효 PIT annual, `equity + debt - cash <= 0`, reverse-DCF censoring, 비양의 signal Open 등 구조적 사유였다. 부채 무공시를 임의로 0으로 간주하거나 solver 범위를 변경하면 계약 위반이므로 중단했다.

최종 stage는 `outcome_stage_authorized=false`, `outcome_vault_opened=false`, `return_data_opened=false`, `value_data_opened=false`로 봉인됐다.

## 4. Source coverage와 무결성

11,193개 연간 스냅샷에서 확인된 source 조합:

| source 조합 | snapshots |
|---|---:|
| Arcana business-info + finance-comment + finance-statement | 8,248 |
| Arcana 3종 + MoatRader original filing | 2,620 |
| MoatRader original filing only | 320 |
| Arcana business-info only (나머지 source 미제공) | 5 |

따라서 Arcana 3종을 모두 읽은 스냅샷은 10,868개이고, MoatRader data-lake 원천 정기공시를 읽은 스냅샷은 2,940개다. 검증한 원천 파일은 35,549개이며 hash mismatch는 0, `original_source_files_modified=false`다.

## 5. 의사결정

1. 현재 Size 결과만으로 Evidence V2를 폐기하거나 Size-neutral score로 교체하지 않는다.
2. Selection weighting은 강한 selection-bias 경고로 유지하되 V2 primary 결과를 덮어쓰지 않는다.
3. expanded-label replication은 coverage gate 실패로 실행하지 않는다.
4. IR·산업자료 추가는 Evidence V3의 별도 pre-outcome 계약으로 다룬다. 현재 V2에는 섞지 않는다.
5. PER+PBR 및 기타 Value 지표를 우선 랭킹으로 사용하지 않는다.

## 6. 주요 산출물

- Size diagnostic: `data-lake/experiments/future-eri-v2-size-diagnostic-v25-full-semantic-v2-20260823`
- Selection weighting production: `data-lake/experiments/future-eri-v2-selection-weighting-v25-full-semantic-v2-20260823`
- Selection weighting reproduction: `data-lake/experiments/future-eri-v2-selection-weighting-v25-full-semantic-v2-repro-20260823`
- Expanded-label pre-outcome v2: `data-lake/experiments/future-eri-v2-label-replication-pre-outcome-v2-20260823`
- Preregistered contract: `docs/experiments/future-eri-v2-label-replication-contract-20260823.json`

