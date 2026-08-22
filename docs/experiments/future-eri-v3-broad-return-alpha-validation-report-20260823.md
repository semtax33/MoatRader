# Future ERI V3 broad-universe return validation

## 결론

사전등록한 broad historical confirmation은 `SUPPORTIVE_HISTORICAL_CONFIRMATION`으로 종료됐다. Frozen Full Evidence V2와 signal-session Open에서 시작하는 t+63 forward return의 월별 Spearman IC는 `0.069408`, Newey-West t-stat은 `4.907`, moving-block bootstrap 95% CI는 `[0.050164, 0.103589]`다.

Size, 1M/3-1/6-1/12-1 Momentum, Value, Growth, Quality를 각각 또는 공동 통제한 뒤에도 neutral IC는 양수였고 bootstrap 하단도 0보다 컸다. 따라서 과거 자료 안에서는 Evidence가 이들 팩터의 단순 proxy라는 설명으로 결과가 사라지지 않는다.

그러나 이 결과는 독립 OOS alpha 검증이 아니다. 기존 1,640개 Future ERI 표본에서 forward return을 secondary statistic으로 이미 관찰한 뒤 broad test를 설계했기 때문이다. 확정적인 alpha 주장은 금지하며, 2026-08-23 이후 새 신호의 live/OOS 검증이 필요하다.

## 고정 설계

- 사전등록 계약: `future-eri-v3-broad-return-alpha-validation-contract-20260823.json`
- 기준 universe: sealed Full/Core Evidence 공통 표본 37,014개
- Primary outcome: signal-session `Close/Open - 1`에 이후 62개 common trading session의 MARCAP `ChangesRatio`를 복리 연결한 Open-to-t+63-Close return
- signal 이전 overnight gap은 제외한다.
- exact signal Open/Close와 exact target Close를 요구하며, 이후 62개 세션 중 최소 50개 return 관측치를 요구한다.
- 월별 최소 20개 관측치, 같은 월·같은 complete-case 표본에서 raw와 neutral IC를 비교한다.
- 추론: Newey-West lag 3개월, moving-block bootstrap block 4개월, 10,000회, seed 42
- PER+PBR 우선순위 또는 랭킹은 사용하지 않았다. Value 지표는 병렬 민감도로 취급했다.
- Sector는 2026년 현재 KRX 업종으로 PIT가 아니므로 joint sensitivity에서만 사용했다.

## 표본 및 봉인

| 단계 | 관측치 | issuer | signal month |
|---|---:|---:|---:|
| Frozen Evidence baseline | 37,014 | 2,321 | 59 |
| Primary return eligible | 32,419 | 2,203 | 51 |
| 월별 IC 최소 20개 통과 | 32,206 | - | 27 |

Return 제외는 exact t+63 horizon 부재 3,916개, 비양수 signal/target price 561개, exact signal Open/Close 부재 118개였다. Broadness gate의 최소 20,000개 관측치, 1,000개 issuer, 12개월 조건을 모두 통과했다.

Outcome을 열기 전 통제변수 가용 관측치는 Size 34,410, 1M 36,830, 3-1 36,795, 6-1 36,680, 12-1 36,189, Value composite 29,798, Growth 23,605, Quality 30,197개였다.

## Raw 및 factor-neutral 결과

표의 `n`은 최소 월별 표본 조건을 통과한 complete-case 관측치 합계다. Retention은 같은 표본의 neutral IC / raw IC다.

| 테스트 | n | 월 | same-sample raw IC | neutral IC | neutral bootstrap 95% CI | retention |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 32,206 | 27 | 0.069408 | 0.069408 | [0.050164, 0.103589] | 100.0% |
| Size | 32,206 | 27 | 0.069408 | 0.065497 | [0.048516, 0.089082] | 94.4% |
| Momentum 1M | 32,201 | 27 | 0.069460 | 0.069417 | [0.050594, 0.102291] | 99.9% |
| Momentum 3-1 | 32,170 | 27 | 0.069370 | 0.062322 | [0.043097, 0.096233] | 89.8% |
| Momentum 6-1 | 32,071 | 27 | 0.069374 | 0.079596 | [0.054799, 0.106818] | 114.7% |
| Momentum 12-1 | 31,638 | 27 | 0.069520 | 0.068021 | [0.043714, 0.101835] | 97.8% |
| Momentum 1M+3-1+6-1+12-1 | 31,638 | 27 | 0.069520 | 0.077564 | [0.057163, 0.113053] | 111.6% |
| Value core composite | 27,711 | 25 | 0.070959 | 0.073654 | [0.050206, 0.110409] | 103.8% |
| Growth | 19,644 | 20 | 0.072341 | 0.062086 | [0.032292, 0.103894] | 85.8% |
| Quality | 25,970 | 23 | 0.056698 | 0.066317 | [0.055830, 0.080606] | 117.0% |
| Momentum+Growth+Quality | 18,326 | 19 | 0.047724 | 0.069654 | [0.062316, 0.088031] | 146.0% |
| All numeric | 18,324 | 19 | 0.047703 | 0.065464 | [0.037723, 0.093387] | 137.2% |
| All numeric + current Sector sensitivity | 18,324 | 19 | 0.047703 | 0.052372 | [0.026883, 0.120173] | 109.8% |

가장 보수적인 개별 경쟁 가설은 3-1 Momentum과 Growth였다. 각각 약 89.8%, 85.8%가 남았다. Full joint complete-case 표본에서는 raw IC 자체가 `0.047703`으로 낮지만, 같은 표본의 neutral IC는 `0.065464`였다. 이는 통제가 신호를 없앤 것이 아니라 complete-case 선택과 팩터 노출 제거 후 양의 성분이 남았음을 뜻한다.

## Value 병렬 민감도

Broad PIT 표본에서 안정적으로 구성 가능한 다섯 Value 지표를 동등한 병렬 민감도로 사용했다. PER+PBR 결합을 primary로 두지 않았다.

| Value 지표 | n | raw IC | neutral IC | neutral bootstrap 95% CI | retention |
|---|---:|---:|---:|---:|---:|
| B/M (PBR inverse) | 27,761 | 0.070938 | 0.068951 | [0.046111, 0.104203] | 97.2% |
| Sales/P (PSR inverse) | 27,741 | 0.070924 | 0.059408 | [0.032934, 0.094722] | 83.8% |
| Operating income/P | 20,485 | 0.102231 | 0.114552 | [0.091968, 0.144816] | 112.1% |
| EBIT/EV | 18,989 | 0.089126 | 0.099505 | [0.083755, 0.109974] | 111.6% |
| Assets/P | 27,770 | 0.070856 | 0.070294 | [0.048182, 0.109975] | 99.2% |

PER, P/FCF, PCR, EV/EBITDA, gross-profit/P, R&D/P, retained-earnings/P, NCAV/P 등 보조 계정이 필요한 지표의 13-way 비교는 기존 1,640개 PIT 완전표본 테스트가 담당한다. 이번 broad primary는 보조 normalized 계정의 coverage/lineage 때문에 표본을 다시 좁히지 않도록 sealed PIT annual snapshot의 core 계정만 사용했다.

## Return anchor 민감도

| Outcome | 월별 IC | bootstrap 95% CI |
|---|---:|---:|
| Primary adjusted Open-to-t+63-Close | 0.069408 | [0.050164, 0.103589] |
| Direct unadjusted target Close / signal Open | 0.057851 | [0.046558, 0.077826] |
| Next-session Open anchor sensitivity | 0.049808 | [0.030493, 0.083573] |

공시일 signal Open보다 한 세션 늦게 진입해도 IC 하단이 양수였다. 따라서 signal 이전 overnight gap 하나가 결과 전체를 만든 것은 아니다.

## Five-band 진단

| Evidence band | n | 평균 t+63 return | 중앙값 t+63 return | 양의 return 비율 |
|---|---:|---:|---:|---:|
| Strong Bear | 2,203 | -3.06% | -6.51% | 33.0% |
| Bear | 3,985 | 0.53% | -3.80% | 39.9% |
| Neutral | 10,773 | 1.43% | -2.93% | 42.7% |
| Bull | 10,124 | 3.76% | -0.98% | 47.2% |
| Strong Bull | 5,334 | 2.50% | -2.50% | 43.8% |

Strong Bull minus Strong Bear 중앙값 spread는 `+4.02%p`다. 다만 Strong Bull이 Bull보다 평균·중앙값·승률 모두 낮아 5-band 완전 단조성은 실패했다. 따라서 결과는 연속 rank IC에는 우호적이지만 `Strong Bull`을 별도 최상위 투자 bucket으로 주장하는 근거는 아니다.

## 1,640 ERI 표본 selection 비교

| 표본 | n | issuer | 월 | raw IC | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Broad return eligible | 32,419 | 2,203 | 51 | 0.069408 | [0.050164, 0.103589] |
| Broad, final ERI와 동일한 19개월 | 23,640 | 2,187 | 19 | 0.081458 | [0.066166, 0.097551] |
| Final ERI 1,640 intersection | 1,640 | 279 | 19 | 0.124808 | [0.083355, 0.175478] |
| Common-security-only sensitivity | 32,388 | 2,199 | 51 | 0.069635 | [0.050363, 0.103967] |

동일 19개월 broad IC `0.081458`과 final ERI IC `0.124808`의 차이는 `+0.043350`이다. 전체 차이 `+0.055401` 중 달력 구성 차이는 `+0.012051`, 동일월 내 final-ERI selection 증분은 `+0.043350`이다. 즉 broad에서도 관계는 살아 있지만, 1,640 표본은 효과 크기를 명확히 증폭시킨다. 이 동일월 분해와 common-security sensitivity는 primary 결과를 본 뒤 추가한 post-primary diagnostic이며 독립 확인으로 간주하지 않는다.

Final ERI 표본은 broad보다 평균 log market cap이 `26.322` 대 `25.893`으로 크고, Growth·Value·Quality coverage도 거의 100% 대 broad의 약 61%·86%·81%다. 따라서 reverse-valuation/financial-data complete-case selection이 신호가 잘 작동하는 기업을 선별했을 가능성을 배제할 수 없다.

## 무결성 및 재현성

- Outcome을 열기 전에 37,014개 control row를 별도 봉인했다.
- 사용된 원천 정기공시 30,607건과 MARCAP 7개 파일의 SHA-256을 검증했다.
- 원본 DART/Arcana/MoatRader 파일은 읽기 전용으로 사용했으며 수정하지 않았다.
- LLM/API 호출 없이 frozen Evidence와 로컬 PIT/price 자료만 사용했다.
- control, return, evaluation을 독립 경로에 재실행했고 모든 파일의 SHA-256이 일치했다.
- 최종 evaluation manifest: `data-lake/experiments/future-eri-v3-broad-return-validation-v2-20260823/build-manifest.json`
- 최종 summary: `data-lake/experiments/future-eri-v3-broad-return-validation-v2-20260823/broad-return-validation-summary.json`
- selection diagnostic: `data-lake/experiments/future-eri-v3-broad-return-validation-v2-20260823/broad-vs-final-eri-selection-comparison.json`

## 판정 및 다음 단계

1. 과거 broad universe에서 Evidence와 t+63 return의 관계는 확인됐다.
2. Size, short/long Momentum, Value, Growth, Quality, joint control로 지워지지 않았다.
3. Strong Bull band의 비단조성과 final ERI selection amplification은 중요한 경고다.
4. 이 결과로 랭킹·포트폴리오·실전 alpha 주장을 열지 않는다.
5. 다음 결정적 검증은 2026-08-23 이후 signal을 정의 변경 없이 누적하는 live/OOS test다.
