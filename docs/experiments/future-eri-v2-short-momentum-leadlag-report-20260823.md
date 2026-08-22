# Future ERI V2 Short-Momentum Neutralization and Lead-Lag Report

기준일: 2026-08-23  
역할: 사전등록된 Momentum 경쟁가설 및 lead-lag 진단. 투자 랭킹 또는 V2 primary retest가 아니다.

## 결론

- 3-1과 6-1은 기존 12-1보다 Evidence→Future ERI IC를 더 많이 약화시켰다.
- 그러나 3-1, 6-1, 3+6+12 joint를 통제한 neutral IC는 각각 raw IC의 약 81%, 80%, 79%가 남았고 bootstrap CI도 0보다 높았다.
- 따라서 Full Evidence V2를 단기 가격 Momentum의 단순 proxy로 보기는 어렵다. 다만 약 20%는 최근 3~6개월 가격 움직임과 공유될 수 있다.
- Evidence와 공시 직전 21거래일 수익률의 월별 Spearman 상관은 0.1132였다. 일부 정보가 공시 전에 가격에 반영되거나 Evidence와 가격이 같은 사업개선 정보를 동시에 반영할 가능성은 남는다.
- 사전등록된 `Evidence → 이후 rolling 1M Momentum rank 상승` lead-lag 가설은 실패했다.
- 보조지표인 Evidence→누적 forward return IC는 t+21/42/63 모두 양수였다. 이는 향후 누적 수익률 관계를 보여주지만, Momentum rank가 지속 상승한다는 경로 또는 인과적 `Evidence → ERI → Momentum` mediation을 입증하지 않는다.

## 1. 사전등록 정의

모든 과거 Momentum control은 공통 KOSPI/KOSDAQ 거래일 달력을 사용하고 신호일을 엄격히 제외했다. MARCAP `ChangesRatio`를 복리로 결합했으며 예상 거래일의 80% 이상이 있을 때만 값을 만들었다.

| Control | 거래일 창 | 최근 구간 제외 | 실제 수익률 구간 |
|---|---:|---:|---:|
| 1M / PreFilingReturn21 | 21 | 0 | 신호 전 21거래일 |
| 3-1 | 63 | 21 | 63번째~22번째 직전 거래일, 42개 |
| 6-1 | 126 | 21 | 126번째~22번째 직전 거래일, 105개 |
| 12-1 exact | 252 | 21 | 252번째~22번째 직전 거래일, 231개 |

Primary joint는 3-1+6-1+12-1이고, 1M까지 포함한 네 팩터 joint는 추가 민감도다. 월별 rank-normal Evidence를 rank-normal control에 residualize했으며, raw와 neutral IC는 항상 동일 complete-case 표본에서 계산했다.

최초 실행은 Momentum 값을 계산하기 전에 pre-outcome feature 1,673개와 최종 Future ERI label 1,640개를 혼동한 계약 검증에서 즉시 중단됐다. 결과를 열기 전에 두 표본 수를 분리하는 amendment를 기록했다.

## 2. Pre-outcome control coverage

| Control | 1,673개 중 유효값 |
|---|---:|
| 1M | 1,673 |
| 3-1 | 1,671 |
| 6-1 | 1,671 |
| 12-1 exact | 1,669 |

이 단계에서 Future ERI 값과 미래 가격수익률은 열지 않았다. 신호일 수익률 사용은 0이고 원천 파일 수정도 0이다.

## 3. Future ERI Momentum neutralization

| Test | N | Raw IC | Neutral IC | Neutral IC bootstrap CI | Retention |
|---|---:|---:|---:|---:|---:|
| 1M | 1,640 | 0.104632 | 0.096913 | [0.078255, 0.158051] | 92.62% |
| 3-1 | 1,638 | 0.102924 | 0.083255 | [0.056005, 0.138865] | 80.89% |
| 6-1 | 1,638 | 0.102924 | 0.082460 | [0.048313, 0.138829] | 80.12% |
| 12-1 exact | 1,636 | 0.103467 | 0.098119 | [0.069216, 0.160011] | 94.83% |
| 3-1 + 6-1 + 12-1 | 1,636 | 0.103467 | 0.081920 | [0.055984, 0.141589] | 79.18% |
| 1M + 3-1 + 6-1 + 12-1 | 1,636 | 0.103467 | 0.082364 | [0.067399, 0.131571] | 79.60% |

기존 달력일 12-1 통제 retention은 약 91.2%였고, 이번 exact-session 12-1은 94.8%다. 3-1과 6-1의 약 80%가 더 중요한 보수적 결과다.

Lead-lag authorization에 필요한 3-1, 6-1, 3+6+12 joint의 retention 70% 이상, neutral IC bootstrap 하단 0 초과, 유효월 12개 이상 조건은 모두 통과했다.

### 공시 전 가격반응

Evidence와 PreFilingReturn21의 월별 Spearman 평균은 `0.113201`, moving-block bootstrap CI는 `[0.037575, 0.181401]`, Newey-West t는 `3.265`다.

이는 공시 Evidence가 완전히 새로운 가격 비반영 정보라고 단정할 수 없다는 경고다. 다만 1M을 통제한 neutral ERI IC가 92.6% 남기 때문에 공시 전 가격반응만으로 Evidence→Future ERI 관계 전체가 설명되지는 않는다.

## 4. Gated future-Momentum lead-lag

Future Momentum은 각 horizon에서 끝나는 rolling 21거래일 수익률의 동일 signal-month complete-case percentile rank로 정의했다. Primary outcome은 이 rank와 공시 전 21거래일 rank의 차이다. 누적 forward return은 secondary다.

| Horizon | Evidence→Δ Momentum rank IC | Bootstrap CI | Strong Bull−Strong Bear median Δ rank | Evidence→누적 forward return IC | Bootstrap CI |
|---|---:|---:|---:|---:|---:|
| t+21 | -0.002027 | [-0.052091, 0.045129] | -0.029181 | 0.113375 | [0.073671, 0.155438] |
| t+42 | -0.038072 | [-0.080298, 0.018550] | -0.123894 | 0.136353 | [0.118618, 0.160887] |
| t+63 | -0.083763 | [-0.129167, -0.032165] | -0.133912 | 0.133813 | [0.089933, 0.187821] |

사전등록 gate는 다음 두 조건을 각각 3개 horizon 중 2개 이상 요구했다.

- Δ Momentum rank IC bootstrap 하단 > 0: 실제 0/3
- Strong Bull−Strong Bear median Δ rank > 0: 실제 0/3

따라서 lead-lag gate는 FAIL이다. 특히 t+63의 Δ rank IC는 유의하게 음수다. Strong Bear의 rolling 1M rank가 낮은 출발점에서 회복하고 Strong Bull의 단기 rank가 높은 출발점에서 정상화되는 단기 mean-reversion 성분이 섞였을 가능성이 있다. 이 설명은 진단적 추론이며 인과 결론이 아니다.

반면 누적 forward return IC는 세 horizon 모두 양수다. Evidence가 이후 누적 가격성과와 연결될 가능성은 보이지만, 이것을 `Momentum rank 상승` 또는 `Expectation revision이 Momentum을 매개했다`고 바꿔 말하면 안 된다.

## 5. 허용되는 표현

현재 결과로 허용되는 표현:

> Full Evidence V2와 Future ERI의 관계는 3-1·6-1 단기 Momentum 통제 후에도 약 80% 남는다. 따라서 장·단기 가격 Momentum의 단순 proxy는 아니지만, 약 20%의 공유 정보와 공시 전 가격반응은 존재한다.

허용되지 않는 표현:

- Evidence와 가격 Momentum은 완전히 독립적이다.
- Evidence가 이후 Momentum rank 상승을 선행한다.
- `Evidence → ERI → Momentum` 인과 mediation이 검증됐다.
- 이 결과를 종목 랭킹이나 매매 신호로 사용한다.

## 6. 재현성과 산출물

생산본과 독립 재현본의 파일 해시가 모두 일치했다.

- pre-outcome controls: 3/3
- ERI neutralization: 4/4
- future-Momentum lead-lag: 5/5

주요 산출물:

- Contract: `docs/experiments/future-eri-v2-short-momentum-leadlag-contract-20260823.json`
- Pre-outcome controls: `data-lake/experiments/future-eri-v2-short-momentum-controls-pre-outcome-v1-20260823`
- ERI neutralization: `data-lake/experiments/future-eri-v2-short-momentum-neutralization-v1-20260823`
- Lead-lag: `data-lake/experiments/future-eri-v2-future-momentum-lead-lag-v1-20260823`

