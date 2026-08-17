# 구현 상태와 운영 경계

## 연결 완료

### Source/PIT/parser

- OpenDART와 SEC EDGAR 공식 원문 수집, rate limit/retry/size/path-traversal 방어, immutable Bronze SHA-256와 revision pointer
- DART XML/HTML, SEC HTML/iXBRL, IR HTML/PDF를 source-neutral `CanonicalDocumentBundle`로 변환
- section/paragraph/list/note/figure/table AST, rowspan/colspan, multi-header, 단위·기간·주석, inline XBRL `StructuredFact`, 원문 provenance
- IR PDF text/table parser, 선택적 OCR, figure crop/vision 후보와 visual ablation audit
- Hankyung 애널리스트 산업 보고서 수집, Synalyst 0.2.15 parser bridge, 산업 전용
  atomic valuation-evidence unit/request와 issuer-MOAT lane 격리
- timezone-aware 공개시각 filtering, scope-safe FY/YTD/TTM financial snapshot, 숫자 provenance와 hard-fail quality gate

### Frozen Evidence Sensor v1

- deterministic atomic segmentation과 독립 다수결 classification
- 원문용 evidence ID, 의미 dedupe용 claim ID, relation과 evidence ledger
- shuffle/duplicate/summary/format/boilerplate/node-order metamorphic gate
- 생성 요약의 score-bearing 재진입 차단과 canonical fact 기반 deterministic summary
- Evidence Sensor version/gate 상수, boss 9/full 30 회귀 fixture, claim-set Jaccard와 score-bearing presence extraction-set reproducibility
- scalar MOAT와 contextual strength는 주 랭킹이 아니라 회귀·진단 경로로 동결

### Expectation Analysis v1

- frozen sensor와 독립적인 valuation-only atomic selector/classifier
- `MOAT_NONE` forward fact를 포함하는 `ValuationDriverEvidence`; evidence 하나당 primary driver 하나, related driver는 diagnostic only
- scalar가 아닌 `CompetitiveAdvantageProfile`과 reference-class 기반 CAP range
- 보고/무형자산 조정 ROIC·ROIIC·재투자율을 병렬 계산하는 capital-allocation profile
- 성장–재투자–ROIIC를 연결하고 CAP/fade/stable state를 명시한 deterministic Economic FCFF
- assets-in-place, PVGO, CAP contribution, failure-adjusted value 분해
- downside/central/upside price-blind scenario와 confidence 기반 range widening
- Possible/Plausible/Probable assumption validation; numeric probability를 가장하지 않음
- 현재 가격에 맞는 성장·마진·ROIIC·CAP 조합 전체를 보존하는 Reverse DCF surface
- intrinsic lane 이후에만 price를 주입하는 타입·PIT invariant와 Expectation Gap 기본 screening
- 일반/성숙/플랫폼/distressed/금융/바이오 model router와 standalone biotech rNPV
- 기존 MOAT×DCF ranker는 명시적인 `--enable-legacy-moat-ranking` 진단 옵션으로만 실행

### Runner/audit

- 한 종목/복수/전체 universe 실행, 병렬 처리, 회사별 실패 격리, checkpoint/resume
- OpenAI structured output, task별 pinned model/effort/cap, atomic replay cache, raw response와 usage audit
- evidence와 valuation lane을 포함한 static-prefix/dynamic/source/output token budget 기록
- 5종목 초과 실행을 막는 계약-고정 preflight
- PIT backtest와 IC/Q5–Q1 도구는 존재하지만 새 Expectation Analysis 규칙을 과거 수익률에 맞춰 튜닝하지 않음

## 외부 데이터가 보장해야 하는 것

- 각 공시와 IR 자료의 실제 시장 공개 가능 시각인 timezone-aware `available_at`
- corporate action이 반영된 수정주가와 정확한 `price_as_of`
- 과거 각 시점의 investable universe와 상장폐지/거래정지 처리 데이터
- PIT reference class, TAM, WACC 구성요소, 업종별 회계/경제구조 분류
- downside/central/upside 숫자 가정의 출처·승인·버전 정책

이 조건이 빠지면 코드의 PIT 필터만으로 survivorship, stale-price, look-ahead 또는 corporate-action bias를 없앨 수 없습니다.

## Fail-closed 또는 아직 필요한 범위

- IR visual extractor는 기존 gold set에서 개선됐지만 unseen-document production gate와 정기 extraction-set 반복 감사가 더 필요
- DART/SEC taxonomy와 버전별 더 정교한 concept mapping 및 segment economics
- 금융회사 excess-return equity 엔진은 router만 있고 구현되지 않아 분석을 거부
- biotech rNPV는 standalone 엔진이며 universe expectation runner 자동 통합은 아직 없음
- Economic DCF 숫자 가정의 완전 자동 생성은 의도적으로 없음; evidence는 숫자를 직접 bump하지 않음
- Reverse DCF grid는 민감도 표면이며 연속 최적화나 유일한 implied solution을 주장하지 않음
- parser 원문/AST/Markdown 동시 비교 viewer와 더 큰 실제 공시 golden corpus
- scheduler, queue, database, auth, monitoring, 승인 workflow를 갖춘 상시 운영 API/UI

현재 결과는 연구·감사용이며 투자 권유나 미래 수익률 보장이 아닙니다.
