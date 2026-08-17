# 구현 상태와 운영 경계

## 연결 완료

- OpenDART `list.json`/`corpCode.xml`/`document.xml` 자동 수집
- SEC `data.sec.gov/submissions` discovery와 `Archives/edgar/data` primary HTML/complete submission 자동 수집
- source별 rate limit, HTTP/DART transient retry, 응답·ZIP size limit, ZIP path traversal 방어
- immutable Bronze version, 원문·metadata·파일별 SHA-256, revision/latest pointer, runner universe CSV 생성
- DART `SECTION-n` XML hierarchy 복원과 원문 ZIP 내 모든 첨부문서 보존
- DART, SEC EDGAR, IR HTML/iXBRL → source-neutral `CanonicalDocumentBundle`
- section/paragraph/list/note/figure/table AST, rowspan/colspan, multi-header, 단위/기간/주석, provenance
- inline XBRL `StructuredFact`와 PIT financial snapshot
- cross-filing chunk exact/near dedup 및 숫자 변경 보존
- deterministic atomic segmentation → 항목별 LLM classification → validation/repair → reliability calibration
- 원문용 evidence ID와 의미 dedupe용 canonical claim ID/set
- SUPPORTS/WEAKENS/CONTRADICTS/UPDATES/DUPLICATES evidence relation
- canonical fact와 Evidence ID만 재조합하는 deterministic section summary → company dossier → 경제적 질문 BM25 retrieval
- generated summary가 완전히 제외된 scoring dossier, factor별 compact claim pack, on-demand raw provenance, coverage 기록
- OpenAI Responses API/Pydantic Structured Outputs transport, retry, usage audit, `store=false`
- compact Structured Outputs와 task별 cap을 쓰는 atomic evidence 분류: pinned `gpt-5.6-luna` (`gpt-5-nano`는 선택적 문장 요약용 예약 설정)
- `MARKET_SHARE`/`CUSTOMER_RETENTION`/`MARGIN_STABILITY`/`COST_ADVANTAGE`/`COUNTER` observable-anchor gate와 issuer-owned product scope guard
- IR visual score-bearing 9건 A/B/C 수동 골드, 독립 3표 재판정, production classifier 6표 재현성 audit
- final-score LLM 호출이 없는 canonical claim Python reducer
- LLM 비의존 Python unlevered DCF, assumptions hash/as-of manifest
- 한 종목/복수 종목/전체 universe 실행, 병렬 처리, 회사별 실패 격리, chunk/section resume
- MOAT/DCF/신뢰도/coverage screening과 재랭킹
- PIT backtest, 종목별 거래 가능 여부·보유 잠금, mark-to-market, turnover/거래비용/슬리피지/용량, 보수적 상장폐지 청산, benchmark 초과수익
- raw/sector/factor-neutral IC, 비중첩 Q5-Q1 평가와 반복·입력순서 변경 MOAT 재현성 gate
- fresh experiment 범위 atomic-evidence replay와 원문 기반 stable evidence identity
- LLM 제안값과 독립적인 deterministic mechanism score/durability/counterevidence penalty
- 구조적 증거 생략을 삭제로 취급하지 않는 PIT evidence ledger와 supersede/retract 이력
- shuffle/duplicate/summary/format/boilerplate/node-order metamorphic zero-tolerance gate
- claim/factor score 불변 및 counterevidence recall 100%를 강제하는 compression-invariance gate
- Python financial feature vector, provenance 포함 valuation summary, cached/cache-write token audit
- evidence·claim Jaccard를 포함한 3~5종목 반복·인접시점 preflight 강제

## 외부 데이터가 반드시 보장해야 하는 것

- 각 공시의 시장 공개 가능 시각인 timezone-aware `available_at`
- corporate action이 반영된 수정주가와 정확한 `price_as_of`
- 과거 각 시점의 전체 investable universe 구성
- 상장폐지일까지의 가격 또는 명시적인 청산 가격
- DCF 가정의 출처·승인·시점 정책

이 조건이 빠지면 코드가 PIT 필터를 수행해도 survivorship bias, stale-price bias 또는 corporate-action 오류를 제거할 수 없습니다.

## 아직 별도 adapter/서비스가 필요한 범위

- DART 명시적 XML taxonomy와 SEC taxonomy/version별 정교한 concept mapping
- IR PDF/PPT production adapter, chart/figure semantic extraction, page/slide bbox viewer (`ir-visual-ablation-v1`의 vision full semantics 15/30, 실제 score-bearing route 3/7로 아직 gate 미통과)
- 실제 모델 tokenizer와 대표 corpus 기반 prompt/model calibration
- parser 원문/AST/Markdown 동시 비교 viewer와 15~30개 실제 공시 golden corpus
- segment별 MOAT 산출 및 revenue/EBIT 가중 company score
- plain/generic/full-context renderer와 체계적인 ablation harness
- 역사적 universe 재구성과 기업행동·상장폐지 golden corpus, 다요인 노출 열 공급
- scheduler, queue, database, auth, monitoring을 갖춘 상시 운영 API/UI

현재 runner의 입력 경계는 DART XML/HTML, SEC HTML/iXBRL, IR HTML과 metadata JSON입니다. PDF/PPT를 `source=IR`로 넣으면 안 되며, 먼저 전용 adapter를 구현해야 합니다.
