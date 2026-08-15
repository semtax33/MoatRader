# 붙여넣은 Canonical Financial Document Model 요구사항 감사

이 문서는 붙여넣은 0~59번 설계를 현재 코드와 대조한 결과입니다. `완료`는 실행 코드와 테스트가 있고, `부분`은 핵심 경로는 있으나 명시된 확장 또는 운영 검증이 남은 상태이며, `미구현`은 제품 범위가 실제로 남은 상태입니다.

| 요구사항 묶음 | 상태 | 현재 증거 | 남은 일 |
|---|---|---|---|
| 0 source-of-truth/provenance | 완료 | `CanonicalDocumentBundle`, AST/fact/asset provenance, evidence node validation, score evidence validation | UI 시각화만 별도 |
| 1 immutable Bronze 원문/SHA/versioning | 완료 | `BronzeFilingStore`, collector version, content-addressed versions, `metadata.json`, `sha256.txt`, `latest.json`; 실제 parser version은 Silver bundle에 기록 | object storage backend 선택 가능 |
| 2~3 DOM parse/normalize, wrapper/`br`/encoding | 완료 | lxml tree, drop rules, inline newline 보존, UTF-8/CP949/BOM | 실제 corpus별 wrapper rule 보강 지속 |
| 4~8 canonical AST/node ID/section hierarchy/path | 완료 | canonical models, stable IDs, HTML heading heuristic, DART `SECTION-n`, section path | DART 전 문서유형의 비표준 tag golden corpus 확대 |
| 9~14 table normalization/order/footnotes | 완료 | rowspan/colspan grid, multi-header, unit/period/numeric raw+parsed, footnote, document order | column별 복합 단위의 추가 실제 corpus 검증 |
| 15 AST validation metrics/gate | 완료 | text/table/numeric retention, paragraph/heading count, unknown count, duplicate ratio, configurable fail-closed gate | 실제 golden threshold 지속 보정 |
| 16 Silver AST/table 저장 | 부분 | run별 `bundle.json`, chunk/table은 bundle 안에 저장 | gzip/Parquet 공용 Silver warehouse export 미구현 |
| 17~18 cross-filing exact/near dedup/change | 완료 | `semantic.dedup`, numeric change 보존 | embedding 기반 선택적 calibration 미구현 |
| 19~22 semantic node/table row-group chunk/token | 완료 | semantic chunker, table metadata 반복, token counter 인터페이스 | 기본 counter는 heuristic이며 실제 model tokenizer calibration 필요 |
| 23~28 local evidence extraction/validation | 완료 | Structured Outputs, source-only prompt, node/numeric/quote validation, retry/repair | human review queue 미구현 |
| 29 canonical evidence dedup | 완료 | `evidence-clusters.json`에 canonical/supporting IDs를 보존하고 scoring dossier에는 canonical evidence만 반영 | 대표 corpus로 threshold calibration 필요 |
| 30 supports/contradicts/weakens/updates | 완료 | 다섯 typed relation을 deterministic similarity/direction/numeric rules로 생성 | semantic calibration corpus 필요 |
| 31~32 section hierarchical summary/citations | 완료 | LLM 재요약 없이 canonical fact를 positive/negative/KPI lane으로 분리하고 모든 claim에 evidence ID 유지 | human calibration 필요 |
| 33 company dossier | 완료 | `CompanyDossier`와 internal-reference validation | 없음 |
| 34 segment preservation/weighted score | 부분 | evidence `segment` 보존 | segment별 MOAT와 revenue/EBIT 가중 aggregation 미구현 |
| 35~36 economic-question retrieval | 완료 | 7개 MOAT 질문과 BM25/strength/reliability ranking | embedding hybrid는 선택적 후속 |
| 37~40 budget/minimum quota/L1-L2-L3 | 완료 | role weights/minimum quotas, counterevidence와 selected raw를 포함한 pack | 대표 corpus tuning 필요 |
| 41 structured Markdown | 완료 | metadata/table/evidence/provenance renderer | 없음 |
| 42~47 renderer ablation | 부분 | structured/hierarchical evidence pack 존재 | plain/generic/full renderer와 ablation runner 미구현 |
| 48~51 final score/evidence/confidence/coverage | 완료 | typed score, cited mechanism requirement, model confidence와 6개 coverage 분리 | 없음 |
| 52 Run Manifest | 완료 | signal/cutoff/model/parser/schema/renderer/prompt/token/hash/temperature | 호출별 audit도 별도 저장 |
| 53~54 modular pipeline/function structure | 완료 | adapters/canonical/semantic/evidence/retrieval/context/render/runner/financial 모듈 | 없음 |
| 55 phased acceptance/golden tests | 부분 | 50개+ 자동 테스트와 retention gate | 실제 어려운 공시 15~30개 golden corpus 미구축 |
| 56 JSON 관계 저장, Graph DB 회피 | 완료 | JSON/JSONL 관계와 artifact store | 없음 |
| 57 원문/AST/Markdown 디버깅 화면 | 미구현 | 파일 artifact는 각각 생성됨 | 3-pane linked viewer 필요 |
| 58 golden document set | 미구현 | 합성/샘플 fixtures만 존재 | 표·깨진 HTML·대형 복합기업 실제 corpus 필요 |
| 59 대형문서 구조적 압축 | 완료 | cross-filing/claim dedup, factor별 compact pack, on-demand raw lookup, numeric feature vector, delta replay, compression-invariance gate | 대형 실제 corpus 토큰·성능 calibration 필요 |
| DART/SEC API 수집/rate-limit/revision | 완료 | OpenDART/SEC 공식 API collector, immutable versions, safe PIT policy | scheduler/queue/monitoring은 서비스 계층 과제 |

## 이번 검토에서 새로 발견해 반영한 항목

1. DART 원문 endpoint는 HTML 한 파일이 아니라 ZIP이므로 ZIP 자체와 모든 member를 보존하고 primary XML/HTML만 downstream input으로 선택했습니다.
2. DART 목록에는 정확한 접수시각이 없어 일자 자정이나 다운로드시각을 사용하는 대신 KST 일자 종료로 보수 처리했습니다.
3. SEC acceptance timestamp는 실제 웹 공개 timestamp가 아니므로 기본 5분 지연을 명시적으로 기록했습니다.
4. SEC recent 배열만 읽으면 과거 공시가 빠지므로 `filings.files`의 추가 submissions JSON까지 기간과 lookback에 맞춰 읽습니다.
5. 정정공시가 수집 범위에 원본 없이 들어오면 `is_amendment=true`를 강제로 만들 수 없는 canonical 계약이므로 `reported_as_amendment`와 `UNRESOLVED`를 source-specific metadata에 보존합니다.
6. DART XML의 `SECTION-1/2/...`를 기존 leaf text로 처리하면 구조가 소실되므로 명시적 section hierarchy로 복원했습니다.

## 완료로 주장하지 않는 범위

이번 구현으로 공식 HTML/XML 원문 수집부터 기존 MOAT/DCF runner까지 연결되지만, 붙여넣은 장기 제품 구상 전체가 끝난 것은 아닙니다. 특히 실제 golden corpus, parser 비교 viewer, segment-weighted MOAT, renderer ablation, IR PDF/PPT/OCR, taxonomy별 정교한 XBRL mapping, scheduler/database/auth/monitoring은 아직 남아 있습니다.
