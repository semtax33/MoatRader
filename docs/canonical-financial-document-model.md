# Canonical Financial Document Model 설계

## 1. 해결하려는 문제

DART HTML/XML, SEC EDGAR HTML/iXBRL, IR PDF/PPT/HTML은 구조와 신뢰도가 다릅니다. 이 차이를 downstream MOAT/DCF 코드까지 전파하면 source가 하나 늘 때마다 분석 엔진을 다시 작성하게 됩니다. 이 모델은 adapter 아래에 canonical boundary를 만들고, 그 아래를 source-agnostic하게 유지합니다.

```text
Source-specific world
        ↓
     Adapter
════════════════════ canonical boundary
 CanonicalDocumentBundle
════════════════════
        ↓
Source-agnostic world
```

ANTLR 같은 grammar parser는 사용하지 않습니다. 입력은 정형 언어가 아니라 깨진 HTML, presentation wrapper, layout heuristic을 포함한 문서이므로 tolerant DOM parser와 deterministic tree transformation이 더 적합합니다.

## 2. 변경할 수 없는 불변조건

1. Bronze 원문은 수정하지 않습니다.
2. AST는 원문 구조를 정규화하지만 새로운 주장을 만들지 않습니다.
3. Evidence는 하나 이상의 canonical `node_id`와 `chunk_id`를 인용합니다.
4. Summary와 MOAT score는 기존 `evidence_id`만 재조합합니다.
5. 재무 계산은 `StructuredFact`와 명시적 assumptions를 사용합니다.
6. `available_at`보다 이른 signal에는 문서와 fact를 노출하지 않습니다.
7. 분류 실패한 visible content는 삭제하지 않고 `unknown_block`으로 남깁니다.
8. 모든 ID는 document ID, source locator, node kind, normalized content로 deterministic하게 생성합니다.
9. Markdown은 언제든 canonical 객체에서 재생성할 수 있어야 합니다.

최종 추적 체인은 다음과 같습니다.

```text
MOAT conclusion
  → evidence_id
  → chunk_id
  → node_id / fact_id
  → SourceRef(xpath 또는 page+bbox)
  → source_document_id + raw_sha256
```

## 3. 최상위 Bundle

`CanonicalDocumentBundle`은 한 AST에 모든 정보를 집어넣지 않습니다.

| 구성 | 용도 | 예시 |
|---|---|---|
| `metadata` | 문서 identity와 PIT 시간 | DART 접수번호, SEC accession은 `source_specific`에 저장 |
| `ast` | 사람이 읽는 문서 구조 | section, paragraph, table, note, figure |
| `facts` | 기계가 계산할 숫자 | XBRL revenue, operating income, shares |
| `assets` | 그림·차트·첨부 | IR chart image |
| `provenance` | 원문·파생 lineage | xpath, page, bbox, transform version |
| `quality` | parser gate | text/table/numeric retention, warnings |

AST Table과 StructuredFact는 중복이 아니라 서로 다른 역할입니다. 전자는 공시 표시와 주석을 보존하고 후자는 DCF 산술과 시계열에 사용합니다.

## 4. Metadata와 PIT

Top-level에는 source-neutral 필드만 둡니다.

```text
source_type
source_document_id
issuer_id / issuer_name / ticker / market
document_type
published_at
available_at
availability_precision / availability_source
reporting_period
raw_sha256
parser_version
source_specific
```

`available_at`은 필수 timezone-aware datetime입니다.

- DART: 접수/공개 시각
- SEC: acceptance datetime
- IR: 거래소 또는 회사 사이트의 실제 게시 시각

날짜만 아는 경우 `availability_precision=DAY`, 수집시각을 대용한 경우 `INFERRED`로 명시해야 합니다. 조용히 현지 자정을 정확한 시각으로 간주하면 안 됩니다.

정정공시는 `is_amendment`, `amends_document_id`로 원본을 연결합니다. 최신 정정값을 사용할 수 있는 시점도 정정공시의 `available_at` 이후입니다.

## 5. Canonical AST

노드 종류는 작게 유지합니다.

```text
section
paragraph
table
list / list_item
note
figure
page_break
unknown_block
```

모든 노드는 다음 정보를 가집니다.

```text
node_id
document order
raw_text / normalized_text
section_path
source_refs
classification(rule_id, confidence, reasons)
source-neutral attributes
```

`SectionNode`는 `title_raw`, `title_normalized`, `level`, `role`을 분리합니다. `사업의 내용`, `Business`, `Business Overview`는 원 제목을 유지하면서 모두 `role=BUSINESS`가 될 수 있습니다.

## 6. DOM에서 AST로

직접 `tag → node`로 매핑하지 않고 다음 두 단계를 사용합니다.

```text
DOM + context
  → HEADING / PARAGRAPH / TABLE / LIST / NOTE / FIGURE event
  → stack 기반 Section Tree
```

우선순위는 DROP, TABLE, LIST, FIGURE, explicit heading, inferred heading, NOTE, PARAGRAPH, WRAPPER 순입니다. 표 안의 짧고 굵은 `<th>`가 heading으로 오인되지 않도록 TABLE이 먼저 소비합니다.

Heading은 tag, 번호 패턴, bold, 길이, 문장 종결, table 내부 여부를 조합한 점수로 분류하고 이유를 남깁니다. 한국어 번호 체계인 로마자, 숫자, `가.`, `(1)`, `①`를 level 후보로 사용합니다.

`div`, `span`, `font`는 의미 타입이 아닙니다. block child가 있는 div는 wrapper이고 inline element는 부모 text에 녹입니다. 분류되지 않은 visible block은 버리지 않습니다.

## 7. 표 계약

표는 다음 정보를 함께 보존합니다.

```text
caption
table-level unit and period
section_path
rectangular rows/cells
multi-level column header paths
footnotes
source locator
```

`rowspan`과 `colspan`은 rectangular grid로 확장합니다. 각 복제 cell에는 `origin_row`, `origin_col`, `propagated`, 원 `source_rowspan/source_colspan`이 남으므로 표시와 원본을 동시에 복원할 수 있습니다.

숫자 cell은 raw string과 `Decimal numeric_value`를 함께 저장합니다. 괄호 숫자는 음수, `%`는 ratio로 파싱합니다. 애매한 값은 `numeric_value=null`로 두며 추측하지 않습니다.

Multi-header는 다음처럼 column path로 보존합니다.

```text
["2025", "매출"]
["2025", "영업이익"]
```

표가 chunk budget보다 크면 row group으로 나누되 caption, unit, period, headers, section path를 매 chunk에서 반복합니다. 한 row 자체는 자르지 않습니다.

## 8. StructuredFact

`StructuredFact`의 핵심 필드는 다음과 같습니다.

```text
fact_id
source concept / canonical concept
raw value / numeric Decimal
unit + scale + currency
instant 또는 duration period
consolidation scope
dimensions / segment / context_id
decimals
available_at / is_restated
statement_type
source_refs / derived_from_ids
```

초기 inline XBRL extractor는 source concept를 그대로 보존합니다. `FinancialSnapshotBuilder`가 제한된 alias map으로 Revenue, EBIT, CFO, Capex 등을 canonicalize합니다. production에서는 taxonomy version별 명시적 mapping table과 sign convention test가 추가되어야 합니다.

파생지표는 Python이 계산하고 input fact ID를 남깁니다.

- EBIT margin
- FCF = CFO - abs(Capex)
- FCF margin
- Revenue CAGR

ROIC에는 invested capital과 average balance 정의가 필요하므로 충분한 facts와 회계정책 mapping 없이 임의 계산하지 않습니다.

## 9. Semantic/LLM 계층

### Pass 1: Local Evidence

각 semantic chunk마다 회사 점수는 묻지 않고 명시된 경제적 사실만 추출합니다. 응답은 `EvidenceExtractionResult` JSON Schema를 따릅니다.

Evidence는 switching cost, network effect, cost/scale advantage, intangible assets, pricing power, retention, competitive threat, concentration, substitution, technology risk 등으로 분류합니다. `DISCLOSED_FACT`, `MANAGEMENT_CLAIM`, `ANALYST_INTERPRETATION`, `FORECAST`, `DERIVED_METRIC`을 분리해 출처 신뢰도를 구분합니다.

Python validator는 다음을 검사합니다.

- chunk ID 일치
- node ID 존재 및 해당 chunk 포함 여부
- raw quote가 실제 chunk substring인지
- 숫자 metric이 source numeral에 존재하는지
- enum/범위는 Pydantic schema로 검증

### Pass 2: Hierarchical Summary

Section summary는 evidence ID만 재조합합니다. 새로운 사실을 만들 수 없습니다. 대형 복합기업은 evidence의 `segment`를 유지한 뒤 segment MOAT를 revenue/EBIT exposure와 별도 결합해야 합니다.

### Pass 3: Company Scoring

최종 입력은 다음 3층입니다.

```text
L1 Structured Summary + Financial Snapshot
L2 Evidence Cards + Counterevidence
L3 selected raw chunks/tables
```

MOAT output은 0~10 점수, mechanism별 evidence ID, counterevidence, durability, confidence와 별도의 document coverage를 포함합니다. 점수가 양수인데 evidence mechanism이 없으면 schema가 거부합니다.

## 10. Token budget

고정 char 제한 대신 실제 model context에서 prompt reserve를 뺀 token budget을 사용합니다. 현재 fallback token counter는 보수적 heuristic이며 production LLM의 tokenizer를 `TokenCounter`로 주입해야 합니다.

`DynamicTokenBudgetAllocator`는 문서에 실제 존재하는 section role만 대상으로 minimum quota를 먼저 만족시키고, 남은 예산을 role weight, retrieval relevance, chunk cost로 배분합니다. 선택 후에는 document order로 되돌려 문맥을 유지합니다.

Retrieval relevance는 다음 경제적 질문별 top-k 결과를 넣는 것을 권장합니다.

- 고객이 공급자를 바꾸기 어려운 이유는 무엇인가?
- 신규 경쟁자의 진입을 막는 것은 무엇인가?
- 경쟁사보다 높은 가격이나 낮은 원가가 지속되는 이유는 무엇인가?
- 시장점유율 유지 근거는 무엇인가?
- 고객/공급자 집중 위험은 무엇인가?
- 해자를 파괴할 반대 증거는 무엇인가?

## 11. DCF와 screening

DCF assumptions와 source facts는 분리합니다. `DcfEngine`은 revenue growth, EBIT margin, tax, D&A, Capex, NWC, WACC, terminal growth, net debt, diluted shares로 unlevered FCF를 계산합니다. WACC가 terminal growth보다 크고 forecast 배열 길이가 같은지 schema가 검증합니다.

종목 screening은 최소 MOAT, margin of safety, model confidence, document coverage를 모두 통과한 후보만 대상으로 합니다. 기본 ranking 식은 코드에 공개되어 있으며, 다른 연구식으로 교체할 수 있습니다.

시장가격은 반드시 `price_as_of`, valuation은 `valuation_as_of`를 가져야 합니다. 같은 시점에서 실제 매매 가능했던 가격과 정보만 사용해야 합니다.

## 12. Data lake

```text
data-lake/
├─ bronze/
│  ├─ dart/
│  ├─ sec-edgar/
│  ├─ ir/
│  └─ analyst/
├─ silver/
│  ├─ canonical_ast/
│  ├─ structured_facts/
│  ├─ assets/
│  ├─ chunks/
│  └─ provenance/
└─ gold/
   ├─ evidence/
   ├─ summaries/
   ├─ company_dossiers/
   ├─ context_packs/
   ├─ moat/
   └─ dcf/
```

Bronze만 source별입니다. Silver부터 동일 schema를 사용합니다. 저장 파일에는 schema/parser/prompt/model version과 input SHA-256을 남겨야 합니다.

## 13. 새 adapter 구현 체크리스트

1. `detect`가 다른 adapter와 모호하지 않게 source를 식별합니다.
2. source document ID와 timezone-aware `available_at`을 만듭니다.
3. source-specific DOM/layout/XBRL intermediate representation을 파싱합니다.
4. section/paragraph/table/list/note/figure로 변환합니다.
5. 숫자 feed가 있으면 StructuredFact로 별도 보존합니다.
6. PDF/PPT는 page/slide와 bbox를 SourceRef에 넣습니다.
7. management claim과 hard fact를 구분할 정보가 있으면 attributes에 보존합니다.
8. raw visible text/table/numeric retention golden test를 추가합니다.
9. 모든 figure의 asset ID와 provenance가 실제 asset에 연결되는지 검증합니다.
10. 같은 입력에서 node/fact ID가 동일한지 확인합니다.

## 14. Production 전 추가 과제

- 대표 실제 DART XML corpus에서 첨부문서별 semantic role과 taxonomy 정교화
- SEC taxonomy/version별 concept mapping, units, dimensions, continuation handling
- IR PDF/PPT layout/OCR/chart adapter와 page-bbox viewer
- evidence contradiction/update relation의 semantic calibration 및 human review queue
- source reliability calibration
- golden corpus 15~30개와 parser retention gate
- model별 실제 tokenizer
- revision-aware fact store와 corporate action 조정 시장가격
- survivorship-free universe snapshot/상장폐지·corporate action 가격 데이터 공급과 벤치마크/IC 분석
