# 애널리스트 산업 보고서 파이프라인

MoatRader는 한국경제 컨센서스의 산업 보고서(`REPORT_TYPE=IN`)를 외부
reference-class evidence로 처리합니다. 회사 공시가 아니므로 산업 성장률·점유율·마진을
특정 회사의 MOAT 근거로 승격하지 않습니다. 산업 자료는 valuation driver의 범위,
시나리오, 업황과 위험을 보정하는 별도 lane으로만 들어갑니다.

## 구성

1. `collect hankyung-industry`가 보고서 목록과 PDF를 내려받아 immutable Bronze에 저장합니다.
2. `IndustryPdfAdapter`가 Synalyst `StructuredPdfParser` 0.2.15를 직접 호출합니다.
3. Synalyst의 page/node/table/cell/review/provenance 출력을 `CanonicalDocumentBundle`로 변환합니다.
4. canonical chunk를 deterministic atomic unit으로 분해하고 valuation-relevant unit만 선택합니다.
5. `valuation-evidence-requests.jsonl`은 `gpt-5.6-luna` valuation 분류기에 넘길 입력입니다.

다운로드 코드의 endpoint, `REPORT_IDX` fallback, 메타데이터 필드와 파일명 규칙은 Synalyst
collector에서 가져왔습니다. MoatRader 쪽에서는 응답 byte limit, PyMuPDF 구조 검증,
password/zero-page 거부, immutable revision 저장을 추가했습니다. Synalyst가 review 대상으로
표시한 header 없는 전체 페이지 layout grid는 semantic table로 공개하지 않고 reading-order
원문으로 되돌립니다. 이 규칙은 newsletter형 보고서에서 본문 전체가 하나의 표로 오인되어
evidence가 사라지는 문제를 막습니다.

## 기존 Synalyst PDF로 파싱 테스트

아래 명령은 네트워크와 downloader를 전혀 호출하지 않습니다. PDF 파일명의 첫
`REPORT_IDX`를 `reports.json`과 조인하고 문서별 결과를 만듭니다.

```powershell
$env:MOATRADER_SYNALYST_ROOT = "D:\Programming\python_example\Synalyst"

moatrader industry prepare `
  --pdf-root "D:\Programming\python_example\Synalyst\data-lake\bronze\consensus\hankyung\industry\2025\pdf" `
  --reports-json "D:\Programming\python_example\Synalyst\data-lake\bronze\consensus\hankyung\industry\2025\json\reports.json" `
  --synalyst-root $env:MOATRADER_SYNALYST_ROOT `
  --limit 5 `
  --maximum-evidence-units 24 `
  --output data-lake\silver\hankyung-industry-smoke
```

문서별 출력은 다음과 같습니다.

```text
HANKYUNG_IN_{REPORT_IDX}/
├─ bundle.json                       canonical AST + PIT metadata + provenance
├─ document.md                       사람이 확인하는 구조화 Markdown
├─ chunks.jsonl                      canonical semantic chunks
├─ industry-evidence-units.jsonl     deterministic valuation evidence candidates
└─ valuation-evidence-requests.jsonl LLM structured-classification requests
```

`manifest.json`에는 입력·성공·실패 수와 문서별 page/chunk/evidence count가 기록됩니다.
한 PDF가 깨져도 나머지 문서는 끝까지 처리합니다.

## 신규 PDF 수집

실제 수집 때만 bearer token이 필요합니다. token은 metadata나 URL에 기록하지 않습니다.

```powershell
$env:HANKYUNG_BEARER_TOKEN = "..."

moatrader collect hankyung-industry `
  --from 2025-01-01 `
  --to 2025-12-31 `
  --industry-code 005 `
  --max-reports 100 `
  --output data-lake\bronze
```

저장 경로는 `data-lake/bronze/hankyung-industry/{source_document_id}/versions/{sha256}`입니다.
산업 보고서는 synthetic ticker를 회사 universe CSV에 섞지 않으므로 collection 결과 JSON만
생성합니다.

## PIT 및 의미론 규칙

- `published_at`: `REPORT_DATE`의 Asia/Seoul 자정
- `available_at`: 초 단위 `REGISTER_DATE`
- `SourceType`: `INDUSTRY`
- `DocumentType`: `ANALYST_REPORT`
- 현재 사실·해석: `INDUSTRY_INTERPRETATION`
- 미래 전망: `FORECAST`
- 경제 범위: `INDUSTRY`

LLM 요청에는 산업 evidence를 issuer fact로 바꾸지 말라는 계약이 포함됩니다. 실제 DCF 숫자,
확률, CAP 기간 또는 공정가치는 LLM이 만들지 않고 후속 Python 검증기가 계산합니다.

## 검증 범위

`tests/test_hankyung_industry.py`는 메타데이터/PIT, API filter와 PDF 구조 검증, immutable
Bronze, downloader 미호출 prepare 경로, Synalyst 저장 1페이지 newsletter의 layout-table
fail-close, 4페이지 문서의 table/figure/page-bbox provenance를 검사합니다. 로컬 Synalyst
fixture가 없는 환경에서는 외부 fixture 테스트만 skip되고 단위 테스트는 계속 실행됩니다.
