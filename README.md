# MoatRader Canonical Financial Document Model

DART, SEC EDGAR, IR처럼 포맷이 다른 금융 문서를 하나의 source-neutral 계약으로 바꾸고, 구조화 Markdown과 LLM evidence 요청을 만드는 Python 기반입니다.

핵심 원칙은 다음과 같습니다.

- Markdown은 저장 원본이 아니라 `CanonicalDocumentBundle`의 출력 형식입니다.
- AST는 사람이 읽는 문서 구조, `StructuredFact`는 DCF 등 계산용 숫자입니다.
- 모든 AST node, fact, asset, evidence는 원문 위치까지 역추적할 수 있어야 합니다.
- `available_at`은 timezone-aware 필수값이며 PIT(point-in-time) 필터에 사용합니다.
- LLM은 구조 복원이나 DCF 산술을 하지 않습니다. LLM의 첫 역할은 semantic chunk에서 evidence를 추출하는 것입니다.

## 현재 구현 범위

- Source-neutral `CanonicalDocumentBundle`
  - `DocumentMetadata`
  - `DocumentAST`
  - `StructuredFact[]`
  - `DocumentAsset[]`
  - `ProvenanceIndex`
  - parser 품질 지표
- OpenDART 공시검색/원문 ZIP 및 SEC submissions/Archives 원문 자동 수집
- immutable content-addressed Bronze 저장, SHA-256, revision/latest pointer, rate limit/retry
- DART XML/HTML, EDGAR HTML/iXBRL, IR HTML adapter와 adapter registry
- UTF-8, BOM, 선언 charset, CP949 fallback 디코딩
- heading heuristic 및 section tree/role 복원
- `rowspan`/`colspan` 확장, multi-level header, 단위, 기간, 주석 보존
- inline XBRL context/fact 추출
- semantic chunking, 큰 표의 row-group chunking, cross-filing exact/near dedup과 숫자 변경 보존
- 구조화 Markdown renderer
- OpenAI Responses API Structured Outputs 기반 evidence/summary/MOAT 실행, retry와 deterministic schema repair
- 근거 ID·node ID·숫자·인용문 validator와 chunk/section 단위 재개 체크포인트
- 경제적 질문별 BM25 검색과 L2/L3 context budget pruning
- L1 summary / L2 evidence / L3 raw source의 Company Evidence Pack
- PIT financial snapshot, Python 파생지표, deterministic unlevered DCF
- MOAT·DCF 할인·신뢰도·coverage 기반 종목 필터/랭커
- 단일/복수/전체 유니버스 병렬 실행과 종목별 실패 격리
- Run manifest, 호출 사용량, input hash 및 JSON Schema 출력

PDF/PPT IR은 canonical 계약과 provenance 필드가 준비되어 있지만 layout parser는 다음 adapter로 구현해야 합니다. OCR/차트 해석을 AST에 억지로 넣지 않고 `IRPdfAdapter` 위에서 해결하는 구조입니다.

## 처리 흐름

```text
OpenDART API ── immutable Bronze ── DART XML/HTML ── DartAdapter ───┐
SEC submissions + Archives ──────── SEC HTML/iXBRL ─ EdgarAdapter ──┼─ CanonicalDocumentBundle
IR HTML/PDF/PPT ─────────────────── IR HTML ──────── IRAdapter ────┘       ├ AST (사람이 읽는 구조)
                                        ├ Structured Facts (계산 가능한 숫자)
                                        ├ Assets
                                        └ Provenance
                                                ↓
                                      Semantic Chunker
                                                ↓
                                  Local Evidence Extraction (LLM)
                                                ↓
                                Evidence validation / dedup / summary
                                                ↓
                             L1 + L2 + L3 Company Evidence Pack
                                      ┌─────────┴─────────┐
                                      ↓                   ↓
                                MOAT scoring        Python DCF
                                      └─────────┬─────────┘
                                                ↓
                                PIT value + quality ranking
```

상세 계약과 확장 규칙은 [Canonical 모델 설계](docs/canonical-financial-document-model.md)를 참고하십시오.
현재 연결 범위와 외부 데이터 책임은 [구현 상태와 운영 경계](docs/implementation-status.md)에 정리되어 있습니다.

## 실행

AGENTS.md에 지정된 Windows 환경에서는 다음 인터프리터를 사용합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& D:\Programming\python_example\Arcana\.venv-llama\Scripts\Activate.ps1
python -m pip install -e .
```

LLM까지 실제 실행할 때는 OpenAI 의존성과 API 키를 준비합니다.

```powershell
python -m pip install -e ".[llm]"
$env:OPENAI_API_KEY = "..."
```

## DART·SEC 원문 자동 수집

OpenDART 인증키는 명령행 인자로 받지 않아 shell history나 결과 metadata에 남지 않게 했습니다.

```powershell
$env:DART_API_KEY = "발급받은-40자리-키"

moatrader collect dart `
  --stock-code 005930 `
  --from 2024-01-01 `
  --to 2026-08-14 `
  --report-kind annual `
  --report-kind semiannual `
  --report-kind quarterly `
  --output data-lake\bronze
```

SEC는 공식 정책에 따라 애플리케이션/회사명과 연락 가능한 이메일이 들어간 `User-Agent`가 필수입니다. 기본 요청률은 초당 5회로 공식 상한인 초당 10회보다 낮습니다.

```powershell
$env:SEC_USER_AGENT = "MoatRader your-email@example.com"

moatrader collect sec `
  --ticker-file data\us-universe-tickers.txt `
  --from 2024-01-01 `
  --to 2026-08-14 `
  --output data-lake\bronze
```

두 명령 모두 `data-lake\bronze\collected-universe.csv`를 갱신합니다. 이 CSV는 그대로 runner에 넣을 수 있으며, DCF와 screening까지 수행하려면 가격·가격시각·DCF 가정 열을 별도로 보강해야 합니다.

```powershell
moatrader moat run `
  --universe data-lake\bronze\collected-universe.csv `
  --as-of 2026-08-14T16:00:00+09:00 `
  --run-id collected-20260814 `
  --dry-run
```

기존 Bronze에서 manifest만 다시 만들 수도 있습니다.

```powershell
moatrader collect manifest `
  --bronze-root data-lake\bronze `
  --output data-lake\bronze\collected-universe.csv
```

원문 저장 계약, revision 처리, PIT 시각 정책과 운영 옵션은 [DART/SEC 수집기](docs/ingestion.md), 붙여넣은 전체 설계의 구현 여부는 [요구사항 감사표](docs/requirements-audit.md)를 참고하십시오.

## 한국 주식 PIT·TTM DCF 입력 준비

`prepare_kr_dcf_manifest.py`는 평가일에 이용 가능했던 OpenDART 정기보고서만 선택합니다. 분기·반기 손익과 현금흐름은 OpenDART의 누적금액을 사용하여 `직전 FY + 당기 YTD - 전년 동기 YTD`로 TTM을 만들고, 현금·부채·운전자본은 최신 보고서의 재무상태표 시점값을 사용합니다. 접수번호에는 시각이 없으므로 접수일 다음 날 00:00부터 이용 가능한 것으로 보수적으로 처리합니다.

```powershell
python scripts\prepare_kr_dcf_manifest.py `
  --universe data\universe.csv `
  --collected-manifest data-lake\bronze\collected-universe.csv `
  --output data-lake\date-inputs\2025-08-31 `
  --as-of 2025-08-31T23:59:59+09:00
```

`--year`를 생략하면 최신 PIT 보고서 기준으로 직전 3개 완료 사업연도를 자동 선택합니다. 명시적으로 고정하려면 `--year 2022 --year 2023 --year 2024`처럼 반복합니다.

- `dcf-inputs\TICKER.json`: TTM 숫자, 산식, 사용한 접수번호·공시시점·재무제표 범위, annual history, 입력 hash
- `assumptions\TICKER.json`: deterministic DCF 엔진 입력
- `dcf-audit.csv`: 종목별 PIT·TTM 요약과 입력 경로/hash
- `exclusions.csv`: 미래공시, TTM 구성요소 부족, 금융사 모델 불일치 등 제외 사유

분기 누적금액이 없거나 동일한 `CFS`/`OFS` 범위의 전기 연간·전년 동기 보고서를 확보하지 못하면 분기값을 임의로 혼합하지 않고 해당 DCF를 제외합니다. 현재 한국 주식 준비기는 12월 결산을 전제로 하며 이 가정은 입력 감사정보에 기록됩니다.

## 한 종목·여러 종목·전체 유니버스 MOAT 실행

유니버스 CSV는 공시 문서 한 개당 한 행입니다. 같은 ticker를 여러 행에 쓰면 DART, EDGAR, IR 등 여러 문서를 한 회사 dossier로 합칩니다. 예시는 `examples\universe.csv`에 있습니다.

필수 열은 `ticker,source,input,metadata`이고, 선택 열은 `issuer_id,issuer_name,current_price,price_as_of,dcf_assumptions`입니다. `source`는 `DART`, `SEC`/`EDGAR`/`SEC_EDGAR`, `IR` 중 하나이며 모든 timestamp에는 timezone offset이 있어야 합니다. 자세한 계약은 [Universe manifest](docs/universe-manifest.md)를 참고하십시오.

먼저 API 호출 없이 문서 파싱과 PIT 필터, DCF 입력, LLM 요청 생성을 점검합니다.

```powershell
moatrader moat run `
  --universe examples\universe.csv `
  --ticker SAMPLE `
  --as-of 2025-05-16T00:00:00+09:00 `
  --run-id sample-dry `
  --dry-run
```

한 종목 실 실행:

```powershell
moatrader moat run `
  --universe examples\universe.csv `
  --ticker SAMPLE `
  --as-of 2025-05-16T00:00:00+09:00 `
  --run-id sample-live
```

여러 종목은 쉼표 목록이나 반복 옵션을 사용합니다.

```powershell
moatrader moat run `
  --universe data\kr-universe.csv `
  --tickers 005930,000660,035420 `
  --as-of 2026-08-14T16:00:00+09:00 `
  --run-id kr-selected-20260814 `
  --workers 3
```

`--ticker`와 `--tickers`를 생략하면 CSV의 전체 종목을 실행합니다.

```powershell
moatrader moat run `
  --universe data\kr-universe.csv `
  --as-of 2026-08-14T16:00:00+09:00 `
  --run-id kr-all-20260814 `
  --workers 3
```

중단되었거나 일부 종목이 실패한 run은 동일한 입력과 run ID로 재개합니다. 완료된 회사와 chunk/section 체크포인트는 다시 호출하지 않습니다.

```powershell
moatrader moat run `
  --universe data\kr-universe.csv `
  --as-of 2026-08-14T16:00:00+09:00 `
  --run-id kr-all-20260814 `
  --workers 3 `
  --resume
```

상태와 기본 랭킹을 확인하거나 임계값을 바꾸어 재랭킹할 수 있습니다.

```powershell
moatrader moat status --run-dir data-lake\gold\runs\kr-all-20260814

moatrader screen rank `
  --run-dir data-lake\gold\runs\kr-all-20260814 `
  --minimum-moat-score 6.5 `
  --minimum-margin-of-safety 0.25 `
  --minimum-model-confidence 0.60 `
  --minimum-document-coverage 0.60
```

LLM은 작업별로 라우팅됩니다. 증거 추출과 섹션·문장 요약은 기본 `gpt-5-nano`, 최종 MOAT scoring은 기본 `gpt-5.6-luna`를 사용합니다. 각각 `--summary-model`/`MOATRADER_SUMMARY_MODEL`, `--moat-model`/`MOATRADER_MOAT_MODEL`로 바꿀 수 있습니다. DCF는 LLM을 호출하지 않고 Python의 deterministic unlevered DCF 엔진으로 계산합니다.

```powershell
moatrader moat run `
  --universe data\kr-universe.csv `
  --as-of 2026-08-14T16:00:00+09:00 `
  --run-id kr-all-20260814 `
  --summary-model gpt-5-nano `
  --moat-model gpt-5.6-luna
```

최종 결과는 run 디렉터리의 `results.csv`, `ranking.csv`, `run-result.json`에, 회사별 근거·요청·RunManifest·체크포인트는 `companies\TICKER\` 아래에 저장됩니다. `llm-calls.jsonl`에는 호출마다 실제 사용 모델이 기록되고 `dcf-manifest.json`에는 `calculation_mode=deterministic_python`, `llm_model=null`이 기록됩니다. 한 회사의 실패는 다른 회사 실행을 중단시키지 않으며 전체 명령은 실패 회사가 있으면 종료 코드 2를 반환합니다.

Parser 품질 gate는 기본적으로 visible text retention 95% 이상, numeric cell retention 99% 이상, 원문/canonical 표 개수 일치를 요구합니다. 원문 특성상 의도적으로 완화할 때만 `--minimum-text-retention`, `--minimum-numeric-retention`, `--allow-table-count-mismatch`, `--allow-low-quality`를 사용하십시오. 판정은 회사별 `quality-gate.json`에 기록됩니다.

시장가격은 반드시 `price_as_of`, valuation은 `valuation_as_of`를 가집니다. 미래 가격과 기본 7일보다 오래된 가격은 거부합니다. 장기 휴장 등으로 완화해야 할 때만 `--maximum-price-age-days`와 백테스트의 `--maximum-signal-price-age-days`를 명시적으로 조정하십시오.

## PIT Factor Backtest

여러 rebalance 시점에 저장한 run 결과와 수정주가(adjusted close) 패널로 랭킹의 사후 수익률을 검증할 수 있습니다. 가격 CSV 필수 열은 `timestamp,ticker,adjusted_close`이고 선택 열 `tradable=false`인 행은 체결 후보에서 제외됩니다. timestamp는 timezone-aware여야 합니다.

```powershell
moatrader backtest run `
  --runs-root data-lake\gold\runs\monthly `
  --prices data\adjusted-prices.csv `
  --end-at 2026-12-31T16:00:00+09:00 `
  --top-n 10 `
  --execution-lag-days 1 `
  --transaction-cost-bps 10 `
  --output data-lake\gold\backtests\monthly-v1
```

각 signal 이후 지정된 lag를 지난 첫 공통 거래 timestamp에 동일가중 체결하고, 보유·신규 종목 모두에 수정주가가 있어야 진행합니다. 가격이 사라진 상장폐지 종목을 조용히 제거하지 않고 실패시키므로, 데이터셋에는 상장폐지일까지의 수정주가나 명시적인 청산 가격을 포함해야 합니다. 결과는 `backtest-result.json`, `equity.csv`, `rebalances.csv`이며 turnover와 거래비용을 포함합니다. 이것은 연구 검증 도구이지 미래 수익률 보장이 아닙니다.

예제 DART 문서를 canonical artifact로 변환합니다.

```powershell
moatrader ingest-html `
  examples\sample-dart.html `
  --metadata examples\sample-dart-metadata.json `
  --source dart `
  --context-tokens 32000 `
  --output data-lake\demo
```

생성물은 다음과 같습니다.

```text
bundle.json                 canonical source of truth
document.md                 구조화 Markdown
financial-snapshot.md       fact 기반 숫자/파생지표
chunks.jsonl                semantic chunks
evidence-requests.jsonl     모델 독립적 LLM 요청 + response JSON Schema
context-allocation.json     token budget 선택 결과
```

JSON Schema 출력:

```powershell
moatrader schema > canonical-document-bundle.schema.json
```

테스트:

```powershell
$env:PYTHONPATH = "$PWD\src"
pytest
```

## 서비스에서의 권장 실행 순서

1. 원문과 SHA-256을 Bronze에 변경 없이 저장합니다.
2. adapter 출력 bundle을 Silver에 저장합니다.
3. parser retention/table/numeric 지표가 기준을 통과한 문서만 다음 단계로 보냅니다.
4. 모든 chunk에 대해 local evidence 요청을 실행하고 Python validator를 통과시킵니다.
5. evidence ID만 인용하는 section summary와 company dossier를 만듭니다.
6. token allocator가 고른 raw chunk와 dossier로 3-layer evidence pack을 만듭니다.
7. LLM은 MOAT를 채점하고, Python은 StructuredFact로 DCF를 계산합니다.
8. 동일한 signal timestamp의 시장가격을 결합해 screening합니다.
9. parser/model/prompt/input hash를 `RunManifest`로 남기고 PIT 백테스트합니다.

`ValueMoatRanker`의 기본 점수는 다음처럼 완전히 공개되어 있습니다.

```text
(MOAT / 10)
× max(0, 1 - price / DCF fair value)
× model confidence
× document coverage
```

이는 수익률 보장이 아니라 연구용 우선순위 함수입니다. 임계값과 식 자체는 survivorship bias, look-ahead bias, 거래비용을 포함한 백테스트로 검증해야 합니다.
