# MoatRader — Evidence-grounded Financial LLM Evaluation & Valuation Pipeline

비결정적인 LLM을 금융 의사결정에 사용할 때 생기는 **evidence instability**를 PIT(point-in-time), provenance, metamorphic testing, deterministic scoring으로 통제하는 투자 리서치 파이프라인입니다.

MoatRader의 목표는 “LLM이 추천한 종목”을 만드는 것이 아닙니다. 같은 경제적 사실이 문장 순서, 중복, 요약문, formatting 변화 때문에 다른 점수로 바뀌지 않도록 **LLM의 역할을 atomic evidence 분류로 제한하고 최종 판단을 감사 가능한 Python 로직으로 옮기는 것**이 핵심입니다.

| 문제 | MoatRader의 통제 방식 |
| --- | --- |
| 입력 순서와 문서 형식에 따라 LLM 판단이 달라짐 | deterministic atomic segmentation + metamorphic invariance gate |
| 같은 근거의 반복이 점수를 부풀림 | canonical claim set + idempotent reducer |
| 생성 요약이 원문처럼 다시 채점됨 | generated summary를 scoring candidate에서 제외 |
| 인용과 숫자를 원문에서 확인하기 어려움 | Evidence ID → node/fact → raw quote provenance |
| 미래 공시·가격이 과거 평가에 섞임 | timezone-aware `available_at`과 PIT filtering |
| 모델 출력이 MOAT·가치평가를 직접 결정함 | LLM은 분류만 수행, MOAT와 DCF는 deterministic Python |
| 검증되지 않은 설정으로 대규모 비용이 발생함 | 5종목 초과 실행은 통과한 preflight report 없이는 차단 |

## 핵심 아이디어

```text
DART / SEC EDGAR / IR HTML
          │
          ▼
CanonicalDocumentBundle ── AST · StructuredFact · Asset · Provenance
          │
          ▼
Deterministic Atomic Evidence
          │
          ▼
LLM: fixed-rubric classification only
          │
          ▼
Validation ── Canonical Claim Set ── Python Reducer
                                      ├─ deterministic MOAT
PIT Financial Snapshot ───────────────└─ deterministic DCF
                                                  │
                                                  ▼
                                  value · quality · confidence ranking
```

LLM에게 구조 복원, 요약 기반 재해석, DCF 산술, 최종 MOAT 채점을 맡기지 않습니다. 모델은 원문에서 결정론적으로 분리된 evidence 한 건을 고정 rubric으로 분류하며, 이후 단계는 검증된 claim과 명시적인 계산식만 사용합니다.

## Reliability architecture

### 1. Evidence identity와 scoring identity의 분리

- `evidence_id`: 원문 위치와 인용을 감사하기 위한 identity
- atomic evidence key: 동일한 원문 evidence의 replay/cache identity
- `claim_id`: 의미가 같은 근거를 중복 제거하기 위한 scoring identity

Replay는 full prompt나 배열 순서가 아니라 atomic evidence identity에 묶입니다. Canonical claim reducer는 교환·결합·멱등성을 갖도록 설계되어 입력 순서와 중복이 최종 점수를 바꾸지 않습니다.

### 2. Metamorphic invariance

동일한 경제적 정보를 유지한 채 다음 변형을 가해도 selected atomic evidence, canonical claim, factor score가 같아야 합니다.

- sentence / paragraph shuffle
- evidence duplication
- generated-summary injection
- whitespace / heading 변경
- irrelevant boilerplate 삽입
- AST node order 변경

Compact evidence pack에는 claim·factor score 불변성과 counterevidence recall 검사를 별도로 적용합니다. 관련 회귀 테스트는 [`tests/test_moat_metamorphic.py`](tests/test_moat_metamorphic.py)에 있습니다.

### 3. Fail-closed preflight

5종목을 초과하는 manifest는 3~5종목 표본의 반복 실행, evidence/claim 일치, 인접 시점 안정성, metamorphic gate, compression-invariance gate를 통과한 preflight report가 있어야 실행됩니다. Universe, as-of date, model, prompt, reasoning, runner 계약이 달라지면 기존 승인은 재사용할 수 없습니다.

### 4. PIT와 provenance

모든 문서는 timezone-aware `available_at`을 가지며 평가 시점 이후에 공개된 공시는 제외됩니다. 계산 숫자는 `StructuredFact`로 관리하고, 파생지표와 DCF 입력에는 사용한 공시·기간·재무제표 범위·가정 출처·input hash를 남깁니다.

한국 주식 TTM은 동일한 CFS/OFS 범위에서 `직전 FY + 당기 YTD - 전년 동기 YTD`로 구성합니다. 필요한 누적값이 없으면 서로 다른 범위나 3개월 값을 임의로 혼합하지 않고 해당 DCF를 제외합니다.

## 구현 범위

- OpenDART와 SEC EDGAR 원문 수집, immutable content-addressed Bronze 저장
- DART XML/HTML, SEC HTML/iXBRL, IR HTML → source-neutral canonical model
- section/table/inline XBRL parsing과 원문 provenance
- semantic chunking, cross-filing dedup, 숫자가 바뀐 유사 문서 보존
- atomic evidence 분류, schema validation/repair, evidence-level replay
- canonical claim reducer, deterministic MOAT, unlevered DCF
- 회사별 실패 격리, checkpoint/resume, 사용량·input hash·run manifest 기록
- PIT backtest, 거래비용·슬리피지·거래 가능 여부·보수적 상장폐지 처리
- raw/sector/factor-neutral IC와 비중첩 Q5–Q1 signal evaluation

소스 구조도 책임 경계를 따릅니다.

```text
src/moatrader/
├─ adapters/     source-specific parsing
├─ canonical/    source-neutral document contract
├─ evidence/     atomic evidence, validation, claim ledger
├─ llm/          constrained transport and replay
├─ retrieval/    economic-question retrieval
├─ financial/    PIT snapshot and deterministic DCF
├─ runner/       execution, isolation, checkpointing
├─ preflight.py  full-universe execution gate
└─ backtest/     point-in-time signal evaluation
```

## 빠른 검증

Python 3.11 이상이 필요합니다. 이 저장소의 Windows 개발 환경에서는 다음과 같이 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& D:\Programming\python_example\Arcana\.venv-llama\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
```

외부 API 없이 예제 DART 문서를 canonical artifact로 변환할 수 있습니다.

```powershell
moatrader ingest-html `
  examples\sample-dart.html `
  --metadata examples\sample-dart-metadata.json `
  --source dart `
  --output data-lake\demo
```

LLM 요청 전까지의 입력·PIT·파싱·DCF 계약은 dry-run으로 확인합니다.

```powershell
moatrader moat run `
  --universe examples\universe.csv `
  --ticker SAMPLE `
  --as-of 2025-05-16T00:00:00+09:00 `
  --run-id sample-dry `
  --dry-run
```

실제 atomic evidence 분류를 실행하려면 OpenAI 의존성과 API 키를 추가합니다.

```powershell
python -m pip install -e ".[llm]"
$env:OPENAI_API_KEY = "..."

moatrader moat run `
  --universe examples\universe.csv `
  --ticker SAMPLE `
  --as-of 2025-05-16T00:00:00+09:00 `
  --run-id sample-live
```

실행 결과에는 회사별 evidence와 요청, `RunManifest`, DCF audit, checkpoint가 저장되며 전체 결과는 `results.csv`, `ranking.csv`, `run-result.json`으로 생성됩니다.

## 검증 전략

테스트는 단순 parser unit test뿐 아니라 시스템의 신뢰성 주장을 직접 검증합니다.

| 검증 영역 | 대표 테스트 |
| --- | --- |
| 입력 변형 불변성·atomic replay·reducer 성질 | `test_moat_metamorphic.py` |
| 반복 실행·입력 순서 재현성 | `test_moat_reproducibility.py` |
| 대규모 실행 승인 계약 | `test_preflight.py` |
| timezone-aware PIT와 TTM 구성 | `test_kr_dcf_pit.py` |
| canonical model과 source-neutral 계약 | `test_model_contract.py` |
| raw/neutral IC와 Q5–Q1 평가 | `test_signal_evaluation.py` |

핵심 실패 조건은 조용히 보정하지 않습니다. 미래 공시, naive timestamp, stale price, provenance 없는 입력, parser 품질 저하, 불일치한 preflight 계약은 명시적으로 거부하거나 결과에서 제외 사유를 기록합니다.

## 운영 경계

현재 직접 지원하는 입력은 DART XML/HTML, SEC HTML/iXBRL, IR HTML입니다. PDF/PPT, OCR, chart 해석은 canonical 계약만 준비되어 있고 전용 adapter는 아직 구현되지 않았습니다. 또한 이 저장소는 연구 파이프라인이며 scheduler, queue, database, auth, monitoring을 갖춘 상시 운영 서비스는 아닙니다.

Backtest가 편향을 자동으로 없애 주는 것도 아닙니다. 역사적 investable universe, corporate action이 반영된 수정주가, 상장폐지 가격, 정확한 공개 가능 시각은 외부 데이터가 보장해야 합니다. 출력은 투자 권유나 미래 수익률 보장이 아닙니다.

## 상세 문서

- [Canonical document model](docs/canonical-financial-document-model.md)
- [구현 상태와 외부 데이터 책임](docs/implementation-status.md)
- [DART / SEC 수집과 Bronze 저장 계약](docs/ingestion.md)
- [Universe manifest 형식](docs/universe-manifest.md)
- [요구사항별 구현 감사표](docs/requirements-audit.md)

기본 ranking은 다음의 공개된 우선순위 함수입니다.

```text
(MOAT / 10)
× max(0, 1 - price / DCF fair value)
× model confidence
× document coverage
```

점수식과 임계값은 고정된 진리가 아니라 PIT backtest와 signal evaluation으로 검증해야 할 연구 가설입니다.
