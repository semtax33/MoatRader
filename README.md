# MoatRader — Evidence → Business Economics → Market Expectations

비결정적인 LLM을 금융 의사결정에 사용할 때 생기는 **evidence instability**를 PIT(point-in-time), provenance, metamorphic testing, deterministic scoring으로 통제하는 투자 리서치 파이프라인입니다.

MoatRader의 목표는 “해자가 높은 회사를 줄 세우는 것”이 아니라 **현재 가격이 요구하는 미래와 PIT 증거가 지지하는 미래의 차이**를 계산하는 것입니다. LLM은 원문에서 경제적 사실과 valuation-driver 의미를 분류하고, Python은 관계·재투자·DCF·3P·Reverse DCF·Expectation Gap을 계산합니다. 과거 MOAT scalar는 Evidence Sensor 회귀/진단용으로만 남습니다.

| 문제 | MoatRader의 통제 방식 |
| --- | --- |
| 입력 순서와 문서 형식에 따라 LLM 판단이 달라짐 | deterministic atomic segmentation + metamorphic invariance gate |
| 같은 근거의 반복이 점수를 부풀림 | canonical claim set + idempotent reducer |
| 생성 요약이 원문처럼 다시 채점됨 | generated summary를 scoring candidate에서 제외 |
| 인용과 숫자를 원문에서 확인하기 어려움 | Evidence ID → node/fact → raw quote provenance |
| 미래 공시·가격이 과거 평가에 섞임 | timezone-aware `available_at`과 PIT filtering |
| 모델 출력이나 MOAT 점수가 가치평가를 직접 결정함 | LLM은 grounded evidence만 분류하고, Economic DCF·3P·Reverse DCF는 deterministic Python이 계산 |
| 주가를 본 뒤 intrinsic 가정을 조정함 | price-blind intrinsic lane과 price-aware market lane을 타입·실행 순서로 분리 |
| 같은 장점을 성장·마진·WACC·terminal growth에 중복 반영 | 한 evidence당 하나의 primary driver만 적용하고 related driver는 진단용으로 제한 |
| 검증되지 않은 설정으로 대규모 비용이 발생함 | 5종목 초과 실행은 통과한 preflight report 없이는 차단 |

## 핵심 아이디어

```text
DART / SEC EDGAR / IR text·table·vision
          │
          ▼
CanonicalDocumentBundle ── AST · StructuredFact · Asset · Provenance
          │
          ▼
Deterministic Atomic Evidence
          │
          ▼
                 ┌─ Frozen Evidence Sensor: MOAT mechanism/outcome/counter
Canonical chunks┤
                 └─ Valuation Lane: revenue·margin·ROIIC·reinvestment·CAP·risk
          │
          ▼
Validation ── ValuationDriverEvidence ── Competitive Advantage Profile
PIT Financial Snapshot ─────────────────┬─ price-blind Economic DCF + 3P
                                        └─ price-aware Reverse DCF surface
                                                        │
                                                        ▼
                                              Expectation Gap ranking
```

LLM에게 구조 복원, 요약 기반 재해석, DCF 산술, 확률 또는 공정가치 생성을 맡기지 않습니다. 동결된 MOAT Evidence Sensor는 회귀 self-test로 유지하며, 별도 valuation lane은 `MOAT_NONE`인 pipeline·approval·CAPEX·mix·capital-allocation 사실도 보존합니다. 동일 evidence를 여러 DCF lever에 중복 적용하지 않으며, confidence는 가치에 곱하지 않고 scenario range를 넓힙니다.

## Reliability architecture

### 1. Evidence identity와 scoring identity의 분리

- `evidence_id`: 원문 위치와 인용을 감사하기 위한 identity
- atomic evidence key: 동일한 원문 evidence의 replay/cache identity
- `claim_id`: 의미가 같은 근거를 중복 제거하기 위한 scoring identity

Replay는 full prompt나 배열 순서가 아니라 atomic evidence identity에 묶입니다. Atomic/canonical audit set은 교환·결합·멱등성을 유지하고, contextual strength attribute와 최종 점수의 반복 재현성은 별도 preflight gate에서 검사합니다.

### 2. Metamorphic invariance

동일한 경제적 정보를 유지한 채 다음 변형을 가해도 selected atomic audit evidence와 canonical claim이 같아야 합니다. Contextual strength score는 동일 broad-context 반복 실행 및 holdout 안정성으로 별도 검증합니다.

- sentence / paragraph shuffle
- evidence duplication
- generated-summary injection
- whitespace / heading 변경
- irrelevant boilerplate 삽입
- AST node order 변경

Compact evidence pack은 audit용 claim·counterevidence 보존만 검사합니다. 경제적 강도 입력인 broad contextual pack은 초기 정확도 단계에서 압축하지 않습니다. 관련 회귀 테스트는 [`tests/test_moat_metamorphic.py`](tests/test_moat_metamorphic.py)와 [`tests/test_moat_strength.py`](tests/test_moat_strength.py)에 있습니다.

### 3. Fail-closed preflight

5종목을 초과하는 manifest는 3~5종목 표본의 반복 실행, evidence/claim 일치, 인접 시점 안정성, metamorphic gate, compression-invariance gate를 통과한 preflight report가 있어야 실행됩니다. Universe, as-of date, model, prompt, reasoning, runner 계약이 달라지면 기존 승인은 재사용할 수 없습니다.

### 4. PIT와 provenance

모든 문서는 timezone-aware `available_at`을 가지며 평가 시점 이후에 공개된 공시는 제외됩니다. 계산 숫자는 `StructuredFact`로 관리하고, 파생지표와 DCF 입력에는 사용한 공시·기간·재무제표 범위·가정 출처·input hash를 남깁니다.

한국 주식 TTM은 동일한 CFS/OFS 범위에서 `직전 FY + 당기 YTD - 전년 동기 YTD`로 구성합니다. 필요한 누적값이 없으면 서로 다른 범위나 3개월 값을 임의로 혼합하지 않고 해당 DCF를 제외합니다.

### 5. 정확도 보존형 token/cache 최적화

LLM 요청은 고정 rubric·schema를 앞에, 기업별 원문 context를 뒤에 둡니다. GPT-5.6 요청은 고정 prefix 끝에 explicit cache breakpoint를 두고 `mode=explicit`, `ttl=30m`을 사용합니다. Cache key는 고정 prefix hash와 32개 stable shard로 구성해 한 key에 호출이 몰리는 것을 줄입니다. JSON schema는 key 순서와 whitespace를 canonical serialization하여 동일 prefix가 byte 수준에서 흔들리지 않게 합니다.

회사별 `llm-token-budget.json`에는 다음을 함께 기록합니다.

- full/compact context token estimate와 compression ratio
- cacheable static prefix, dynamic suffix, output token cap
- 실제 input/output/cached/cache-write token
- cache read/write 비율과 provider tokens per call

경제적 강도용 broad context는 정확도 검증 단계에서 압축하지 않습니다. Factor별 pruning, 축약 field alias, atomic reasoning 하향, small→large confidence escalation, canonical-state delta 재사용은 holdout 품질 동등성이 확인되기 전에는 production 기본값으로 쓰지 않습니다. `previous_response_id`나 opaque compaction도 PIT provenance/replay를 대신하지 않습니다.

## 구현 범위

- OpenDART와 SEC EDGAR 원문 수집, immutable content-addressed Bronze 저장
- DART XML/HTML, SEC HTML/iXBRL, IR HTML → source-neutral canonical model
- section/table/inline XBRL parsing과 원문 provenance
- semantic chunking, cross-filing dedup, 숫자가 바뀐 유사 문서 보존
- atomic audit evidence 분류, schema validation/repair, evidence-level replay
- broad canonical-chunk strength retrieval과 contextual grounding/reconciliation
- 동결된 Evidence Sensor v1과 MOAT scalar diagnostic
- ValuationDriverEvidence, CompetitiveAdvantageProfile, CAP range
- growth–reinvestment–ROIIC–CAP/fade를 연결한 Economic FCFF와 기존 FCFF sanity check
- Possible/Plausible/Probable validator, Reverse DCF expectation surface, Expectation Gap
- reported/economic ROIC 병렬 capital-allocation·intangible-capital 분석
- 일반기업/성숙기업/금융/바이오/플랫폼/distressed valuation model router와 biotech rNPV
- 회사별 실패 격리, checkpoint/resume, 사용량·input hash·run manifest 기록
- PIT backtest, 거래비용·슬리피지·거래 가능 여부·보수적 상장폐지 처리
- raw/sector/factor-neutral IC와 비중첩 Q5–Q1 signal evaluation

소스 구조도 책임 경계를 따릅니다.

```text
src/moatrader/
├─ adapters/     source-specific parsing
├─ canonical/    source-neutral document contract
├─ evidence/     atomic evidence, validation, claim ledger
├─ business/     valuation drivers, CAP, lifecycle, capital allocation
├─ valuation/    Economic DCF, 3P, scenarios, Reverse DCF, model router
├─ expectations/ intrinsic/market lane merge and expectation gap
├─ llm/          constrained transport and replay
├─ retrieval/    economic-question retrieval
├─ financial/    PIT snapshot and legacy deterministic FCFF
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
moatrader analyze run `
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

moatrader analyze run `
  --universe examples\universe.csv `
  --ticker SAMPLE `
  --as-of 2025-05-16T00:00:00+09:00 `
  --run-id sample-live
```

`expectation_assumptions`가 있는 회사는 `valuation-driver-evidence.json`, `intrinsic-valuation.json`, `three-p.json`, `reverse-dcf-surface.json`, `expectation-gap.json`을 추가로 생성합니다. 전체 주 출력은 `opportunities.csv`이며 `ranking.csv`는 `--enable-legacy-moat-ranking`을 명시한 경우에만 과거 진단 결과를 담습니다.

기존 `moatrader moat ...` 명령은 호환 alias로 유지되지만 새 문서와 운영 경로는 `moatrader analyze ...`를 사용합니다. 숫자 가정 예시는 [`examples/sample-expectation-assumptions.json`](examples/sample-expectation-assumptions.json)에 있습니다.

## 검증 전략

테스트는 단순 parser unit test뿐 아니라 시스템의 신뢰성 주장을 직접 검증합니다.

| 검증 영역 | 대표 테스트 |
| --- | --- |
| 입력 변형 불변성·atomic replay·audit reducer 성질 | `test_moat_metamorphic.py` |
| contextual grounding·reconciliation·strength/confidence 분리 | `test_moat_strength.py` |
| 반복 실행·입력 순서 재현성 | `test_moat_reproducibility.py` |
| 대규모 실행 승인 계약 | `test_preflight.py` |
| timezone-aware PIT와 TTM 구성 | `test_kr_dcf_pit.py` |
| canonical model과 source-neutral 계약 | `test_model_contract.py` |
| raw/neutral IC와 Q5–Q1 평가 | `test_signal_evaluation.py` |
| Economic DCF·3P·Reverse DCF·price isolation·Expectation Gap | `test_expectation_pivot.py` |

핵심 실패 조건은 조용히 보정하지 않습니다. 미래 공시, naive timestamp, stale price, provenance 없는 입력, parser 품질 저하, 불일치한 preflight 계약은 명시적으로 거부하거나 결과에서 제외 사유를 기록합니다.

## 운영 경계

현재 직접 지원하는 입력은 DART XML/HTML, SEC HTML/iXBRL, IR HTML/PDF입니다. IR PDF text/table parser와 선택적 OCR/vision audit lane은 구현되어 있지만 전체 PDF vision의 자동 production 활성화는 unseen-PDF audit 전까지 보류합니다. 또한 이 저장소는 연구 파이프라인이며 scheduler, queue, database, auth, monitoring을 갖춘 상시 운영 서비스는 아닙니다.

Backtest가 편향을 자동으로 없애 주는 것도 아닙니다. 역사적 investable universe, corporate action이 반영된 수정주가, 상장폐지 가격, 정확한 공개 가능 시각은 외부 데이터가 보장해야 합니다. 출력은 투자 권유나 미래 수익률 보장이 아닙니다.

## 상세 문서

- [Canonical document model](docs/canonical-financial-document-model.md)
- [구현 상태와 외부 데이터 책임](docs/implementation-status.md)
- [DART / SEC 수집과 Bronze 저장 계약](docs/ingestion.md)
- [Universe manifest 형식](docs/universe-manifest.md)
- [Expectation Analysis 계약](docs/expectation-analysis.md)
- [요구사항별 구현 감사표](docs/requirements-audit.md)

기본 screening은 다음의 연구 질문을 구현합니다.

```text
Evidence-based possible/plausible/probable future
− Market-implied growth/margin/ROIIC/CAP surface
= Expectation Gap
```

MOAT×DCF scalar ranking은 기본 경로에서 비활성화되었습니다. 새 driver/3P 계약을 수익률로 튜닝하지 않고 먼저 고정한 뒤, 보지 않은 날짜의 fresh holdout에서 Expectation Gap을 검증해야 합니다.
