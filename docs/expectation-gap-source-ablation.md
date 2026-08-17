# Expectation GAP 소스 ablation

`scripts/run_expectation_gap_ablation.py`는 같은 PIT universe와 같은 forward return을 두고
다음 세 레인을 비교하는 research-only runner입니다.

1. `DART_ONLY`
2. `DART_IR`
3. `DART_IR_INDUSTRY`

기존 MOAT/DCF 점수나 LLM replay 결과는 입력으로 쓰지 않습니다. 기존 Bronze DART 원문,
PIT TTM 정규화 숫자와 로컬에 저장된 Synalyst 산업 PDF만 deterministic input으로 재사용하며,
IR PDF와 산업 PDF의 valuation 분류는 실험 디렉터리 안에서 `gpt-5.6-luna`로 새로 만듭니다.

## 고정 점수 계약

- Cheap = `0.50 ExpectationGap + 0.30 ProbableMOS + 0.20 PlausibleMOS`
- Improving = `0.35 ΔV_probable + 0.25 ΔV_plausible + 0.25 DriverBreadth + 0.15 EvidenceRevision`
- Non-Fragile = `100 - Fragility`
- Composite = `100 × Cheap^0.40 × Improving^0.35 × NonFragile^0.25` (각 축을 0~1로 정규화)
- `NonFragile < 30`은 `HIGH_FRAGILITY`로 rank 제외
- 직전 3개월 값이 없으면 Improving을 0으로 대체하지 않고 `INSUFFICIENT_EVIDENCE`

Fragility는 WACC 민감도 20%, terminal-g 민감도 15%, 시나리오 분산 20%, terminal
value 비중 15%, 단일 driver 의존 15%, evidence weakness 15%입니다. 가중치와 gate는
수익률을 조인하기 전에 고정합니다.

IR과 산업보고서는 임의의 성장률·마진 bump를 만들지 않습니다. 중앙 DCF는 DART PIT
숫자로 유지하고, 추가 소스는 3P evidence coverage, counter/range width와 fragility에만
영향을 줍니다. 산업 자료는 issuer fact가 아니라 reference-class/scenario evidence입니다.

## 실행

새 실험은 비어 있는 output에서 시작합니다. 같은 실험의 문서별 실패를 이어서 처리할 때만
`--resume`을 사용합니다.

```powershell
$python = "D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe"

& $python scripts\run_expectation_gap_ablation.py `
  --dates D:\Programming\python_example\MoatPoC\dates.csv `
  --universe D:\Programming\python_example\MoatPoC\universe.csv `
  --base-root data-lake\backtests\kr-signal-fresh-20260815-v080-rerun-01 `
  --synalyst-root D:\Programming\python_example\Synalyst `
  --output data-lake\experiments\expectation-gap-ablation-20260818-v1 `
  --stage all
```

긴 실행은 `prepare`, `classify`, `score`, `evaluate` 단계로 나눌 수 있습니다. 구조화
분류 실패는 문서 1건 단위로 다시 실행하면 이미 성공한 실험 내부 checkpoint를 보존합니다.

## 2026-08-18 실행 결과

- 입력: 4 dates × 150 stocks = 600 stock-date
- realized return: 600/600
- canonical PDF parse: 139/139
- valuation unit이 있는 문서: 133, fresh `gpt-5.6-luna` 분류: 133/133
- IR evidence 적용: 113 stock-date
- industry evidence 적용: 395 stock-date
- 일반 FCFF 적용 가능: 278/600. 나머지는 missing PIT input 또는 비양수 중앙/상방 equity value
- 첫 날짜는 과거 revision이 없으므로 composite 비교에서 제외

세 레인 모두 composite가 존재하는 137개 stock-date의 결과는 다음과 같습니다.

| Lane | Raw IC | Sector-neutral IC | Q5-Q1 | Worst-decile | Downside capture |
|---|---:|---:|---:|---:|---:|
| DART only | 0.1837 | 0.2008 | 5.66% | -17.42% | 0.6704 |
| DART + IR | 0.1973 | 0.1830 | 7.43% | -17.42% | 0.5942 |
| DART + IR + industry | 0.1737 | 0.1845 | 4.30% | -13.10% | 0.6397 |

해석은 탐색적입니다. IR은 raw IC, Q5-Q1과 downside capture를 개선했지만 sector-neutral IC는
낮아졌습니다. 산업보고서는 worst-decile을 완화했지만 alpha 지표는 IR-only보다 나빠졌습니다.
관측 시점이 세 개뿐이고 common sample도 137개이므로 가중치를 바꾸거나 production alpha로
승격할 근거는 아닙니다. 다음 holdout에서는 현재 계약을 freeze하고 새 날짜만 추가해야 합니다.

전체 산출물은 해당 실험의 `signals.csv`, `returns.csv`, `evaluation.json`, `evaluation.md`에
있습니다. `evaluation.json`에는 날짜별 IC/Q5-Q1, top-portfolio compound return/MDD와
same-sample horse race가 포함됩니다.
