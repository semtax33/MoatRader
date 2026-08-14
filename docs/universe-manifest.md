# Universe manifest 계약

`moatrader moat run`은 UTF-8 CSV manifest를 입력으로 받습니다. 한 행은 한 회사가 아니라 한 공시 문서를 뜻합니다. 따라서 같은 ticker에 DART 사업보고서, 분기보고서, IR HTML, SEC 10-K를 여러 행으로 넣을 수 있습니다.

## 열

| 열 | 필수 | 의미 |
|---|---:|---|
| `ticker` | 예 | 실행과 결과 디렉터리에 쓰는 유일한 종목 식별자 |
| `source` | 예 | `DART`, `SEC`, `EDGAR`, `SEC_EDGAR`, `IR` |
| `input` | 예 | 원문 HTML/iXBRL 경로. 상대 경로는 CSV 위치 기준 |
| `metadata` | 예 | source adapter용 UTF-8 JSON metadata 경로 |
| `issuer_id` | 아니요 | DART corp code, SEC CIK 등 안정적인 발행인 ID |
| `issuer_name` | 아니요 | 표시용 회사명 |
| `current_price` | 아니요 | DCF와 결합할 양수 가격 |
| `price_as_of` | 조건부 | 가격 timestamp. `current_price`와 함께 제공하며 offset 필수 |
| `dcf_assumptions` | 아니요 | `DcfAssumptions` JSON 경로 |

회사 수준 선택 열은 같은 ticker의 여러 행에서 충돌하면 안 됩니다. 빈 행과 값이 있는 행을 섞는 것은 허용하지만 서로 다른 두 값은 거부합니다.

```csv
ticker,source,input,metadata,issuer_id,issuer_name,current_price,price_as_of,dcf_assumptions
005930,DART,raw/005930-annual.html,meta/005930-annual.json,00126380,삼성전자,80200,2026-08-14T15:30:00+09:00,dcf/005930.json
005930,IR,raw/005930-ir.html,meta/005930-ir.json,00126380,삼성전자,80200,2026-08-14T15:30:00+09:00,dcf/005930.json
```

## PIT 규칙

- `--as-of`보다 `available_at`이 늦은 문서는 그 run에서 제외됩니다.
- 모든 문서가 제외된 회사는 `NO_PIT_DOCUMENTS`가 됩니다.
- DCF valuation timestamp는 run의 정확한 `--as-of`를 보존합니다.
- 가격은 명시적인 `price_as_of`를 보존합니다. 현재 구현은 두 timestamp가 timezone-aware인지 검증하지만 거래 가능성, 휴장일, corporate action 조정 여부는 데이터 공급 단계에서 보장해야 합니다.

## 실행 선택

- `--ticker 005930`: 한 종목
- `--ticker 005930 --ticker 000660`: 반복 선택
- `--tickers 005930,000660`: 쉼표 목록
- 선택 옵션 생략: manifest 전체

manifest에 없는 ticker를 요청하면 실행 전에 실패하므로 오타로 빈 분석이 생성되지 않습니다.
