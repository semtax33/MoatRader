# DART/SEC/IR/산업 보고서 원문 수집기

## 선택한 원천

DART는 OpenDART 공식 `list.json`, `corpCode.xml`, `document.xml`을 사용합니다. SEC는 별도 API key가 필요 없는 공식 `data.sec.gov/submissions/CIK##########.json`으로 공시를 찾고 `www.sec.gov/Archives/edgar/data`에서 primary HTML과 complete submission text를 함께 내려받습니다. `edgartools`가 없어도 원 출처와 URL을 직접 감사할 수 있도록 공식 API를 직접 사용했습니다.

- OpenDART 공시검색: <https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019001>
- OpenDART 원문파일: <https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019003>
- SEC EDGAR data APIs: <https://www.sec.gov/search-filings/edgar-application-programming-interfaces>
- SEC automated-access 정책: <https://www.sec.gov/about/webmaster-frequently-asked-questions>

## 보안과 요청 정책

- DART key는 기본 `DART_API_KEY` 환경변수에서만 읽고 URL/error/metadata에는 저장하지 않습니다.
- SEC는 기본 `SEC_USER_AGENT`에 `애플리케이션 또는 회사명 + 연락 이메일`이 있어야 합니다.
- SEC 기본 요청률은 5 req/s이며 CLI에서 공식 상한 10 req/s 초과를 거부합니다.
- DART 기본 요청률은 2 req/s입니다.
- HTTP 408/409/425/429/5xx와 OpenDART 800/900은 bounded exponential retry합니다. 인증·quota 오류는 반복하지 않습니다.
- response byte limit, ZIP 총 압축해제 크기/entry 수, 암호화 entry, path traversal과 중복 경로를 fail-closed로 검사합니다.

## PIT 시각 정책

DART `list.json`은 `rcept_dt` 일자만 제공합니다. collector는 같은 날의 임의 시각을 발명하지 않고 해당 날짜 `23:59:59.999999+09:00`을 `available_at`으로 사용하며 precision을 `DAY`로 기록합니다. 따라서 당일 장중 signal에는 보수적으로 포함되지 않습니다.

SEC submissions의 `acceptanceDateTime`은 `published_at`으로 그대로 보존합니다. SEC는 filing acceptance와 웹 문서 공개 사이에 보통 1–3분의 지연이 있고 정확한 최초 공개 timestamp를 제공하지 않으므로, 기본 `available_at`은 acceptance+5분이고 precision은 `INFERRED`입니다. `--availability-lag-minutes`로 연구 정책을 바꿀 수 있지만 0은 짧은 look-ahead 위험을 받아들이는 설정입니다.

## 저장 구조와 revision

```text
data-lake/bronze/
├─ dart/{rcept_no}/
│  ├─ latest.json
│  └─ versions/{version_sha256}/
│     ├─ original.zip
│     ├─ documents/{원 ZIP 경로}
│     ├─ metadata.json
│     └─ sha256.txt
└─ sec-edgar/{accession}/
   ├─ latest.json
   └─ versions/{version_sha256}/
      ├─ documents/{primaryDocument}
      ├─ original-submission.txt
      ├─ metadata.json
      └─ sha256.txt
```

`raw_sha256`는 downstream parser가 읽는 primary document의 hash입니다. `version_sha256`는 파일명과 모든 저장 파일 hash의 합성 hash라서 primary HTML 또는 complete submission/ZIP 어느 쪽이 바뀌어도 새 version이 됩니다. 기존 version은 수정하지 않습니다. 기본 실행은 known ID를 건너뛰며 `--refresh`가 실제 byte revision을 확인합니다.

DART 정정공시는 정정 prefix를 제거한 보고서명과 회사가 같은 기존 원본을 연결합니다. SEC `/A`는 base form과 report date가 같은 기존 accession을 연결합니다. 수집 범위에 원본이 없어 연결하지 못하면 거짓 ID를 만들지 않고 `source_specific.amendment_link_status=UNRESOLVED`로 남깁니다.

## 명령

```powershell
$env:DART_API_KEY = "..."
moatrader collect dart `
  --stock-code 005930 `
  --from 2024-01-01 `
  --to 2026-08-14 `
  --output data-lake\bronze
```

`--stock-code` 또는 `--corp-code`는 반복할 수 있습니다. 대형 지정 유니버스는 `--stock-code-file`/`--corp-code-file`에 쉼표 또는 줄바꿈으로 코드를 넣습니다. `#` 뒤는 주석입니다. `--all-companies`는 OpenDART의 회사 미지정 검색기간 제한 때문에 약 3개월 이하만 허용합니다. 정정공시까지 보존하는 것이 기본이고, `--final-only`는 원 API에서 최종보고서만 요청합니다.

```powershell
$env:SEC_USER_AGENT = "MoatRader your-email@example.com"
moatrader collect sec `
  --ticker AAPL `
  --ticker MSFT `
  --from 2024-01-01 `
  --to 2026-08-14 `
  --output data-lake\bronze
```

SEC 기본 form은 `10-K`, `10-Q`, `20-F`, `40-F`와 각각의 amendment입니다. `--form 8-K`처럼 반복 옵션으로 바꿀 수 있습니다. ticker 대신 `--cik`를 사용할 수 있고, 대형 지정 유니버스는 `--ticker-file`/`--cik-file`을 사용합니다. 오래된 기간은 submissions 응답의 추가 history JSON까지 자동으로 따라갑니다.

두 collector는 조회기간·종목/CIK·form·refresh 정책·collector version을 담은 collection별 결과 JSON과 combined `collected-universe.csv`를 만듭니다. API key와 SEC User-Agent는 기록하지 않습니다. CSV는 parser/runner가 바로 읽을 수 있지만 가격과 DCF assumptions는 시장데이터 계층의 별도 책임입니다.

## 운영상 남은 경계

CLI collector는 재실행·revision·부분 실패 격리를 제공하지만 daemon scheduler, distributed queue, database transaction, alerting, API key vault는 아직 포함하지 않습니다. 상시 서비스에서는 CLI를 scheduler에서 호출하고 collection 결과의 `failures`와 latest pointer 무결성을 모니터링해야 합니다.

한국경제 컨센서스의 애널리스트 산업 보고서 수집과 Synalyst 저장 PDF 재사용 절차는
[`industry-reports.md`](industry-reports.md)를 따릅니다. 산업 보고서는 회사 universe
manifest에 넣지 않고 reference-class evidence catalog로 분리합니다.
