from moatrader.ingestion import ResilientHttpClient


def test_http_url_redaction_never_exposes_dart_key() -> None:
    safe = ResilientHttpClient.redact_url(
        "https://opendart.fss.or.kr/api/list.json?crtfc_key=secret-value&corp_code=00126380"
    )

    assert "secret-value" not in safe
    assert "REDACTED" in safe
    assert "corp_code=00126380" in safe
