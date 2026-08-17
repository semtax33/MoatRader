from __future__ import annotations

import hashlib
import os
from datetime import date

import requests
from pydantic import Field

from moatrader.canonical.models import ContractModel


class KrxAuthenticationError(RuntimeError):
    pass


class KrxSectorRow(ContractModel):
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    issuer_name: str
    market: str
    sector: str
    trade_date: date
    raw_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class KrxDataClient:
    LOGIN_PAGE = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
    LOGIN_JSP = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
    LOGIN_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
    DATA_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    SECTOR_BLD = "dbms/MDC/STAT/standard/MDCSTAT03901"

    def __init__(
        self,
        *,
        login_id: str | None = None,
        password: str | None = None,
        verify_tls: bool = True,
        timeout_seconds: int = 30,
    ) -> None:
        self.login_id = login_id or os.getenv("KRX_ID") or ""
        self.password = password or os.getenv("KRX_PW") or ""
        self.verify_tls = verify_tls
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd?screenId=MDCSTAT039",
                "User-Agent": "MoatRader PIT sector collector/1",
                "X-Requested-With": "XMLHttpRequest",
            }
        )
        self._authenticated = False

    def login(self) -> None:
        if not self.login_id or not self.password:
            raise KrxAuthenticationError("KRX_ID and KRX_PW are required")
        self.session.get(
            self.LOGIN_PAGE,
            timeout=self.timeout_seconds,
            verify=self.verify_tls,
        ).raise_for_status()
        self.session.get(
            self.LOGIN_JSP,
            timeout=self.timeout_seconds,
            verify=self.verify_tls,
        ).raise_for_status()
        payload = {
            "mbrNm": "",
            "telNo": "",
            "di": "",
            "certType": "",
            "mbrId": self.login_id,
            "pw": self.password,
        }
        response = self.session.post(
            self.LOGIN_URL,
            data=payload,
            timeout=self.timeout_seconds,
            verify=self.verify_tls,
        )
        response.raise_for_status()
        data = response.json()
        code = str(data.get("_error_code") or "")
        if code == "CD011":
            payload["skipDup"] = "Y"
            response = self.session.post(
                self.LOGIN_URL,
                data=payload,
                timeout=self.timeout_seconds,
                verify=self.verify_tls,
            )
            response.raise_for_status()
            data = response.json()
            code = str(data.get("_error_code") or "")
        if code != "CD001":
            raise KrxAuthenticationError(
                f"KRX login failed with code {code or 'UNKNOWN'}"
            )
        self._authenticated = True

    def sector_snapshot(self, trade_date: date) -> list[KrxSectorRow]:
        if not self._authenticated:
            self.login()
        results: list[KrxSectorRow] = []
        for market in ("STK", "KSQ"):
            response = self.session.post(
                self.DATA_URL,
                data={
                    "bld": self.SECTOR_BLD,
                    "locale": "en_US",
                    "mktId": market,
                    "trdDd": trade_date.strftime("%Y%m%d"),
                    "csvxls_isNo": "false",
                },
                timeout=self.timeout_seconds,
                verify=self.verify_tls,
            )
            response.raise_for_status()
            if response.text.strip() == "LOGOUT":
                self._authenticated = False
                raise KrxAuthenticationError("KRX session expired")
            payload = response.json()
            digest = hashlib.sha256(response.content).hexdigest()
            for row in payload.get("block1", []):
                code = str(row.get("ISU_SRT_CD") or "").strip()
                if len(code) != 6 or not code.isdigit():
                    continue
                results.append(
                    KrxSectorRow(
                        ticker=code,
                        issuer_name=str(row.get("ISU_ABBRV") or "").strip(),
                        market=str(row.get("MKT_TP_NM") or market).strip(),
                        sector=str(row.get("IDX_IND_NM") or "UNKNOWN").strip(),
                        trade_date=trade_date,
                        raw_response_sha256=digest,
                    )
                )
        return results
