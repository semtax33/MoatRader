from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from moatrader.api.models import ReportCatalog, ReportSummary, ResearchReport
from moatrader.api.repository import (
    ResearchArtifactNotFoundError,
    ResearchArtifactRepository,
)
from moatrader.api.service import SCHEMA_VERSION, FundamentalResearchService


def _default_data_root() -> Path:
    configured = os.getenv("MOATRADER_DATA_ROOT")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "data-lake"


def create_app(*, data_root: Path | None = None) -> FastAPI:
    repository = ResearchArtifactRepository(data_root or _default_data_root())
    service = FundamentalResearchService(repository)
    application = FastAPI(
        title="MoatRader Fundamental Research API",
        version="1.0.0",
        description=(
            "PIT evidence, model-routed valuation, reverse-DCF expectations, "
            "sensitivity, and thesis diagnostics. This is not an investment recommendation API."
        ),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:4173",
            "http://localhost:4173",
        ],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @application.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "schema_version": SCHEMA_VERSION}

    @application.get(
        "/api/research",
        response_model=ReportCatalog,
        tags=["fundamental-research"],
    )
    def list_reports() -> ReportCatalog:
        reports = []
        for result in service.catalog_results():
            dcf = result.get("dcf") or {}
            moat = result.get("moat_score") or {}
            reports.append(
                ReportSummary(
                    ticker=str(result["ticker"]),
                    issuer_name=str(result.get("issuer_name") or result["ticker"]),
                    as_of=datetime.fromisoformat(result["valuation_as_of"]).date(),
                    status=str(result.get("status", "UNKNOWN")),
                    moat_score=(
                        float(moat["economic_moat_score"])
                        if moat.get("economic_moat_score") is not None
                        else None
                    ),
                    current_price=(
                        float(result["current_price"])
                        if result.get("current_price") is not None
                        else None
                    ),
                )
            )
        return ReportCatalog(schema_version=SCHEMA_VERSION, reports=reports)

    @application.get(
        "/api/research/{ticker}",
        response_model=ResearchReport,
        response_model_exclude_none=False,
        tags=["fundamental-research"],
    )
    def get_report(
        ticker: str,
        as_of: date | None = Query(
            default=None,
            description="Optional PIT report cutoff (YYYY-MM-DD).",
        ),
    ) -> ResearchReport:
        try:
            return service.get_report(ticker.upper(), as_of=as_of)
        except ResearchArtifactNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @application.get("/api/current-research", tags=["all-security-research"])
    def list_current_reports() -> dict[str, object]:
        reports = repository.latest_current_reports()
        return {
            "schema_version": "kr-all-current-reports/catalog-1",
            "count": len(reports),
            "reports": [
                {
                    "ticker": report.get("ticker"),
                    "name": report.get("name"),
                    "market": report.get("market"),
                    "security_type": report.get("security_type"),
                    "cutoff": report.get("cutoff"),
                    "status": report.get("status"),
                    "overlay_action": (report.get("evidence_overlay") or {}).get("action"),
                    "valuation_status": (report.get("valuation") or {}).get("status"),
                }
                for report in reports
            ],
        }

    @application.get(
        "/api/current-research/{ticker}",
        tags=["all-security-research"],
    )
    def get_current_report(
        ticker: str,
        as_of: date | None = Query(
            default=None,
            description="Optional all-security report cutoff (YYYY-MM-DD).",
        ),
    ) -> dict[str, object]:
        try:
            return repository.load_current_report(ticker.upper(), as_of=as_of)
        except ResearchArtifactNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return application


app = create_app()


def main() -> None:
    uvicorn.run(
        "moatrader.api.app:app",
        host=os.getenv("MOATRADER_API_HOST", "127.0.0.1"),
        port=int(os.getenv("MOATRADER_API_PORT", "8010")),
        reload=False,
    )


if __name__ == "__main__":
    main()
