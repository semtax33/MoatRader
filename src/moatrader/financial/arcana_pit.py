from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from moatrader.backtest.universe_corrected import extract_arcana_annual_metrics


class ArcanaAnnualPitStore:
    """Read-only PIT adapter for Arcana's normalized annual DART cache."""

    def __init__(self, *, metadata_path: Path, snapshot_root: Path) -> None:
        self.metadata_path = metadata_path.resolve()
        self.snapshot_root = snapshot_root.resolve()
        metadata = pd.read_csv(
            self.metadata_path,
            dtype={"stock_code": str, "rcept_no": str},
            low_memory=False,
        )
        metadata["stock_code"] = metadata["stock_code"].str.zfill(6)
        metadata["report_date"] = pd.to_datetime(metadata["report_date"])
        annual = metadata[
            (metadata["source_type"] == "statement") & (metadata["fiscal_month"] == 12)
        ].copy()
        self.metadata = annual
        self.by_ticker = {
            str(ticker): frame.sort_values(["fiscal_year", "report_date", "rcept_no"])
            for ticker, frame in annual.groupby("stock_code", sort=False)
        }
        self.cache: dict[tuple[str, int], dict[str, float | int | None] | None] = {}

    def snapshot_path(self, ticker: str, fiscal_year: int) -> Path:
        return self.snapshot_root / f"kr_normalized_{ticker.zfill(6)}_{fiscal_year}.12.csv"

    def load(self, ticker: str, fiscal_year: int) -> dict[str, float | int | None] | None:
        key = (ticker.zfill(6), int(fiscal_year))
        if key in self.cache:
            return self.cache[key]
        path = self.snapshot_path(*key)
        if not path.exists():
            self.cache[key] = None
            return None
        try:
            metrics = extract_arcana_annual_metrics(pd.read_csv(path, low_memory=False))
        except Exception:
            metrics = None
        self.cache[key] = metrics
        return metrics

    def history(
        self,
        ticker: str,
        cutoff: date,
    ) -> tuple[list[tuple[int, dict[str, float | int | None]]], list[dict[str, Any]]]:
        ticker = ticker.zfill(6)
        ticker_rows = self.by_ticker.get(ticker)
        if ticker_rows is None:
            return [], []
        rows = ticker_rows[ticker_rows["report_date"] <= pd.Timestamp(cutoff)]
        rows = rows.drop_duplicates("fiscal_year", keep="last")
        history: list[tuple[int, dict[str, float | int | None]]] = []
        sources: list[dict[str, Any]] = []
        for row in rows.itertuples(index=False):
            year = int(row.fiscal_year)
            metrics = self.load(ticker, year)
            if metrics is None:
                continue
            history.append((year, metrics))
            sources.append(
                {
                    "fiscal_year": year,
                    "report_date": pd.Timestamp(row.report_date).date().isoformat(),
                    "rcept_no": str(row.rcept_no),
                    "source_url": str(row.source_url),
                    "snapshot_path": str(self.snapshot_path(ticker, year)),
                }
            )
        return history, sources

    def annual_coverage(self) -> list[dict[str, int]]:
        counts = self.metadata.groupby("fiscal_year").agg(
            report_count=("rcept_no", "size"),
            ticker_count=("stock_code", "nunique"),
        )
        return [
            {
                "fiscal_year": int(year),
                "report_count": int(row.report_count),
                "ticker_count": int(row.ticker_count),
            }
            for year, row in counts.iterrows()
        ]


class ArcanaPeriodicPitStore:
    """Read-only PIT adapter for Arcana's quarterly and annual DART cache.

    A signal date sees only reports whose ``report_date`` and period end are no
    later than the cutoff.  Comparable observations use the same fiscal month
    from the prior two fiscal years, so a Q1 disclosure is never compared with
    an annual observation.  The source cache is never modified.
    """

    def __init__(self, *, metadata_path: Path, snapshot_root: Path) -> None:
        self.metadata_path = metadata_path.resolve()
        self.snapshot_root = snapshot_root.resolve()
        metadata = pd.read_csv(
            self.metadata_path,
            dtype={"stock_code": str, "rcept_no": str},
            low_memory=False,
        )
        metadata["stock_code"] = metadata["stock_code"].str.zfill(6)
        metadata["report_date"] = pd.to_datetime(metadata["report_date"])
        metadata["period_end_date"] = pd.to_datetime(metadata["period_end_date"])
        statements = metadata[metadata["source_type"] == "statement"].copy()
        statements["fiscal_year"] = pd.to_numeric(
            statements["fiscal_year"], errors="raise"
        ).astype(int)
        statements["fiscal_month"] = pd.to_numeric(
            statements["fiscal_month"], errors="raise"
        ).astype(int)
        statements = statements[statements["fiscal_month"].isin([3, 6, 9, 12])]
        self.metadata = statements
        self.by_ticker = {
            str(ticker): frame.sort_values(
                ["period_end_date", "report_date", "rcept_no"], kind="stable"
            )
            for ticker, frame in statements.groupby("stock_code", sort=False)
        }
        self.cache: dict[
            tuple[str, int, int], dict[str, float | int | None] | None
        ] = {}

    def snapshot_path(self, ticker: str, fiscal_year: int, fiscal_month: int) -> Path:
        return self.snapshot_root / (
            f"kr_normalized_{ticker.zfill(6)}_{int(fiscal_year)}.{int(fiscal_month):02d}.csv"
        )

    def load(
        self,
        ticker: str,
        fiscal_year: int,
        fiscal_month: int,
    ) -> dict[str, float | int | None] | None:
        key = (ticker.zfill(6), int(fiscal_year), int(fiscal_month))
        if key in self.cache:
            return self.cache[key]
        path = self.snapshot_path(*key)
        if not path.exists():
            self.cache[key] = None
            return None
        try:
            metrics = extract_arcana_annual_metrics(pd.read_csv(path, low_memory=False))
        except Exception:
            metrics = None
        self.cache[key] = metrics
        return metrics

    @staticmethod
    def _source(row: Any, snapshot_path: Path) -> dict[str, Any]:
        return {
            "fiscal_year": int(row.fiscal_year),
            "fiscal_month": int(row.fiscal_month),
            "period_end_date": pd.Timestamp(row.period_end_date).date().isoformat(),
            "report_date": pd.Timestamp(row.report_date).date().isoformat(),
            "rcept_no": str(row.rcept_no),
            "source_url": str(row.source_url),
            "snapshot_path": str(snapshot_path),
        }

    def latest_with_comparables(
        self,
        ticker: str,
        cutoff: date,
        *,
        prior_years: int = 2,
    ) -> tuple[
        list[tuple[int, int, dict[str, float | int | None]]],
        list[dict[str, Any]],
    ]:
        """Return latest visible period followed by same-period prior years."""

        if prior_years < 0:
            raise ValueError("prior_years must be non-negative")
        ticker = ticker.zfill(6)
        ticker_rows = self.by_ticker.get(ticker)
        if ticker_rows is None:
            return [], []
        cutoff_ts = pd.Timestamp(cutoff)
        visible = ticker_rows[
            (ticker_rows["report_date"] <= cutoff_ts)
            & (ticker_rows["period_end_date"] <= cutoff_ts)
        ].copy()
        if visible.empty:
            return [], []
        visible = visible.drop_duplicates(
            ["fiscal_year", "fiscal_month"], keep="last"
        )
        latest = visible.sort_values(
            ["period_end_date", "report_date", "rcept_no"], kind="stable"
        ).iloc[-1]
        latest_year = int(latest["fiscal_year"])
        month = int(latest["fiscal_month"])
        observations: list[tuple[int, int, dict[str, float | int | None]]] = []
        sources: list[dict[str, Any]] = []
        for year in range(latest_year, latest_year - prior_years - 1, -1):
            candidates = visible[
                (visible["fiscal_year"] == year) & (visible["fiscal_month"] == month)
            ]
            if candidates.empty:
                continue
            row = candidates.sort_values(
                ["report_date", "rcept_no"], kind="stable"
            ).iloc[-1]
            metrics = self.load(ticker, year, month)
            if metrics is None:
                continue
            observations.append((year, month, metrics))
            sources.append(
                self._source(
                    row,
                    self.snapshot_path(ticker, year, month),
                )
            )
        return observations, sources

    def periodic_coverage(self) -> list[dict[str, int]]:
        counts = self.metadata.groupby(["fiscal_year", "fiscal_month"]).agg(
            report_count=("rcept_no", "size"),
            ticker_count=("stock_code", "nunique"),
        )
        return [
            {
                "fiscal_year": int(year),
                "fiscal_month": int(month),
                "report_count": int(row.report_count),
                "ticker_count": int(row.ticker_count),
            }
            for (year, month), row in counts.iterrows()
        ]
