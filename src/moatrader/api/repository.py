from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


_TICKER_RE = re.compile(r"^[0-9A-Za-z._-]{1,32}$")


class ResearchArtifactNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class ResearchArtifacts:
    directory: Path
    result: dict[str, Any]
    dossier: dict[str, Any]
    moat_score: dict[str, Any]
    dcf_assumptions: dict[str, Any]
    financial_snapshot: dict[str, Any]
    run_manifest: dict[str, Any]
    evidence_ledger: dict[str, Any]

    @property
    def valuation_at(self) -> datetime:
        return datetime.fromisoformat(self.result["valuation_as_of"])


class ResearchArtifactRepository:
    """Read-only adapter over immutable MoatRader company run artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _paths(self, ticker: str | None = None) -> list[Path]:
        if ticker is not None:
            if not _TICKER_RE.fullmatch(ticker):
                raise ValueError("ticker contains unsupported characters")
            leaf = f"companies/{ticker}/result.json"
        else:
            leaf = "companies/*/result.json"
        patterns = (
            f"backtests/*/runs/*/{leaf}",
            f"runs/*/{leaf}",
            leaf,
        )
        paths: list[Path] = []
        seen: set[Path] = set()
        for pattern in patterns:
            for candidate in self.root.glob(pattern):
                resolved = candidate.resolve()
                if resolved not in seen:
                    paths.append(candidate)
                    seen.add(resolved)
        return paths

    def _current_report_paths(self, ticker: str | None = None) -> list[Path]:
        if ticker is not None:
            if not _TICKER_RE.fullmatch(ticker):
                raise ValueError("ticker contains unsupported characters")
            leaf = f"reports/{ticker}/report.json"
        else:
            leaf = "reports/*/report.json"
        patterns = (
            f"backtests/*/research-reports/*/{leaf}",
            f"research-reports/*/{leaf}",
            leaf,
        )
        paths: list[Path] = []
        seen: set[Path] = set()
        for pattern in patterns:
            for candidate in self.root.glob(pattern):
                resolved = candidate.resolve()
                if resolved not in seen:
                    paths.append(candidate)
                    seen.add(resolved)
        return paths

    @staticmethod
    def _read(path: Path, *, required: bool = True) -> dict[str, Any]:
        if not path.exists():
            if required:
                raise ResearchArtifactNotFoundError(f"required artifact is missing: {path.name}")
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _candidates(self, ticker: str) -> list[tuple[datetime, Path, dict[str, Any]]]:
        found: list[tuple[datetime, Path, dict[str, Any]]] = []
        for result_path in self._paths(ticker):
            try:
                result = self._read(result_path)
                if result.get("status") != "COMPLETE" or not result.get("valuation_as_of"):
                    continue
                found.append(
                    (
                        datetime.fromisoformat(result["valuation_as_of"]),
                        result_path.parent,
                        result,
                    )
                )
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
        return sorted(found, key=lambda item: (item[0], str(item[1])))

    @staticmethod
    def _selection_quality(directory: Path, result: dict[str, Any]) -> tuple[Any, ...]:
        manifest: dict[str, Any] = {}
        manifest_path = directory / "run-manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
        coverage = (
            (result.get("moat_score") or {})
            .get("document_coverage", {})
            .get("moat_evidence_coverage")
        )
        return (
            manifest.get("model") == "gpt-5.6-luna",
            coverage is not None,
            float(coverage or 0),
            int(result.get("evidence_count") or 0),
            str(manifest.get("created_at") or ""),
            str(directory),
        )

    def _select(self, candidates: list[tuple[datetime, Path, dict[str, Any]]]) -> tuple[datetime, Path, dict[str, Any]]:
        latest_date = max(item[0].date() for item in candidates)
        same_date = [item for item in candidates if item[0].date() == latest_date]
        return max(same_date, key=lambda item: self._selection_quality(item[1], item[2]))

    def load(self, ticker: str, *, as_of: date | None = None) -> ResearchArtifacts:
        candidates = self._candidates(ticker.upper())
        if as_of is not None:
            candidates = [item for item in candidates if item[0].date() <= as_of]
        if not candidates:
            suffix = f" as of {as_of.isoformat()}" if as_of else ""
            raise ResearchArtifactNotFoundError(f"completed research report not found for {ticker}{suffix}")
        _, directory, result = self._select(candidates)
        return ResearchArtifacts(
            directory=directory,
            result=result,
            dossier=self._read(directory / "dossier.json"),
            moat_score=self._read(directory / "moat-score.json"),
            dcf_assumptions=self._read(directory / "dcf-assumptions.json"),
            financial_snapshot=self._read(directory / "financial-snapshot.json"),
            run_manifest=self._read(directory / "run-manifest.json"),
            evidence_ledger=self._read(directory / "evidence-ledger-snapshot.json", required=False),
        )

    def previous(self, ticker: str, before: datetime) -> ResearchArtifacts | None:
        candidates = [item for item in self._candidates(ticker.upper()) if item[0].date() < before.date()]
        if not candidates:
            return None
        return self.load(ticker, as_of=max(item[0].date() for item in candidates))

    def latest_results(self) -> list[dict[str, Any]]:
        grouped: dict[str, list[tuple[datetime, Path, dict[str, Any]]]] = {}
        for result_path in self._paths():
            try:
                result = self._read(result_path)
                if result.get("status") != "COMPLETE" or not result.get("valuation_as_of"):
                    continue
                ticker = str(result["ticker"]).upper()
                at = datetime.fromisoformat(result["valuation_as_of"])
                grouped.setdefault(ticker, []).append((at, result_path.parent, result))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
        selected = [self._select(items)[2] for items in grouped.values()]
        return sorted(selected, key=lambda item: item["ticker"])

    @staticmethod
    def _current_report_cutoff(report: dict[str, Any]) -> datetime:
        cutoff = str(report.get("cutoff") or "").strip()
        if not cutoff:
            raise ValueError("current report has no cutoff")
        parsed = datetime.fromisoformat(cutoff)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("current report cutoff must be timezone-aware")
        return parsed

    def load_current_report(
        self,
        ticker: str,
        *,
        as_of: date | None = None,
    ) -> dict[str, Any]:
        candidates: list[tuple[datetime, Path, dict[str, Any]]] = []
        for path in self._current_report_paths(ticker.upper()):
            try:
                report = self._read(path)
                cutoff = self._current_report_cutoff(report)
                if as_of is not None and cutoff.date() > as_of:
                    continue
                candidates.append((cutoff, path, report))
            except (ValueError, json.JSONDecodeError):
                continue
        if not candidates:
            suffix = f" as of {as_of.isoformat()}" if as_of else ""
            raise ResearchArtifactNotFoundError(
                f"current all-security report not found for {ticker}{suffix}"
            )
        return max(candidates, key=lambda item: (item[0], str(item[1])))[2]

    def latest_current_reports(self) -> list[dict[str, Any]]:
        grouped: dict[str, list[tuple[datetime, Path, dict[str, Any]]]] = {}
        for path in self._current_report_paths():
            try:
                report = self._read(path)
                ticker = str(report["ticker"]).upper()
                cutoff = self._current_report_cutoff(report)
                grouped.setdefault(ticker, []).append((cutoff, path, report))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
        selected = [
            max(items, key=lambda item: (item[0], str(item[1])))[2]
            for items in grouped.values()
        ]
        return sorted(selected, key=lambda item: str(item["ticker"]))
