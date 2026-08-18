from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.experiments.integrity import sha256_file


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _key(row: dict[str, str]) -> tuple[str, str]:
    return str(row.get("date", "")).strip(), str(row.get("ticker", "")).strip()


class MissingnessBreakdown(ContractModel):
    method: str
    applicability_status: str
    missing_fields: list[str]
    count: int = Field(gt=0)


class CoverageReport(ContractModel):
    schema_version: str = "v7-coverage-research/1"
    row_count: int = Field(gt=0)
    router_eligible_count: int = Field(ge=0)
    valuation_generated_count: int = Field(ge=0)
    rank_eligible_count: int = Field(ge=0)
    invalid_valuation_count: int = Field(ge=0)
    model_not_applicable_count: int = Field(ge=0)
    router_eligible_rate: float = Field(ge=0, le=1)
    cheap_rank_coverage_rate: float = Field(ge=0, le=1)
    method_route_counts: dict[str, int]
    method_rank_eligible_counts: dict[str, int]
    method_invalid_valuation_counts: dict[str, int]
    missing_input_counts: dict[str, int]
    missingness: list[MissingnessBreakdown]
    source_sha256: dict[str, str]
    return_data_accessed: bool = False

    @model_validator(mode="after")
    def accounting_is_complete(self) -> "CoverageReport":
        if self.return_data_accessed:
            raise ValueError("v7 coverage research must not access return data")
        if self.rank_eligible_count + self.invalid_valuation_count + self.model_not_applicable_count != self.row_count:
            raise ValueError("coverage states must partition all rows")
        if self.valuation_generated_count != self.rank_eligible_count + self.invalid_valuation_count:
            raise ValueError("generated valuations must partition into valid and invalid ranks")
        return self


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build_coverage_report(*, routing_path: Path, signals_path: Path) -> CoverageReport:
    routing = _read_csv(routing_path)
    signals = _read_csv(signals_path)
    route_by_key = {_key(row): row for row in routing}
    signal_by_key = {_key(row): row for row in signals}
    if len(route_by_key) != len(routing) or len(signal_by_key) != len(signals):
        raise ValueError("routing and signals must have unique date/ticker rows")
    if set(route_by_key) != set(signal_by_key):
        missing_signal = sorted(set(route_by_key) - set(signal_by_key))[:5]
        missing_route = sorted(set(signal_by_key) - set(route_by_key))[:5]
        raise ValueError(
            f"routing/signals row mismatch; missing signals={missing_signal}, missing routes={missing_route}"
        )

    method_routes: Counter[str] = Counter()
    method_valid: Counter[str] = Counter()
    method_invalid: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    missing_groups: Counter[tuple[str, str, tuple[str, ...]]] = Counter()
    router_eligible = 0
    valuation_generated = 0
    rank_eligible = 0
    invalid_valuation = 0
    model_not_applicable = 0

    for key in sorted(route_by_key):
        route = route_by_key[key]
        signal = signal_by_key[key]
        method = str(route.get("primary_method") or signal.get("method") or "UNKNOWN")
        method_routes[method] += 1
        if str(route.get("applicability_status", "")).strip() == "ELIGIBLE":
            router_eligible += 1
        if _truthy(route.get("valuation_generated")):
            valuation_generated += 1
        status = str(signal.get("alpha_status", "")).strip()
        if _truthy(signal.get("rank_eligible")):
            rank_eligible += 1
            method_valid[method] += 1
        elif status == "INVALID_VALUATION":
            invalid_valuation += 1
            method_invalid[method] += 1
        else:
            model_not_applicable += 1
            fields = tuple(
                sorted(
                    field.strip()
                    for field in str(route.get("missing_fields", "")).split(";")
                    if field.strip()
                )
            )
            for field in fields:
                missing_fields[field] += 1
            missing_groups[(method, str(route.get("applicability_status", "")), fields)] += 1

    row_count = len(routing)
    return CoverageReport(
        row_count=row_count,
        router_eligible_count=router_eligible,
        valuation_generated_count=valuation_generated,
        rank_eligible_count=rank_eligible,
        invalid_valuation_count=invalid_valuation,
        model_not_applicable_count=model_not_applicable,
        router_eligible_rate=router_eligible / row_count,
        cheap_rank_coverage_rate=rank_eligible / row_count,
        method_route_counts=dict(sorted(method_routes.items())),
        method_rank_eligible_counts=dict(sorted(method_valid.items())),
        method_invalid_valuation_counts=dict(sorted(method_invalid.items())),
        missing_input_counts=dict(sorted(missing_fields.items(), key=lambda item: (-item[1], item[0]))),
        missingness=[
            MissingnessBreakdown(
                method=method,
                applicability_status=applicability,
                missing_fields=list(fields),
                count=count,
            )
            for (method, applicability, fields), count in sorted(
                missing_groups.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        source_sha256={
            "routing.csv": sha256_file(routing_path),
            "signals.csv": sha256_file(signals_path),
        },
    )


def write_coverage_artifacts(report: CoverageReport, *, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "coverage.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_directory / "missingness.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["method", "applicability_status", "missing_fields", "count"],
        )
        writer.writeheader()
        for item in report.missingness:
            writer.writerow(
                {
                    "method": item.method,
                    "applicability_status": item.applicability_status,
                    "missing_fields": ";".join(item.missing_fields),
                    "count": item.count,
                }
            )
