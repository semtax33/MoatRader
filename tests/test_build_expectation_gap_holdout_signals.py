from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from scripts.build_expectation_gap_holdout_signals import main


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_builder_assembles_and_normalizes_all_150_return_blind_signals(
    tmp_path: Path, monkeypatch
) -> None:
    as_of = "2026-08-31"
    universe_path = tmp_path / "universe.csv"
    routing_path = tmp_path / "routing.csv"
    valuation_path = tmp_path / "valuation.csv"
    research_path = tmp_path / "research.json"
    output_path = tmp_path / "signals.json"
    universe: list[dict[str, object]] = []
    routing: list[dict[str, object]] = []
    valuation: list[dict[str, object]] = []
    research: list[dict[str, object]] = []
    for index in range(150):
        code = f"{index:06d}"
        fair_value = 100 + index
        universe.append({"stock_code": code, "name": f"Issuer {index}"})
        routing.append(
            {
                "date": as_of,
                "ticker": code,
                "issuer_name": f"Issuer {index}",
                "sector": "Industrials",
                "sector_evidence_ref": f"KRX:MDCSTAT03901:{as_of}:{code}",
                "profile_sha256": f"{index:064x}",
            }
        )
        valuation.append(
            {
                "date": as_of,
                "ticker": code,
                "method": "ECONOMIC_FCFF",
                "economic_archetype": "GENERAL_OPERATING",
                "market_price": "100",
                "primary_fair_value_per_share": str(fair_value),
                "raw_value_gap": str(fair_value / 100 - 1),
                "alpha_status": "VALID",
            }
        )
        research.append(
            {
                "ticker": code,
                "risk": {
                    "fragility_score": 20,
                    "three_p": {
                        "possible": "PASS",
                        "plausible": "IN_RANGE",
                        "probable": "SUPPORTED",
                        "hard_gate_pass": True,
                        "review_required": False,
                    },
                    "industry_counterevidence_count": 0,
                    "industry_range_widener_count": 0,
                    "industry_evidence_available": True,
                },
                "confirmation": {
                    "improving": None,
                    "status": "INSUFFICIENT_EVIDENCE",
                },
                "source_references": [
                    {
                        "document_id": f"DART:{code}:2026Q2",
                        "source_type": "DART",
                        "available_at": "2026-08-14T09:00:00+09:00",
                    }
                ],
            }
        )
    _write_csv(universe_path, universe)
    _write_csv(routing_path, routing)
    _write_csv(valuation_path, valuation)
    research_path.write_text(json.dumps(research), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_expectation_gap_holdout_signals.py",
            "--as-of",
            as_of,
            "--universe",
            str(universe_path),
            "--routing",
            str(routing_path),
            "--valuation-signals",
            str(valuation_path),
            "--research-inputs",
            str(research_path),
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(result) == 150
    assert result[0]["alpha"]["cheap"]["method_archetype_percentile"] == 0
    assert result[-1]["alpha"]["cheap"]["method_archetype_percentile"] == 100
