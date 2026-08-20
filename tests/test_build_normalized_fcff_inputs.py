import hashlib
import json
from pathlib import Path

from scripts.build_normalized_fcff_inputs import (
    build_normalized_input,
    reconstruct_annual_history,
)
from moatrader.valuation import RoutedValuationExecutor, ValuationMethod


def _write_payload(path: Path, year: int, revenue: int, ebit: int) -> str:
    rows = []
    for field, multiplier in (
        ("bfefrmtrm_amount", 1),
        ("frmtrm_amount", 2),
        ("thstrm_amount", 3),
    ):
        rows.extend(
            [
                {
                    "bsns_year": str(year),
                    "reprt_code": "11011",
                    "sj_div": "IS",
                    "account_id": "ifrs-full_Revenue",
                    "account_nm": "Revenue",
                    field: str(revenue * multiplier),
                },
                {
                    "bsns_year": str(year),
                    "reprt_code": "11011",
                    "sj_div": "IS",
                    "account_id": "ifrs-full_OperatingIncomeLoss",
                    "account_nm": "OperatingIncomeLoss",
                    field: str(ebit * multiplier),
                },
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "000", "list": rows}), encoding="utf-8"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    date_dir = tmp_path / "date-inputs" / "2025-08-31"
    dcf_path = date_dir / "dcf-inputs" / "000001.json"
    sources = []
    for year in (2022, 2023, 2024):
        payload_path = date_dir / "source" / "financials" / "000001" / f"{year}-11011-CFS.json"
        digest = _write_payload(payload_path, year, 100 * (year - 2019), 10 * (year - 2019))
        sources.append(
            {
                "business_year": year,
                "report_code": "11011",
                "receipt_no": f"R{year}",
                "available_at": f"{year + 1}-03-20T00:00:00+09:00",
                "fs_div": "CFS",
                "payload_path": str(payload_path),
                "payload_sha256": digest,
            }
        )
    dcf = {
        "metrics": {"revenue": "1000", "ebit": "100"},
        "annual_sources": sources,
        "assumptions": {
            "base_period": "2025H1",
            "tax_rate": "0.24",
            "wacc": "0.10",
            "terminal_growth": "0.02",
            "net_debt": "100",
            "diluted_shares": "10",
        },
    }
    dcf_path.parent.mkdir(parents=True, exist_ok=True)
    dcf_path.write_text(json.dumps(dcf), encoding="utf-8")
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "issuer_id": "CORP1",
                "series": [
                    {"concept": "TOTAL_EQUITY", "points": [{"period": "2025-06-30", "value": "800"}]},
                    {"concept": "TOTAL_DEBT", "points": [{"period": "2025-06-30", "value": "300"}]},
                    {"concept": "CASH", "points": [{"period": "2025-06-30", "value": "200"}]},
                ],
            }
        ),
        encoding="utf-8",
    )
    return dcf_path, snapshot_path, dcf


def test_reconstructs_five_years_from_comparative_columns(tmp_path: Path) -> None:
    dcf_path, _snapshot_path, dcf = _fixture(tmp_path)
    result = reconstruct_annual_history(
        dcf, dcf_path=dcf_path, ticker="000001"
    )

    assert [item.fiscal_year for item in result.observations] == [
        2020,
        2021,
        2022,
        2023,
        2024,
    ]
    # Later filings replace comparative values for overlapping years.
    assert result.observations[-2].source_refs[0].startswith("OPENDART:R2024:2023")
    assert result.excluded_fiscal_years == [2018, 2019]


def test_builds_valid_routed_input_and_real_engine_value(tmp_path: Path) -> None:
    dcf_path, snapshot_path, _dcf = _fixture(tmp_path)
    envelope = build_normalized_input(
        ticker="000001",
        as_of="2025-08-31",
        dcf_path=dcf_path,
        snapshot_path=snapshot_path,
    )
    prepared = RoutedValuationExecutor.prepare(envelope)

    assert envelope.method == ValuationMethod.NORMALIZED_FCFF
    assert prepared.actual_engine == "NormalizedFcffEngine"
    assert prepared.assumptions.normalization.included_fiscal_years == list(
        range(2020, 2025)
    )
    assert "NO_LLM:DETERMINISTIC_BUILDER" in prepared.assumptions.provenance
