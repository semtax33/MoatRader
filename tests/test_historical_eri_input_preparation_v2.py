from __future__ import annotations

import subprocess
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from moatrader.expectations.historical_evidence import sha256_file
from scripts.prepare_historical_evidence_index_eri_inputs_v2 import (
    _git_blob_sha256,
    _label_replication_coverage_gate,
    extract_pit_economic_metrics_from_html,
)
from scripts.seal_historical_evidence_index_eri_feature_panel_v2 import (
    _assert_pre_outcome,
)


def test_extract_pit_economic_metrics_scales_and_sums_debt() -> None:
    document = """
    <html><body>
      <p>(단위: 백만원)</p>
      <table>
        <tr><td>매출액</td><td>1,000</td></tr>
        <tr><td>영업이익</td><td>100</td></tr>
      </table>
      <table>
        <tr><td>자산총계</td><td>2,000</td></tr>
        <tr><td>자본총계</td><td>1,200</td></tr>
        <tr><td>현금및현금성자산</td><td>200</td></tr>
        <tr><td>단기차입금</td><td>100</td></tr>
        <tr><td>장기차입금</td><td>300</td></tr>
      </table>
    </body></html>
    """

    assert extract_pit_economic_metrics_from_html(document) == {
        "revenue": Decimal("1000000000"),
        "operating_profit": Decimal("100000000"),
        "total_assets": Decimal("2000000000"),
        "total_equity": Decimal("1200000000"),
        "cash": Decimal("200000000"),
        "debt": Decimal("400000000"),
    }


def test_extract_pit_economic_metrics_accepts_standard_dart_parenthesized_labels() -> None:
    document = """
    <html><body>
      <p>(단위: 원)</p>
      <table>
        <tr><td>수익(매출액)</td><td>1,000</td></tr>
        <tr><td>영업이익(손실)</td><td>100</td></tr>
      </table>
      <table>
        <tr><td>자산총계</td><td>2,000</td></tr>
        <tr><td>자본총계</td><td>1,200</td></tr>
        <tr><td>현금및현금성자산</td><td>200</td></tr>
        <tr><td>차입금및사채</td><td>400</td></tr>
        <tr><td>단기차입금</td><td>100</td></tr>
      </table>
    </body></html>
    """

    result = extract_pit_economic_metrics_from_html(document)
    assert result["revenue"] == Decimal("1000")
    assert result["operating_profit"] == Decimal("100")
    # Aggregate fallback is used only because no original debt label matched;
    # this fixture deliberately includes an original component, so it wins and
    # prevents aggregate/component double counting.
    assert result["debt"] == Decimal("100")


def test_extract_pit_economic_metrics_uses_debt_aggregate_fallback_without_components() -> None:
    document = """
    <html><body>
      <p>(단위: 원)</p>
      <table>
        <tr><td>매출액</td><td>1,000</td></tr>
        <tr><td>영업이익</td><td>100</td></tr>
      </table>
      <table>
        <tr><td>자산총계</td><td>2,000</td></tr>
        <tr><td>자본총계</td><td>1,200</td></tr>
        <tr><td>현금및현금성자산</td><td>200</td></tr>
        <tr><td>차입금및사채</td><td>400</td></tr>
      </table>
    </body></html>
    """

    assert extract_pit_economic_metrics_from_html(document)["debt"] == Decimal("400")


def test_extract_pit_economic_metrics_sums_conservative_current_noncurrent_debt_aliases() -> None:
    document = """
    <html><body>
      <p>(단위: 원)</p>
      <table>
        <tr><td>매출액</td><td>1,000</td></tr>
        <tr><td>영업이익</td><td>100</td></tr>
      </table>
      <table>
        <tr><td>자산총계</td><td>2,000</td></tr>
        <tr><td>자본총계</td><td>1,200</td></tr>
        <tr><td>현금및현금성자산</td><td>200</td></tr>
        <tr><td>차입금(유동)</td><td>125</td></tr>
        <tr><td>차입금(비유동)</td><td>275</td></tr>
        <tr><td>리스부채(유동)</td><td>25</td></tr>
        <tr><td>리스부채(비유동)</td><td>75</td></tr>
      </table>
    </body></html>
    """

    assert extract_pit_economic_metrics_from_html(document)["debt"] == Decimal("500")


def test_extract_pit_economic_metrics_prefers_debt_total_over_fallback_components() -> None:
    document = """
    <html><body>
      <p>(단위: 원)</p>
      <table>
        <tr><td>매출액</td><td>1,000</td></tr>
        <tr><td>영업이익</td><td>100</td></tr>
      </table>
      <table>
        <tr><td>자산총계</td><td>2,000</td></tr>
        <tr><td>자본총계</td><td>1,200</td></tr>
        <tr><td>현금및현금성자산</td><td>200</td></tr>
        <tr><td>전환사채총액</td><td>400</td></tr>
        <tr><td>유동성전환사채</td><td>100</td></tr>
      </table>
    </body></html>
    """

    assert extract_pit_economic_metrics_from_html(document)["debt"] == Decimal("400")


def test_label_replication_coverage_gate_is_outcome_blind_and_size_aware() -> None:
    seoul = ZoneInfo("Asia/Seoul")
    rows = {
        "SMALL_1": SimpleNamespace(
            issuer_id="000001", signal_timestamp=datetime(2024, 1, 2, 8, 0, tzinfo=seoul)
        ),
        "MID_1": SimpleNamespace(
            issuer_id="000002", signal_timestamp=datetime(2024, 2, 2, 8, 0, tzinfo=seoul)
        ),
        "LARGE_1": SimpleNamespace(
            issuer_id="000003", signal_timestamp=datetime(2024, 3, 2, 8, 0, tzinfo=seoul)
        ),
    }
    sizes = {
        row.signal_timestamp.date(): {row.issuer_id: observation_id.split("_")[0]}
        for observation_id, row in rows.items()
    }
    contract = {
        "pre_outcome_coverage_gate": {
            "minimum_reverse_expectation_count": 3,
            "minimum_potential_t63_label_count": 3,
            "minimum_small_reverse_expectation_count": 1,
            "minimum_mid_reverse_expectation_count": 1,
            "minimum_unique_reverse_expectation_issuers": 3,
            "minimum_signal_month_count_with_potential_labels": 3,
            "maximum_large_share_of_potential_labels": 0.34,
            "selected_without_expanded_future_eri_values": True,
        }
    }

    result = _label_replication_coverage_gate(
        contract,
        expectation_ids=set(rows),
        complete_inventory_ids=set(rows),
        full_rows=rows,  # type: ignore[arg-type]
        signal_sizes=sizes,
    )

    assert result["status"] == "PASS"
    assert result["selected_without_expanded_future_eri_values"] is True
    assert result["metrics"]["large_share_of_potential_labels"] == pytest.approx(1 / 3)
    assert result["reverse_expectation_size_counts"] == {"LARGE": 1, "MID": 1, "SMALL": 1}


def test_pre_outcome_feature_panel_rejects_future_labels_recursively() -> None:
    with pytest.raises(ValueError, match="future_eri"):
        _assert_pre_outcome(
            [
                {
                    "observation_id": "OBS_1",
                    "nested": [{"future_eri": "0.25"}],
                }
            ]
        )


def test_pre_outcome_feature_panel_allows_target_metadata_without_value() -> None:
    _assert_pre_outcome(
        [
            {
                "observation_id": "OBS_1",
                "target_session": "2025-03-31",
                "target_price_at": "2025-03-31T15:30:00+09:00",
                "target_price_source_id": "MARCAP:hash:2025-03-31:000001:CLOSE",
                "outcome_values_included": False,
            }
        ]
    )


def test_recorded_pre_outcome_script_hash_matches_git_blob() -> None:
    workspace = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    repository_path = "src/moatrader/expectations/future_eri.py"
    assert _git_blob_sha256(
        workspace,
        commit=commit,
        repository_path=repository_path,
    ) == sha256_file(workspace / repository_path)
