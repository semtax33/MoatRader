from __future__ import annotations

import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from moatrader.expectations.historical_evidence import sha256_file
from scripts.prepare_historical_evidence_index_eri_inputs_v2 import (
    _git_blob_sha256,
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
