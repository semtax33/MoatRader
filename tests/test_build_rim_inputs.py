from decimal import Decimal
from pathlib import Path

from scripts.build_rim_inputs import _decimal_text, _context_value


D = Decimal


def test_dart_numeric_cell_applies_reported_thousand_won_scale() -> None:
    assert _decimal_text("3,153,514,080", negated=False, scale=3) == D(
        "3153514080000"
    )
    assert _decimal_text("(34,482,138)", negated=False, scale=3) == D(
        "-34482138000"
    )


def test_context_value_prefers_undimensioned_consolidated_cell() -> None:
    cells = [
        (
            "ifrs-full_ProfitLoss",
            "CFY2026dHYA_ifrs-full_ConsolidatedMember_axis_member",
            D("90"),
        ),
        (
            "ifrs-full_ProfitLoss",
            "CFY2026dHYA_ifrs-full_ConsolidatedMember",
            D("100"),
        ),
    ]

    assert _context_value(
        cells,
        account="ifrs-full_ProfitLoss",
        period_pattern=r"^CFY\d+dHYA(?:_|$)",
    ) == D("100")
