from __future__ import annotations

from decimal import Decimal

import pandas as pd

from moatrader.expectations.driver_signals import DriverName, supported_driver_estimate
from moatrader.expectations.revision import (
    RevisionStatus,
    SurfaceStatus,
    assumptions_with_driver,
    driver_sensitivities,
    dynamic_implied_revision,
    expectation_surface_revision,
    periodic_value_factor_vector,
    turbo_driver,
)
from moatrader.financial.arcana_pit import ArcanaPeriodicPitStore
from moatrader.valuation.economic_dcf import EconomicDcfEngine


D = Decimal


def annual_history() -> list[tuple[int, dict[str, Decimal | None]]]:
    return [
        (
            2020,
            {
                "revenue": D("1000"),
                "ebit": D("120"),
                "cash": D("40"),
                "debt": D("100"),
                "total_equity": D("500"),
            },
        ),
        (
            2021,
            {
                "revenue": D("1120"),
                "ebit": D("145"),
                "cash": D("50"),
                "debt": D("105"),
                "total_equity": D("550"),
            },
        ),
        (
            2022,
            {
                "revenue": D("1260"),
                "ebit": D("176"),
                "cash": D("60"),
                "debt": D("110"),
                "total_equity": D("610"),
            },
        ),
    ]


def test_periodic_value_factor_vector_preserves_driver_components() -> None:
    periods = [
        (2022, 3, {"revenue": 144, "ebit": 21.6, "total_equity": 100, "debt": 10, "cash": 10}),
        (2021, 3, {"revenue": 120, "ebit": 15, "total_equity": 80, "debt": 10, "cash": 10}),
        (2020, 3, {"revenue": 100, "ebit": 10, "total_equity": 60, "debt": 10, "cash": 10}),
    ]
    vector = periodic_value_factor_vector(periods, wacc=D("0.10"))

    assert vector.growth_yoy == D("0.2")
    assert vector.growth_acceleration is not None
    assert abs(vector.growth_acceleration) < D("1e-12")
    assert vector.nopat_margin_change is not None and vector.nopat_margin_change > 0
    assert vector.operating_leverage_spread is not None and vector.operating_leverage_spread > 0
    assert vector.roiic_change is not None and vector.roiic_change > 0
    assert vector.incremental_sales_efficiency_change == D("0.2")
    assert vector.roic_spread_change is not None and vector.roic_spread_change > 0
    assert vector.positive_roic_spread_persistence == D("1")
    assert set(vector.components_for(DriverName.GROWTH)) == {
        "growth_yoy",
        "growth_acceleration",
    }


def test_periodic_store_is_cutoff_safe_and_same_month_comparable(tmp_path) -> None:
    metadata_rows = []
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    for year, report_date, revenue in (
        (2020, "2020-05-15", 100),
        (2021, "2021-05-15", 110),
        (2022, "2022-05-16", 125),
        (2023, "2023-05-15", 140),
    ):
        metadata_rows.append(
            {
                "security_id": "KRX:000001",
                "stock_code": "000001",
                "fiscal_year": year,
                "fiscal_month": 3,
                "period_end_date": f"{year}-03-31",
                "report_date": report_date,
                "rcept_no": f"{year}0001",
                "report_name": "Q1",
                "source_type": "statement",
                "source_url": f"https://example.test/{year}",
                "updated_at": report_date,
            }
        )
        pd.DataFrame(
            {
                "canonical_account_id": [
                    "REVENUE",
                    "OPERATING_INCOME",
                    "TOTAL_EQUITY",
                    "CASH_AND_EQUIVALENTS",
                    "LONG_TERM_DEBT",
                ],
                "normalized_amount": [revenue, revenue * 0.1, 50, 5, 10],
                "fiscal_year": [year] * 5,
            }
        ).to_csv(snapshot_root / f"kr_normalized_000001_{year}.03.csv", index=False)
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata, index=False)
    store = ArcanaPeriodicPitStore(metadata_path=metadata, snapshot_root=snapshot_root)

    periods, sources = store.latest_with_comparables(
        "000001",
        pd.Timestamp("2023-05-14").date(),
    )
    assert [(year, month) for year, month, _metrics in periods] == [
        (2022, 3),
        (2021, 3),
        (2020, 3),
    ]
    assert all(source["report_date"] <= "2023-05-14" for source in sources)


def test_turbo_driver_is_largest_positive_normalized_sensitivity() -> None:
    estimate = supported_driver_estimate(
        annual_history(), size_bucket="MID", diluted_shares=D("10")
    )
    sensitivities = driver_sensitivities(estimate.assumptions())
    selected = turbo_driver(sensitivities)
    assert selected is not None
    eligible = [item for item in sensitivities.values() if item.eligible]
    assert sensitivities[selected].absolute_price_change_per_shock == max(
        item.absolute_price_change_per_shock for item in eligible
    )


def test_dynamic_revision_uses_one_curve_and_recovers_continuous_branch() -> None:
    estimate = supported_driver_estimate(
        annual_history(), size_bucket="MID", diluted_shares=D("10")
    )
    base = estimate.assumptions()
    engine = EconomicDcfEngine()
    entry_driver = D("0.08")
    target_driver = D("0.12")
    entry_price = engine.value(
        assumptions_with_driver(base, DriverName.GROWTH, entry_driver)
    ).fair_value_per_share
    target_price = engine.value(
        assumptions_with_driver(base, DriverName.GROWTH, target_driver)
    ).fair_value_per_share

    revision = dynamic_implied_revision(
        base=base,
        driver=DriverName.GROWTH,
        entry_price=entry_price,
        target_price=target_price,
    )
    assert revision.status == RevisionStatus.SOLVED
    assert revision.entry_implied is not None
    assert revision.target_implied is not None
    assert abs(revision.entry_implied - entry_driver) < D("0.005")
    assert abs(revision.target_implied - target_driver) < D("0.01")
    assert revision.implied_revision is not None and revision.implied_revision > 0


def test_multidimensional_surface_retains_region_and_emits_all_driver_revisions() -> None:
    estimate = supported_driver_estimate(
        annual_history(), size_bucket="MID", diluted_shares=D("10")
    )
    base = estimate.assumptions()
    engine = EconomicDcfEngine()
    entry_price = engine.value(
        base.model_copy(
            update={
                "revenue_growth": D("0.10"),
                "target_nopat_margin": D("0.15"),
                "roiic": D("0.25"),
                "competitive_advantage_period_years": 10,
                "explicit_forecast_years": 15,
            }
        )
    ).fair_value_per_share
    target_price = entry_price * D("1.05")
    surface = expectation_surface_revision(
        base=base,
        entry_price=entry_price,
        target_price=target_price,
    )

    assert surface.status == SurfaceStatus.SOLVED
    assert surface.entry is not None and surface.entry.effective_point_count >= D("5")
    assert surface.entry.driver_p10[DriverName.GROWTH] <= surface.entry.driver_p90[DriverName.GROWTH]
    assert set(surface.driver_revision) == set(DriverName)
    assert any(value != 0 for value in surface.driver_revision.values())
