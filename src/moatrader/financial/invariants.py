from __future__ import annotations

from decimal import Decimal as D

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.apv import ApvAssumptions, ApvCase, ApvEngine
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from moatrader.valuation.biotech_rnpv import (
    BiotechRnpvAssumptions,
    BiotechRnpvEngine,
    PipelineAsset,
)
from moatrader.valuation.economic_dcf import EconomicDcfEngine
from moatrader.valuation.nav import NavAsset, NavAssumptions, NavEngine
from moatrader.valuation.rim import RimAssumptions, RimEngine
from moatrader.valuation.sotp import (
    SotpAssumptions,
    SotpEngine,
    SotpPart,
    SotpValueBasis,
)
from moatrader.valuation.base import ValuationMethod


class InvariantCheck(ContractModel):
    invariant: str
    case_id: str
    lower_input: D
    upper_input: D
    lower_output: D
    upper_output: D
    expected_direction: str
    passed: bool


class InvariantSuiteReport(ContractModel):
    schema_version: str = "valuation-invariants/1"
    checks: list[InvariantCheck] = Field(min_length=100)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    return_data_accessed: bool = False

    @model_validator(mode="after")
    def suite_must_be_green(self) -> "InvariantSuiteReport":
        if self.return_data_accessed:
            raise ValueError("valuation invariants must not access return data")
        if self.passed_count != sum(item.passed for item in self.checks):
            raise ValueError("passed invariant count does not match checks")
        if self.failed_count != len(self.checks) - self.passed_count:
            raise ValueError("failed invariant count does not match checks")
        if self.failed_count:
            failed = [f"{item.invariant}:{item.case_id}" for item in self.checks if not item.passed]
            raise ValueError("economic invariant failure: " + ", ".join(failed[:10]))
        return self


def _economic(*, wacc: D, scale: D = D("1")) -> EconomicDcfAssumptions:
    return EconomicDcfAssumptions(
        base_revenue=D("1000") * scale,
        base_nopat_margin=D("0.10"),
        base_invested_capital=D("800") * scale,
        revenue_growth=D("0.05"),
        target_nopat_margin=D("0.12"),
        roiic=D("0.15"),
        competitive_advantage_period_years=3,
        fade_years=2,
        explicit_forecast_years=5,
        stable_growth=D("0.02"),
        stable_nopat_margin=D("0.10"),
        stable_roic=D("0.10"),
        wacc=wacc,
        diluted_shares=D("10"),
    )


def _rim(*, roe: D, scale: D) -> RimAssumptions:
    return RimAssumptions(
        book_equity=D("1000") * scale,
        roe_path=[roe] * 5,
        cost_of_equity=D("0.10"),
        payout_ratio=D("0.40"),
        terminal_roe=roe,
        terminal_growth=D("0.03"),
        diluted_shares=D("10"),
        provenance=["INVARIANT_SUITE"],
    )


def _rnpv(*, probability: D, scale: D) -> BiotechRnpvAssumptions:
    return BiotechRnpvAssumptions(
        assets=[
            PipelineAsset(
                name="Asset A",
                years_to_launch=3,
                probability_of_approval=probability,
                launch_value=D("1000") * scale,
                remaining_development_costs=[D("20"), D("20"), D("20")],
                evidence_ids=["INVARIANT_SUITE"],
            )
        ],
        discount_rate=D("0.12"),
        net_cash=D("100"),
        diluted_shares=D("10"),
    )


def _apv_case(*, shield: D, scale: D) -> ApvCase:
    return ApvCase(
        unlevered_fcff=[D("100") * scale] * 5,
        terminal_cash_flow=D("100") * scale,
        terminal_growth=D("0.02"),
        unlevered_cost_of_capital=D("0.10"),
        tax_shields=[shield] * 5,
        tax_shield_discount_rate=D("0.06"),
    )


def _record(
    *,
    invariant: str,
    case_id: str,
    lower_input: D,
    upper_input: D,
    lower_output: D,
    upper_output: D,
    direction: str,
) -> InvariantCheck:
    passed = upper_output > lower_output if direction == "INCREASES" else upper_output < lower_output
    return InvariantCheck(
        invariant=invariant,
        case_id=case_id,
        lower_input=lower_input,
        upper_input=upper_input,
        lower_output=lower_output,
        upper_output=upper_output,
        expected_direction=direction,
        passed=passed,
    )


def run_reference_invariant_suite() -> InvariantSuiteReport:
    """Run 120 deterministic, return-free economic direction checks."""

    checks: list[InvariantCheck] = []
    for index in range(1, 21):
        scale = D(index) / D("10") + D("0.5")

        low_wacc = D("0.08")
        high_wacc = D("0.11")
        low_value = EconomicDcfEngine().value(_economic(wacc=low_wacc, scale=scale)).fair_value_per_share
        high_value = EconomicDcfEngine().value(_economic(wacc=high_wacc, scale=scale)).fair_value_per_share
        checks.append(
            _record(
                invariant="FCFF_WACC_UP_FAIR_VALUE_DOWN",
                case_id=str(index),
                lower_input=low_wacc,
                upper_input=high_wacc,
                lower_output=low_value,
                upper_output=high_value,
                direction="DECREASES",
            )
        )

        low_roe = D("0.11")
        high_roe = D("0.16")
        low_value = RimEngine().value(_rim(roe=low_roe, scale=scale)).fair_value_per_share
        high_value = RimEngine().value(_rim(roe=high_roe, scale=scale)).fair_value_per_share
        checks.append(
            _record(
                invariant="RIM_EXCESS_ROE_UP_FAIR_VALUE_UP",
                case_id=str(index),
                lower_input=low_roe,
                upper_input=high_roe,
                lower_output=low_value,
                upper_output=high_value,
                direction="INCREASES",
            )
        )

        low_probability = D("0.25")
        high_probability = D("0.75")
        low_value = BiotechRnpvEngine().value(
            _rnpv(probability=low_probability, scale=scale)
        ).fair_value_per_share
        high_value = BiotechRnpvEngine().value(
            _rnpv(probability=high_probability, scale=scale)
        ).fair_value_per_share
        checks.append(
            _record(
                invariant="RNPV_SUCCESS_PROBABILITY_UP_VALUE_UP",
                case_id=str(index),
                lower_input=low_probability,
                upper_input=high_probability,
                lower_output=low_value,
                upper_output=high_value,
                direction="INCREASES",
            )
        )

        base_asset = D("500") * scale
        asset_increment = D("100")
        def nav_value(asset_value: D) -> D:
            return NavEngine().value(
                NavAssumptions(
                    assets=[
                        NavAsset(
                            name="Asset A",
                            base_value=asset_value,
                            evidence_ids=["INVARIANT_SUITE"],
                        )
                    ],
                    cash=D("100"),
                    debt=D("200"),
                    diluted_shares=D("10"),
                )
            ).fair_value_per_share
        checks.append(
            _record(
                invariant="NAV_ASSET_VALUE_UP_EQUITY_VALUE_UP",
                case_id=str(index),
                lower_input=base_asset,
                upper_input=base_asset + asset_increment,
                lower_output=nav_value(base_asset),
                upper_output=nav_value(base_asset + asset_increment),
                direction="INCREASES",
            )
        )

        part_value = D("600") * scale
        def sotp_value(value: D) -> D:
            part_a = SotpPart(
                name="A",
                method=ValuationMethod.ECONOMIC_FCFF,
                value_basis=SotpValueBasis.ENTERPRISE,
                downside_value=value * D("0.8"),
                base_value=value,
                upside_value=value * D("1.2"),
                ownership_pct=D("0.6"),
                included_cashflows=["A"],
                provenance=["INVARIANT_SUITE"],
            )
            part_b = SotpPart(
                name="B",
                method=ValuationMethod.NAV,
                value_basis=SotpValueBasis.EQUITY,
                downside_value=D("160"),
                base_value=D("200"),
                upside_value=D("240"),
                included_cashflows=["B"],
                provenance=["INVARIANT_SUITE"],
            )
            return SotpEngine().value(
                SotpAssumptions(parts=[part_a, part_b], diluted_shares=D("10"))
            ).fair_value_per_share
        lower_sotp = sotp_value(part_value)
        upper_sotp = sotp_value(part_value + D("100"))
        if upper_sotp - lower_sotp != D("6"):
            upper_sotp = lower_sotp
        checks.append(
            _record(
                invariant="SOTP_PART_PLUS_100_OWNERSHIP_ADJUSTED",
                case_id=str(index),
                lower_input=part_value,
                upper_input=part_value + D("100"),
                lower_output=lower_sotp,
                upper_output=upper_sotp,
                direction="INCREASES",
            )
        )

        low_shield = D("5")
        high_shield = D("15")
        def apv_value(shield: D) -> D:
            case = _apv_case(shield=shield, scale=scale)
            return ApvEngine().value(
                ApvAssumptions(
                    downside=case,
                    base=case,
                    upside=case,
                    debt=D("100"),
                    diluted_shares=D("10"),
                )
            ).fair_value_per_share
        checks.append(
            _record(
                invariant="APV_TAX_SHIELD_UP_VALUE_UP",
                case_id=str(index),
                lower_input=low_shield,
                upper_input=high_shield,
                lower_output=apv_value(low_shield),
                upper_output=apv_value(high_shield),
                direction="INCREASES",
            )
        )

    passed = sum(item.passed for item in checks)
    return InvariantSuiteReport(
        checks=checks,
        passed_count=passed,
        failed_count=len(checks) - passed,
    )
