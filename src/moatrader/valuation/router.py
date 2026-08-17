from __future__ import annotations

from pydantic import Field

from moatrader.canonical.models import ContractModel
from moatrader.valuation.base import (
    ApplicabilityStatus,
    ModelApplicability,
    ValuationMethod,
)
from moatrader.valuation.profile import EconomicArchetype, ValuationProfile


ROUTER_CONTRACT_VERSION = "valuation-router/1"


REQUIRED_DATA: dict[ValuationMethod, tuple[str, ...]] = {
    ValuationMethod.ECONOMIC_FCFF: (
        "revenue",
        "ebit",
        "invested_capital",
        "valuation_assumptions",
        "diluted_shares",
    ),
    ValuationMethod.NORMALIZED_FCFF: (
        "revenue_history",
        "margin_history",
        "invested_capital",
        "valuation_assumptions",
        "diluted_shares",
    ),
    ValuationMethod.RIM: ("book_equity", "net_income", "diluted_shares"),
    ValuationMethod.RNPV: (
        "pipeline_assets",
        "clinical_phase",
        "reference_pos",
        "launch_value",
        "development_costs",
        "diluted_shares",
    ),
    ValuationMethod.SCENARIO_DCF: (
        "revenue",
        "scenario_assumptions",
        "valuation_assumptions",
        "diluted_shares",
    ),
    ValuationMethod.APV: (
        "unlevered_cashflows",
        "debt_schedule",
        "tax_shields",
        "diluted_shares",
    ),
    ValuationMethod.NAV: ("asset_values", "debt", "diluted_shares"),
    ValuationMethod.SOTP: (
        "segment_values",
        "cashflow_scopes",
        "ownership_pct",
        "diluted_shares",
    ),
}


class ValuationRoute(ContractModel):
    contract_version: str = ROUTER_CONTRACT_VERSION
    issuer_id: str
    as_of: str
    economic_archetype: EconomicArchetype
    primary_method: ValuationMethod
    secondary_method: ValuationMethod | None = None
    applicability: ModelApplicability
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rationale: list[str] = Field(min_length=1)


class ValuationProfileRouter:
    """Selects a method from economic structure before price or valuation exists."""

    def route(self, profile: ValuationProfile) -> ValuationRoute:
        method, secondary, rationale = self._select(profile)
        required = list(REQUIRED_DATA[method])
        available = set(profile.available_data)
        missing = sorted(set(required) - available)
        applicability = ModelApplicability(
            method=method,
            status=(
                ApplicabilityStatus.INSUFFICIENT_DATA
                if missing
                else ApplicabilityStatus.ELIGIBLE
            ),
            required_fields=required,
            missing_fields=missing,
            reason_codes=([f"INSUFFICIENT_FOR_{method.value}"] if missing else []),
        )
        return ValuationRoute(
            issuer_id=profile.issuer_id,
            as_of=profile.as_of.isoformat(),
            economic_archetype=profile.economic_archetype,
            primary_method=method,
            secondary_method=secondary,
            applicability=applicability,
            profile_sha256=profile.fingerprint(),
            rationale=rationale,
        )

    @staticmethod
    def _select(
        profile: ValuationProfile,
    ) -> tuple[ValuationMethod, ValuationMethod | None, list[str]]:
        if profile.is_financial_intermediary:
            return (
                ValuationMethod.RIM,
                ValuationMethod.PB_ROE_CROSS_CHECK,
                ["Deposits/debt are operating inputs; value equity residual income."],
            )
        if profile.multi_segment and profile.segment_heterogeneity_material:
            return (
                ValuationMethod.SOTP,
                ValuationMethod.REVERSE_DCF,
                ["Materially heterogeneous segments require separately scoped valuations."],
            )
        if profile.pipeline_assets_material:
            return (
                ValuationMethod.RNPV,
                ValuationMethod.SOTP,
                ["Material discrete pipeline outcomes require probability- and timing-adjusted rNPV."],
            )
        if profile.is_reit or profile.is_resource_company or profile.asset_value_primary:
            return (
                ValuationMethod.NAV,
                ValuationMethod.NORMALIZED_EARNINGS,
                ["Separable physical assets, not a perpetual operating cash flow, drive value."],
            )
        if profile.leverage_path_material:
            return (
                ValuationMethod.APV,
                ValuationMethod.ECONOMIC_FCFF,
                ["A material changing debt path requires financing effects to be valued separately."],
            )
        if profile.economic_archetype == EconomicArchetype.LOSS_MAKING_GROWTH or profile.ebit_positive is False:
            return (
                ValuationMethod.SCENARIO_DCF,
                ValuationMethod.REVERSE_DCF,
                ["Commercial but loss-making economics require explicit mature-state scenarios."],
            )
        if profile.materially_cyclical:
            return (
                ValuationMethod.NORMALIZED_FCFF,
                ValuationMethod.NORMALIZED_EARNINGS,
                ["Cyclical cash flows require PIT mid-cycle normalization."],
            )
        return (
            ValuationMethod.ECONOMIC_FCFF,
            ValuationMethod.REVERSE_DCF,
            ["General operating economics support growth-reinvestment-ROIIC FCFF."],
        )
