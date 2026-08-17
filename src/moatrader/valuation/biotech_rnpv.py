from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class PipelineAsset(ContractModel):
    name: str = Field(min_length=1)
    years_to_launch: int = Field(ge=0, le=30)
    probability_of_approval: Decimal = Field(ge=0, le=1)
    launch_value: Decimal = Field(ge=0)
    remaining_development_costs: list[Decimal] = Field(default_factory=list, max_length=30)
    evidence_ids: list[str] = Field(min_length=1)


class BiotechRnpvAssumptions(ContractModel):
    assets: list[PipelineAsset] = Field(min_length=1)
    discount_rate: Decimal = Field(gt=0, lt=1)
    net_cash: Decimal = Decimal(0)
    diluted_shares: Decimal = Field(gt=0)


class PipelineAssetValue(ContractModel):
    name: str
    probability_adjusted_launch_value: Decimal
    present_value_of_costs: Decimal
    rnpv: Decimal


class BiotechRnpvValuation(ContractModel):
    assets: list[PipelineAssetValue]
    equity_value: Decimal
    fair_value_per_share: Decimal


class BiotechRnpvEngine:
    def value(self, assumptions: BiotechRnpvAssumptions) -> BiotechRnpvValuation:
        values: list[PipelineAssetValue] = []
        for asset in assumptions.assets:
            launch_pv = (
                asset.launch_value
                * asset.probability_of_approval
                / ((Decimal(1) + assumptions.discount_rate) ** asset.years_to_launch)
            )
            cost_pv = sum(
                (
                    cost
                    / ((Decimal(1) + assumptions.discount_rate) ** year)
                    for year, cost in enumerate(asset.remaining_development_costs, start=1)
                ),
                Decimal(0),
            )
            values.append(
                PipelineAssetValue(
                    name=asset.name,
                    probability_adjusted_launch_value=launch_pv,
                    present_value_of_costs=cost_pv,
                    rnpv=launch_pv - cost_pv,
                )
            )
        equity = sum((item.rnpv for item in values), Decimal(0)) + assumptions.net_cash
        return BiotechRnpvValuation(
            assets=values,
            equity_value=equity,
            fair_value_per_share=equity / assumptions.diluted_shares,
        )
