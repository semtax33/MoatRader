from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel, SCHEMA_VERSION


class RunManifest(ContractModel):
    run_id: str
    signal_at: datetime
    evidence_cutoff: datetime
    model: str
    parser_version: str
    ast_schema_version: str = SCHEMA_VERSION
    renderer_version: str
    prompt_version: str
    token_budget: int = Field(gt=0)
    input_tokens: int = Field(ge=0)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    temperature: float = Field(ge=0.0, le=2.0)
    created_at: datetime

    @model_validator(mode="after")
    def point_in_time_order(self) -> "RunManifest":
        timestamps = (self.signal_at, self.evidence_cutoff, self.created_at)
        if any(value.tzinfo is None or value.utcoffset() is None for value in timestamps):
            raise ValueError("manifest timestamps must be timezone-aware")
        if self.evidence_cutoff > self.signal_at:
            raise ValueError("evidence cutoff must not be after the signal")
        if self.input_tokens > self.token_budget:
            raise ValueError("input tokens exceed declared token budget")
        return self

