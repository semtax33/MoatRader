from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from moatrader.expectations.downstream_validation import (
    AnalystRevisionObservationV1,
    FundamentalValidationObservationV1,
    ReturnNeutralizationObservationV1,
    evaluate_analyst_revision,
    evaluate_future_fundamentals,
    evaluate_return_value_neutralization,
)
from moatrader.expectations.historical_evidence import sha256_file


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_models(path: Path, model: type[Any]) -> list[Any]:
    text = path.read_text(encoding="utf-8-sig")
    raw: Iterable[object] = json.loads(text) if text.lstrip().startswith("[") else (
        json.loads(line) for line in text.splitlines() if line.strip()
    )
    return [model.model_validate(item) for item in raw]


def run(
    *,
    mechanism_stage_status: Path,
    output: Path,
    analyst_input: Path | None = None,
    fundamental_input: Path | None = None,
    return_input: Path | None = None,
    authorize_return_stage: bool = False,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    mechanism = json.loads(mechanism_stage_status.read_text(encoding="utf-8"))
    passed = bool(
        mechanism.get("downstream_stage_authorized")
        or mechanism.get("mechanism_gate_passed")
    )
    output.mkdir(parents=True, exist_ok=True)
    if not passed:
        status = {
            "schema_version": "moatrader-future-eri-downstream-stage-v1/1",
            "status": "BLOCKED_ERI_MECHANISM_GATE",
            "analyst_input_opened": False,
            "fundamental_input_opened": False,
            "return_input_opened": False,
            "per_pbr_role": "NOT_OPENED",
        }
        _write_json(output / "stage-status.json", status)
        return status

    audits: dict[str, str] = {"mechanism_stage_status": sha256_file(mechanism_stage_status)}
    secondary_completed = False
    if analyst_input is not None:
        analyst = _read_models(analyst_input, AnalystRevisionObservationV1)
        _write_json(
            output / "analyst-revision-report.json",
            evaluate_analyst_revision(analyst, mechanism_gate_passed=True),
        )
        audits["analyst_input"] = sha256_file(analyst_input)
        secondary_completed = True
    if fundamental_input is not None:
        fundamentals = _read_models(fundamental_input, FundamentalValidationObservationV1)
        _write_json(
            output / "future-fundamentals-report.json",
            evaluate_future_fundamentals(fundamentals, mechanism_gate_passed=True),
        )
        audits["fundamental_input"] = sha256_file(fundamental_input)
        secondary_completed = True

    if not authorize_return_stage:
        return_status = "NOT_AUTHORIZED_EXPLICIT_RETURN_GATE_CLOSED"
        return_opened = False
    elif not secondary_completed:
        return_status = "BLOCKED_SECONDARY_VALIDATION_NOT_COMPLETED"
        return_opened = False
    elif return_input is None:
        return_status = "BLOCKED_RETURN_INPUT_NOT_SUPPLIED"
        return_opened = False
    else:
        return_rows = _read_models(return_input, ReturnNeutralizationObservationV1)
        _write_json(
            output / "return-value-neutralization-report.json",
            evaluate_return_value_neutralization(return_rows, mechanism_gate_passed=True),
        )
        audits["return_input"] = sha256_file(return_input)
        return_status = "EVALUATED_PREDICTED_F_SCORE_RETURN_AND_VALUE_NEUTRALIZATION"
        return_opened = True
    status = {
        "schema_version": "moatrader-future-eri-downstream-stage-v1/1",
        "status": "MECHANISM_PASSED_DOWNSTREAM_ROUTED",
        "analyst_input_opened": analyst_input is not None,
        "fundamental_input_opened": fundamental_input is not None,
        "secondary_validation_completed": secondary_completed,
        "return_stage_status": return_status,
        "return_input_opened": return_opened,
        "actual_future_eri_used_as_signal": False,
        "signal_rank_policy": "F_SCORE_ONLY_NO_VALUE_PRIMARY_RANKING",
        "primary_value_neutralization_spec": "ALL_VALUE_METRICS_JOINT",
        "per_pbr_primary_ranking": False,
        "per_pbr_role": "COMPARATOR_CONTROL_ONLY" if return_opened else "NOT_OPENED",
        "input_hashes": audits,
    }
    _write_json(output / "stage-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run analyst/fundamental checks and explicitly gated return neutralization."
    )
    parser.add_argument("--mechanism-stage-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyst-input", type=Path)
    parser.add_argument("--fundamental-input", type=Path)
    parser.add_argument("--return-input", type=Path)
    parser.add_argument("--authorize-return-stage", action="store_true")
    args = parser.parse_args()
    result = run(
        mechanism_stage_status=args.mechanism_stage_status,
        output=args.output,
        analyst_input=args.analyst_input,
        fundamental_input=args.fundamental_input,
        return_input=args.return_input,
        authorize_return_stage=args.authorize_return_stage,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
