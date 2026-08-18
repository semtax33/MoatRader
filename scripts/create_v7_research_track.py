from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from moatrader.experiments.coverage import build_coverage_report, write_coverage_artifacts
from moatrader.experiments.integrity import (
    audit_v6_integrity,
    ensure_new_experiment_output,
    sha256_file,
    snapshot_protected_files,
)
from moatrader.experiments.shadow import ExpectationGapResearchContract
from moatrader.financial.invariants import run_reference_invariant_suite


def _aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return parsed


def _write_json(path: Path, value: object) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a new return-blind v7 research track without modifying v6."
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--v6-contract", type=Path, required=True)
    parser.add_argument("--v6-stability-a", type=Path, required=True)
    parser.add_argument("--v6-stability-b", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at", type=_aware_datetime, required=True)
    parser.add_argument("--first-signal-at", type=_aware_datetime, required=True)
    parser.add_argument("--weeks", type=int, default=12)
    parser.add_argument("--expected-universe-count", type=int, default=150)
    args = parser.parse_args()
    if args.weeks < 1:
        raise ValueError("weeks must be positive")

    root = args.repository_root.resolve()
    contract_path = args.v6_contract.resolve()
    stability_a = args.v6_stability_a.resolve()
    stability_b = args.v6_stability_b.resolve()
    protected = [contract_path.parent, stability_a, stability_b]
    output = ensure_new_experiment_output(
        args.output,
        required_label="v7",
        protected_roots=protected,
    )

    before = snapshot_protected_files(
        repository_root=root,
        contract_path=contract_path,
        stability_directories=[stability_a, stability_b],
    )
    integrity = audit_v6_integrity(
        repository_root=root,
        contract_path=contract_path,
        stability_directory=stability_a,
        stability_directories_for_snapshot=[stability_a, stability_b],
    )
    coverage = build_coverage_report(
        routing_path=args.routing.resolve(),
        signals_path=args.signals.resolve(),
    )
    if coverage.row_count != args.expected_universe_count * 4:
        raise ValueError(
            "v7 baseline expects four development dates across the configured universe"
        )
    invariant_report = run_reference_invariant_suite()
    schedule = [args.first_signal_at + timedelta(days=7 * index) for index in range(args.weeks)]
    contract = ExpectationGapResearchContract.create(
        created_at=args.created_at,
        parent_v6_contract_file_sha256=sha256_file(contract_path),
        parent_v6_contract_payload_sha256=integrity.contract_payload_sha256,
        parent_engineering_input_sha256={
            "routing.csv": sha256_file(args.routing.resolve()),
            "signals.csv": sha256_file(args.signals.resolve()),
        },
        expected_universe_count=args.expected_universe_count,
        scheduled_signal_at=schedule,
        cadence="WEEKLY",
        research_horizons_calendar_days=[21, 42, 77],
        primary_horizon_calendar_days=77,
        overlapping_horizons_are_not_independent=True,
        v6_results_must_not_modify_v7=True,
        v7_results_must_only_modify_v8=True,
        analyst_market_opinion_intrinsic_access=False,
        return_inputs_forbidden_before_signal_seal=True,
        return_data_accessed=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        staged.mkdir()
        write_coverage_artifacts(coverage, output_directory=staged / "coverage")
        _write_json(staged / "research-contract.json", contract)
        _write_json(staged / "v6-integrity-at-fork.json", integrity)
        _write_json(staged / "valuation-invariant-results.json", invariant_report)
        with (staged / "shadow-schedule.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["signal_at", "horizon_21_end", "horizon_42_end", "horizon_77_end"],
            )
            writer.writeheader()
            for signal_at in schedule:
                writer.writerow(
                    {
                        "signal_at": signal_at.isoformat(),
                        "horizon_21_end": (signal_at + timedelta(days=21)).isoformat(),
                        "horizon_42_end": (signal_at + timedelta(days=42)).isoformat(),
                        "horizon_77_end": (signal_at + timedelta(days=77)).isoformat(),
                    }
                )

        after = snapshot_protected_files(
            repository_root=root,
            contract_path=contract_path,
            stability_directories=[stability_a, stability_b],
        )
        if before != after:
            changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
            raise RuntimeError("v6 changed during v7 creation: " + ", ".join(changed))
        artifact_hashes = {
            path.relative_to(staged).as_posix(): sha256_file(path)
            for path in sorted(staged.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": "v7-research-build-manifest/1",
            "v6_unchanged_during_build": True,
            "v6_pristine_against_frozen_contract": integrity.pristine_against_contract,
            "artifact_sha256": artifact_hashes,
            "return_data_accessed": False,
        }
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        manifest["manifest_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        _write_json(staged / "build-manifest.json", manifest)
        staged.rename(output)

    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
