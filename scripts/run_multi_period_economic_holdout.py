from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_manifest_tickers(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            str(row.get("ticker") or "").strip().zfill(6)
            for row in csv.DictReader(stream)
            if str(row.get("ticker") or "").strip()
        }


def _batch_status(run_dir: Path, expected: set[str]) -> dict[str, Any]:
    result_path = run_dir / "run-result.json"
    if not result_path.is_file():
        return {"complete": False, "statuses": {}, "missing": sorted(expected)}
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"complete": False, "statuses": {}, "missing": sorted(expected)}
    statuses = {
        str(row.get("ticker") or "").zfill(6): str(row.get("status") or "")
        for row in payload.get("companies", [])
    }
    reusable = {"COMPLETE", "NO_PIT_DOCUMENT"}
    missing = sorted(expected - set(statuses))
    failed = sorted(ticker for ticker in expected if statuses.get(ticker) not in reusable)
    return {
        "complete": not missing and not failed,
        "statuses": statuses,
        "missing": missing,
        "failed": failed,
    }


def _command(
    *,
    manifest: Path,
    output: Path,
    run_id: str,
    date: str,
    replay_cache: Path,
    experiment_id: str,
    resume: bool,
    workers: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "moatrader.cli",
        "moat",
        "run",
        "--universe",
        str(manifest),
        "--as-of",
        f"{date}T23:59:59+09:00",
        "--output",
        str(output),
        "--run-id",
        run_id,
        "--experiment-id",
        experiment_id,
        "--llm-replay-cache",
        str(replay_cache),
        "--summary-model",
        "gpt-5-nano",
        "--moat-model",
        "gpt-5.6-luna",
        "--summary-reasoning-effort",
        "low",
        "--atomic-reasoning-effort",
        "medium",
        "--moat-reasoning-effort",
        "medium",
        "--context-tokens",
        "64000",
        "--prompt-reserve-tokens",
        "8000",
        "--strength-context-tokens",
        "100000",
        "--strength-prompt-reserve-tokens",
        "12000",
        "--max-output-tokens",
        "8000",
        "--minimum-text-retention",
        "0.95",
        "--minimum-numeric-retention",
        "0.99",
        "--minimum-structured-fact-retention",
        "0.99",
        "--maximum-price-age-days",
        "7",
        "--maximum-atomic-evidence-units",
        "24",
        "--consolidate-section-summaries",
        "--workers",
        str(workers),
        "--validation-attempts",
        "2",
        "--api-retries",
        "4",
        "--api-timeout",
        "180",
    ]
    if resume:
        command.append("--resume")
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    protocol = json.loads((root / "FROZEN_protocol.json").read_text(encoding="utf-8-sig"))
    experiment_id = str(protocol["experiment_id"])
    dates = list(protocol["primary_holdout_dates"])
    replay_cache = root / "llm-replay"
    log_root = root / "orchestration" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    if not child_env.get("OPENAI_API_KEY") and child_env.get("OPENAPI_KEY"):
        child_env["OPENAI_API_KEY"] = child_env["OPENAPI_KEY"]
    if not child_env.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY or OPENAPI_KEY is required")

    attempts: list[dict[str, Any]] = []
    for pass_number in range(1, args.max_passes + 1):
        pending = 0
        for date in dates:
            output = root / "runs" / date
            manifests = sorted((root / "manifests" / date).glob("batch-*.csv"))
            if not manifests:
                raise RuntimeError(f"no manifests found for {date}")
            for manifest in manifests:
                run_id = manifest.stem
                run_dir = output / run_id
                expected = _read_manifest_tickers(manifest)
                before = _batch_status(run_dir, expected)
                if before["complete"]:
                    continue
                pending += 1
                command = _command(
                    manifest=manifest.resolve(),
                    output=output.resolve(),
                    run_id=run_id,
                    date=date,
                    replay_cache=replay_cache.resolve(),
                    experiment_id=experiment_id,
                    resume=run_dir.exists(),
                    workers=args.workers,
                )
                started_at = datetime.now(timezone.utc).isoformat()
                process = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    env=child_env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                log_path = log_root / f"{date}-{run_id}-pass-{pass_number}.log"
                log_path.write_text(
                    process.stdout + "\n--- STDERR ---\n" + process.stderr,
                    encoding="utf-8",
                )
                after = _batch_status(run_dir, expected)
                attempts.append(
                    {
                        "pass": pass_number,
                        "date": date,
                        "run_id": run_id,
                        "started_at": started_at,
                        "return_code": process.returncode,
                        "complete": after["complete"],
                        "failed": after.get("failed", []),
                        "missing": after.get("missing", []),
                        "log": str(log_path.resolve()),
                    }
                )
                summary = {
                    "schema_version": "moatrader-multi-period-run-orchestration/1",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "experiment_id": experiment_id,
                    "attempts": attempts,
                }
                (root / "orchestration" / "run-summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        if pending == 0:
            break

    batches: list[dict[str, Any]] = []
    for date in dates:
        for manifest in sorted((root / "manifests" / date).glob("batch-*.csv")):
            run_id = manifest.stem
            status = _batch_status(
                root / "runs" / date / run_id,
                _read_manifest_tickers(manifest),
            )
            batches.append({"date": date, "run_id": run_id, **status})
    complete = all(batch["complete"] for batch in batches)
    summary = {
        "schema_version": "moatrader-multi-period-run-orchestration/1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "batch_count": len(batches),
        "complete_batch_count": sum(batch["complete"] for batch in batches),
        "complete": complete,
        "batches": batches,
        "attempts": attempts,
    }
    (root / "orchestration" / "run-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run every frozen primary multi-period batch, resuming only incomplete batches."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-passes", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.max_passes < 1:
        raise ValueError("max_passes must be positive")
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
