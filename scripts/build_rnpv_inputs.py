from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from moatrader.valuation import (
    CommonRnpvEngine,
    RnpvBuildInput,
    RnpvBuilder,
    RoutedValuationInput,
    ValuationMethod,
)

try:
    from scripts.build_normalized_fcff_inputs import decimal, file_sha256, read_json
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from build_normalized_fcff_inputs import (  # type: ignore[no-redef]
        decimal,
        file_sha256,
        read_json,
    )


BUILD_REPORT_VERSION = "rnpv-input-build/1"


def build_rnpv_input(
    *,
    entry: dict[str, Any],
    base_root: Path,
) -> RoutedValuationInput:
    as_of = str(entry["as_of"])
    ticker = str(entry["ticker"]).zfill(6)
    dcf_path = base_root / "date-inputs" / as_of / "dcf-inputs" / f"{ticker}.json"
    snapshot_path = (
        base_root
        / "runs"
        / f"kr-signal-{as_of}"
        / "companies"
        / ticker
        / "financial-snapshot.json"
    )
    dcf_input = read_json(dcf_path)
    snapshot = read_json(snapshot_path)
    metrics = dcf_input.get("metrics") or {}
    legacy = dcf_input.get("assumptions") or {}
    cash = decimal(metrics.get("cash"), field="cash")
    debt = decimal(metrics.get("debt"), field="debt")
    source = RnpvBuildInput(
        issuer_id=str(snapshot.get("issuer_id") or ticker),
        as_of=as_of,
        assets=entry["assets"],
        net_cash=cash - debt,
        diluted_shares=decimal(
            legacy.get("diluted_shares"), field="diluted_shares"
        ),
        evidence_available_at=entry["evidence_available_at"],
        provenance=[
            f"PIT_DCF_INPUT:SHA256:{file_sha256(dcf_path)}",
            f"PIT_FINANCIAL_SNAPSHOT:SHA256:{file_sha256(snapshot_path)}",
            f"EVIDENCE_MANIFEST_ENTRY:{ticker}:{as_of}",
        ],
    )
    assumptions = RnpvBuilder().build(source)
    CommonRnpvEngine().value(assumptions)
    source_refs = list(
        dict.fromkeys(
            source.provenance
            + list(source.evidence_available_at)
            + ["POLICY:rnpv-policy/1"]
        )
    )
    return RoutedValuationInput(
        issuer_id=source.issuer_id,
        as_of=as_of,
        method=ValuationMethod.RNPV,
        assumptions=assumptions.model_dump(mode="json"),
        source_refs=source_refs,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output must be new and empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.evidence)
    entries = manifest.get("entries") or []
    generated: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for entry in entries:
        as_of = str(entry.get("as_of") or "")
        ticker = str(entry.get("ticker") or "").zfill(6)
        try:
            envelope = build_rnpv_input(entry=entry, base_root=args.base_root)
            output_path = args.output / as_of / f"{ticker}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    envelope.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            generated.append({"date": as_of, "ticker": ticker})
        except Exception as exc:
            skipped.append(
                {
                    "date": as_of,
                    "ticker": ticker,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    report: dict[str, Any] = {
        "schema_version": BUILD_REPORT_VERSION,
        "evidence_manifest_sha256": file_sha256(args.evidence),
        "base_root": str(args.base_root),
        "llm_call_count": 0,
        "evidence_entry_count": len(entries),
        "generated_count": len(generated),
        "skipped_count": len(skipped),
        "generated": generated,
        "skipped": skipped,
    }
    (args.output / "_build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build PIT rNPV inputs from role-separated asset evidence."
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("evidence", "base_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["skipped_count"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
