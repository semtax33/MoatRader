from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from moatrader.universe.live_kr import build_live_kr_universe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a new immutable all-Korean-security research workspace."
    )
    parser.add_argument("--marcap", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--sample-ticker", action="append", required=True)
    parser.add_argument("--require-exact-market-date", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    universe_path = output / "inputs" / "universe.csv"
    workspace_path = output / "workspace-manifest.json"
    if universe_path.exists() or workspace_path.exists():
        raise FileExistsError("workspace universe or manifest already exists")

    source_path = args.marcap.resolve()
    build = build_live_kr_universe(pd.read_parquet(source_path), as_of=args.as_of)
    if args.require_exact_market_date and build.source_as_of != args.as_of:
        raise ValueError(
            f"exact market date required: requested={args.as_of}, "
            f"source={build.source_as_of}"
        )

    sample = sorted({str(value).strip().upper().zfill(6) for value in args.sample_ticker})
    if not 3 <= len(sample) <= 5:
        raise ValueError("preflight sample must contain 3 to 5 unique tickers")
    tickers = set(build.universe["stock_code"])
    missing_sample = sorted(set(sample) - tickers)
    if missing_sample:
        raise ValueError(f"preflight sample is outside the live universe: {missing_sample}")

    universe_path.parent.mkdir(parents=True, exist_ok=True)
    build.universe.to_csv(universe_path, index=False, encoding="utf-8-sig")
    (output / "inputs" / "stock-codes.txt").write_text(
        "\n".join(build.universe["stock_code"]) + "\n",
        encoding="utf-8",
    )
    (output / "inputs" / "preflight-sample.txt").write_text(
        "\n".join(sample) + "\n",
        encoding="utf-8",
    )
    build.universe[build.universe["stock_code"].isin(sample)].to_csv(
        output / "inputs" / "preflight-sample.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dates_path = output / "inputs" / "dates.csv"
    pd.DataFrame([{"date": args.as_of.isoformat()}]).to_csv(
        dates_path,
        index=False,
        encoding="utf-8-sig",
    )

    counts_by_market = {
        str(key): int(value)
        for key, value in build.universe.groupby("market").size().items()
    }
    counts_by_type = {
        str(key): int(value)
        for key, value in build.universe.groupby("security_type").size().items()
    }
    source_manifest = {
        "schema_version": "moatrader-live-kr-universe-source/1",
        "requested_as_of": args.as_of.isoformat(),
        "market_source_as_of": build.source_as_of.isoformat(),
        "price_as_of": build.price_as_of.isoformat(),
        "market_source_age_days": (args.as_of - build.source_as_of).days,
        "provider": "FinanceData/marcap",
        "provider_commit": args.source_commit,
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "universe_count": len(build.universe),
        "counts_by_market": counts_by_market,
        "counts_by_security_type": counts_by_type,
        "finance_hint_count": int(build.universe["finance_hint"].sum()),
        "holding_hint_count": int(build.universe["holding_hint"].sum()),
    }
    source_manifest_path = output / "inputs" / "universe-source-manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    experiment_hash = hashlib.sha256(
        f"{_sha256(universe_path)}:{_sha256(dates_path)}".encode("utf-8")
    ).hexdigest()[:12]
    workspace = {
        "schema_version": "moatrader-kr-all-research/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": f"{output.name}-{experiment_hash}",
        "fresh_run": True,
        "source_result_reuse": False,
        "universe_scope": "ALL_KOSPI_KOSDAQ_KONEX_LISTED_SECURITIES",
        "universe_count": len(build.universe),
        "dates": [args.as_of.isoformat()],
        "expected_report_count": len(build.universe),
        "preflight_required": len(build.universe) > 5,
        "preflight_status": "PENDING",
        "preflight_sample_tickers": sample,
        "preflight_sample_size": len(sample),
        "input_sha256": {
            "universe.csv": _sha256(universe_path),
            "dates.csv": _sha256(dates_path),
            "universe-source-manifest.json": _sha256(source_manifest_path),
        },
    }
    workspace_path.write_text(
        json.dumps(workspace, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"workspace={output}")
    print(f"universe={len(build.universe)}")
    print(f"requested_as_of={build.requested_as_of}")
    print(f"market_source_as_of={build.source_as_of}")
    print(f"preflight_sample={','.join(sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
