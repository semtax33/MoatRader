from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "moatrader-multi-period-economic-holdout/1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ticker(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("ticker is blank")
    return text.zfill(6)


def _index(
    rows: list[dict[str, str]], *, ticker_column: str, source: str
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        ticker = _ticker(row.get(ticker_column))
        if ticker in result:
            raise ValueError(f"duplicate ticker in {source}: {ticker}")
        result[ticker] = row
    return result


def _index_manifest(
    rows: list[dict[str, str]], *, source: str
) -> dict[str, dict[str, str]]:
    """Index a filing manifest while retaining its legitimate multi-document rows."""

    result: dict[str, dict[str, str]] = {}
    identity_fields = ("issuer_id", "current_price", "price_as_of")
    for row in rows:
        ticker = _ticker(row.get("ticker"))
        prior = result.get(ticker)
        if prior is None:
            result[ticker] = row
            continue
        inconsistent = [
            field
            for field in identity_fields
            if str(prior.get(field) or "").strip() != str(row.get(field) or "").strip()
        ]
        if inconsistent:
            raise ValueError(
                f"inconsistent duplicate ticker in {source}: {ticker} fields={inconsistent}"
            )
    return result


def broad_sector(industry_code: str) -> str:
    """Map Korean SIC codes to stable broad sectors suitable for a small panel."""

    digits = "".join(character for character in str(industry_code or "") if character.isdigit())
    if len(digits) < 2:
        return "UNKNOWN"
    division = int(digits[:2])
    if digits.startswith("212"):
        return "HEALTHCARE"
    if digits.startswith("28202"):
        return "MATERIALS_ENERGY"
    if digits.startswith(("751", "752")):
        return "CONSUMER_DISCRETIONARY"
    if 1 <= division <= 3:
        return "AGRICULTURE"
    if 5 <= division <= 9 or 19 <= division <= 25:
        return "MATERIALS_ENERGY"
    if 10 <= division <= 12:
        return "CONSUMER_STAPLES"
    if 13 <= division <= 15 or division in {30, 45, 47, 55, 56, 85, 90, 91, 95, 96}:
        return "CONSUMER_DISCRETIONARY"
    if 16 <= division <= 18:
        return "MATERIALS_ENERGY"
    if division == 26 or 58 <= division <= 63:
        return "IT_COMMUNICATION"
    if division in {27, 28, 29, 31, 32, 33, 34, 41, 42, 46, 49, 50, 51, 52, 69, 70, 71, 72, 73, 74, 75}:
        return "INDUSTRIALS"
    if 35 <= division <= 39:
        return "UTILITIES"
    if 64 <= division <= 66:
        return "FINANCIALS"
    if division == 68:
        return "REAL_ESTATE"
    if division == 86:
        return "HEALTHCARE"
    return "OTHER"


def _fetch_dart_sector(corp_code: str, *, api_key: str) -> dict[str, str]:
    query = urllib.parse.urlencode({"crtfc_key": api_key, "corp_code": corp_code})
    request = urllib.request.Request(
        f"https://opendart.fss.or.kr/api/company.json?{query}",
        headers={"User-Agent": "MoatRader multi-period holdout"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "000":
        raise RuntimeError(
            f"DART company lookup failed for {corp_code}: "
            f"{payload.get('status')} {payload.get('message')}"
        )
    industry_code = str(payload.get("induty_code") or "").strip()
    return {
        "industry_code": industry_code,
        "sector": broad_sector(industry_code),
        "dart_corp_name": str(payload.get("corp_name") or ""),
    }


def load_or_fetch_sectors(
    *,
    tickers: list[str],
    corp_codes: dict[str, str],
    sector_cache: Path,
    dart_api_key_env: str,
) -> dict[str, dict[str, str]]:
    existing = (
        _index(_read_csv(sector_cache), ticker_column="ticker", source="sector cache")
        if sector_cache.is_file()
        else {}
    )
    missing = [ticker for ticker in tickers if ticker not in existing]
    if missing:
        api_key = os.getenv(dart_api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{dart_api_key_env} is required to fetch {len(missing)} missing sector rows"
            )
        rows = list(existing.values())
        for position, ticker in enumerate(missing):
            details = _fetch_dart_sector(corp_codes[ticker], api_key=api_key)
            rows.append(
                {
                    "ticker": ticker,
                    "corp_code": corp_codes[ticker],
                    **details,
                    "source": "OPENDART_COMPANY_API",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            if position + 1 < len(missing):
                time.sleep(0.05)
        rows.sort(key=lambda row: _ticker(row["ticker"]))
        _write_csv(
            sector_cache,
            rows,
            (
                "ticker",
                "corp_code",
                "dart_corp_name",
                "industry_code",
                "sector",
                "source",
                "retrieved_at",
            ),
        )
        existing = _index(rows, ticker_column="ticker", source="sector cache")
    unknown = [ticker for ticker in tickers if existing[ticker].get("sector") in {None, "", "UNKNOWN"}]
    if unknown:
        raise RuntimeError(f"sector classification is unavailable for: {unknown}")
    return existing


def trading_horizons(
    trading_dates: list[str], signal_dates: list[str], *, horizon_sessions: int
) -> dict[str, tuple[str, str]]:
    if horizon_sessions < 1:
        raise ValueError("horizon_sessions must be positive")
    ordered = sorted(dict.fromkeys(trading_dates))
    parsed = [datetime.strptime(value, "%Y-%m-%d").date() for value in ordered]
    result: dict[str, tuple[str, str]] = {}
    for signal_date in signal_dates:
        cutoff = datetime.strptime(signal_date, "%Y-%m-%d").date()
        prior = [index for index, value in enumerate(parsed) if value <= cutoff]
        if not prior:
            raise ValueError(f"market calendar has no session on or before {signal_date}")
        signal_index = prior[-1]
        target_index = signal_index + horizon_sessions
        if target_index >= len(ordered):
            raise ValueError(
                f"market calendar lacks {horizon_sessions} sessions after {signal_date}"
            )
        result[signal_date] = (ordered[signal_index], ordered[target_index])
    return result


def _collect_market_calendar(start: str, end: str) -> list[str]:
    from pykrx import stock

    frame = stock.get_market_ohlcv(
        start.replace("-", ""), end.replace("-", ""), "005930"
    )
    if frame.empty:
        raise RuntimeError("KRX market calendar query returned no rows")
    return [value.date().isoformat() for value in frame.index]


def _collect_ticker_closes(ticker: str, start: str, end: str) -> dict[str, float]:
    from pykrx import stock

    frame = stock.get_market_ohlcv(
        start.replace("-", ""), end.replace("-", ""), ticker
    )
    if frame.empty:
        return {}
    return {
        index.date().isoformat(): float(close)
        for index, close in zip(frame.index, frame["종가"], strict=True)
        if float(close) > 0
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"holdout output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    workspace = args.workspace.resolve()
    source_protocol_path = args.source_protocol.resolve()
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8-sig"))
    tickers = [_ticker(value) for value in source_protocol["economic"]["tickers"]]
    if len(tickers) < 20 or len(tickers) > 30:
        raise ValueError(f"expected 20..30 frozen tickers, found {len(tickers)}")
    dates = [
        (row.get("as_of") or "").strip()
        for row in _read_csv(args.dates.resolve())
        if (row.get("as_of") or "").strip()
    ]
    if len(dates) != 4 or len(set(dates)) != 4:
        raise ValueError(f"expected four unique signal dates, found {dates}")

    universe = _index(
        _read_csv(args.universe.resolve()),
        ticker_column="stock_code",
        source="universe",
    )
    old_scores = {
        (_ticker(row.get("stock_code")), (row.get("as_of") or "").strip()): row
        for row in _read_csv(args.old_scores.resolve())
        if row.get("economic_moat_score_100") not in {None, ""}
    }
    manifests = {
        date: _read_csv(workspace / "date-inputs" / date / "universe-manifest.csv")
        for date in dates
    }
    manifests_by_ticker = {
        date: _index_manifest(rows, source=f"manifest {date}")
        for date, rows in manifests.items()
    }
    for ticker in tickers:
        if ticker not in universe:
            raise ValueError(f"frozen ticker is missing from universe: {ticker}")
        if any(ticker not in manifests_by_ticker[date] for date in dates):
            raise ValueError(f"frozen ticker is missing a date manifest: {ticker}")
        if any((ticker, date) not in old_scores for date in dates):
            raise ValueError(f"frozen ticker is missing an old score date: {ticker}")

    corp_codes = {
        ticker: str(manifests_by_ticker[dates[0]][ticker].get("issuer_id") or "").strip()
        for ticker in tickers
    }
    if any(not value for value in corp_codes.values()):
        raise ValueError("one or more frozen tickers lack a DART corp code")
    sector_cache = output / "sector-classification.csv"
    sectors = load_or_fetch_sectors(
        tickers=tickers,
        corp_codes=corp_codes,
        sector_cache=sector_cache,
        dart_api_key_env=args.dart_api_key_env,
    )

    calendar_start = (
        datetime.strptime(min(dates), "%Y-%m-%d") - timedelta(days=20)
    ).date().isoformat()
    calendar = _collect_market_calendar(calendar_start, args.price_end)
    horizons = trading_horizons(
        calendar, dates, horizon_sessions=args.horizon_sessions
    )
    closes = {
        ticker: _collect_ticker_closes(ticker, calendar_start, args.price_end)
        for ticker in tickers
    }

    panel: list[dict[str, Any]] = []
    price_mismatches: list[dict[str, Any]] = []
    for date_index, signal_date in enumerate(dates):
        signal_session, return_session = horizons[signal_date]
        split = "SEEN_ANCHOR" if date_index == 0 else "PRIMARY_HOLDOUT"
        for ticker in tickers:
            source_row = manifests_by_ticker[signal_date][ticker]
            source_price = float(source_row["current_price"])
            krx_signal_price = closes[ticker].get(signal_session)
            return_price = closes[ticker].get(return_session)
            return_eligible = bool(
                krx_signal_price is not None
                and return_price is not None
                and source_price > 0
            )
            if krx_signal_price is not None and source_price != krx_signal_price:
                price_mismatches.append(
                    {
                        "date": signal_date,
                        "ticker": ticker,
                        "manifest_price": source_price,
                        "krx_price": krx_signal_price,
                    }
                )
            panel.append(
                {
                    "date": signal_date,
                    "split": split,
                    "ticker": ticker,
                    "company_name": universe[ticker].get("name") or "",
                    "market": universe[ticker].get("market") or "",
                    "size_bucket": universe[ticker].get("size_bucket") or "",
                    "industry_code": sectors[ticker].get("industry_code") or "",
                    "sector": sectors[ticker].get("sector") or "",
                    "old_holistic_score": float(
                        old_scores[(ticker, signal_date)]["economic_moat_score_100"]
                    ),
                    "signal_session": signal_session,
                    "return_session": return_session,
                    "signal_price": source_price,
                    "krx_signal_price": krx_signal_price,
                    "return_price": return_price,
                    "forward_return": (
                        return_price / source_price - 1 if return_eligible else None
                    ),
                    "return_eligible": return_eligible,
                }
            )

    if price_mismatches:
        raise RuntimeError(
            f"manifest/KRX signal price mismatches: {price_mismatches[:5]}"
        )
    eligible_by_date = {
        date: sum(row["return_eligible"] for row in panel if row["date"] == date)
        for date in dates
    }
    if any(count < 20 for count in eligible_by_date.values()):
        raise RuntimeError(f"fewer than 20 realized returns in a date: {eligible_by_date}")

    panel_fields = (
        "date",
        "split",
        "ticker",
        "company_name",
        "market",
        "size_bucket",
        "industry_code",
        "sector",
        "old_holistic_score",
        "signal_session",
        "return_session",
        "signal_price",
        "krx_signal_price",
        "return_price",
        "forward_return",
        "return_eligible",
    )
    _write_csv(output / "FROZEN_panel.csv", panel, panel_fields)

    sample_rows = [
        {
            "ticker": ticker,
            "company_name": universe[ticker].get("name") or "",
            "market": universe[ticker].get("market") or "",
            "size_bucket": universe[ticker].get("size_bucket") or "",
            "industry_code": sectors[ticker].get("industry_code") or "",
            "sector": sectors[ticker].get("sector") or "",
        }
        for ticker in tickers
    ]
    _write_csv(
        output / "FROZEN_sample.csv",
        sample_rows,
        ("ticker", "company_name", "market", "size_bucket", "industry_code", "sector"),
    )

    manifest_hashes: dict[str, str] = {}
    primary_dates = dates[1:]
    for date in primary_dates:
        source_rows = manifests[date]
        fields = list(source_rows[0])
        for offset in range(0, len(tickers), args.batch_size):
            batch = set(tickers[offset : offset + args.batch_size])
            rows = [row for row in source_rows if _ticker(row.get("ticker")) in batch]
            if {_ticker(row.get("ticker")) for row in rows} != batch:
                raise RuntimeError(f"manifest batch is incomplete for {date}: {sorted(batch)}")
            path = output / "manifests" / date / f"batch-{offset // args.batch_size + 1}.csv"
            _write_csv(path, rows, fields)
            manifest_hashes[path.relative_to(output).as_posix()] = _sha256(path)

    experiment_id = "moat-multi-period-economic-20260817-v1"
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "selection": (
            "the pre-existing 24-name return-blind economic cohort is frozen across all dates; "
            "date 1 is a seen anchor and dates 2..4 are the primary temporal holdout"
        ),
        "candidate_selection_used_forward_returns": False,
        "sample_count": len(tickers),
        "dates": dates,
        "seen_anchor_dates": dates[:1],
        "primary_holdout_dates": primary_dates,
        "stock_date_count": len(panel),
        "primary_stock_date_count": len(tickers) * len(primary_dates),
        "return_definition": {
            "kind": "FIXED_MARKET_SESSIONS_CLOSE_TO_CLOSE",
            "horizon_sessions": args.horizon_sessions,
            "price_end": args.price_end,
            "dates": {
                date: {
                    "signal_session": horizons[date][0],
                    "return_session": horizons[date][1],
                }
                for date in dates
            },
        },
        "realized_return_count_by_date": eligible_by_date,
        "evaluation": {
            "minimum_eligible_per_date": 20,
            "primary_metrics": [
                "date_spearman_rank_ic",
                "sector_neutral_rank_ic",
                "equal_count_q5_minus_q1_seeded_tie_monte_carlo",
            ],
            "secondary_metrics": [
                "market_size_neutral_rank_ic",
                "pooled_within_date_rank_ic",
                "tie_concentration",
            ],
            "signals": [
                "OLD_HOLISTIC",
                "CURRENT_PUBLIC",
                "CURRENT_STABLE_RANK_KEY",
            ],
            "winsorization": {"lower": 0.01, "upper": 0.99},
            "quantiles": 5,
            "tie_simulations": 10000,
            "seed": "multi-period-economic-20260817-v1",
        },
        "anchor_current_run_root": str(args.anchor_current_run_root.resolve()),
        "inputs": {
            "source_protocol": str(source_protocol_path),
            "source_protocol_sha256": _sha256(source_protocol_path),
            "workspace_manifest_sha256": _sha256(workspace / "workspace-manifest.json"),
            "universe": str(args.universe.resolve()),
            "universe_sha256": _sha256(args.universe.resolve()),
            "dates": str(args.dates.resolve()),
            "dates_sha256": _sha256(args.dates.resolve()),
            "old_scores": str(args.old_scores.resolve()),
            "old_scores_sha256": _sha256(args.old_scores.resolve()),
            "sector_classification_sha256": _sha256(sector_cache),
            "manifest_sha256": manifest_hashes,
        },
    }
    (output / "FROZEN_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "workspace-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": protocol["frozen_at"],
                "experiment_id": experiment_id,
                "fresh_run": True,
                "source_result_reuse": False,
                "universe_count": len(tickers),
                "dates": dates,
                "expected_signal_count": len(tickers) * len(primary_dates),
                "preflight_required": False,
                "preflight_status": "NOT_REQUIRED",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a four-date, fixed-horizon multi-period MOAT economic holdout."
    )
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--source-protocol", required=True, type=Path)
    parser.add_argument("--anchor-current-run-root", required=True, type=Path)
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--dates", required=True, type=Path)
    parser.add_argument("--old-scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon-sessions", type=int, default=50)
    parser.add_argument("--price-end", default="2026-08-14")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dart-api-key-env", default="DART_API_KEY")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 5:
        raise ValueError("batch_size must be 1..5 so no large-manifest preflight is required")
    protocol = prepare(args)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
