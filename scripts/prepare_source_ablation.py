from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from moatrader.adapters import (
    PaddlePdfOcrAdapter,
    RawDocument,
    enrich_ir_table_semantics,
)
from moatrader.adapters.pdf import PARSER_VERSION as IR_PDF_PARSER_VERSION
from moatrader.ingestion import (
    BronzeFilingStore,
    KindCompanyIdentity,
    KindIrClient,
    KindIrCollector,
    ResilientHttpClient,
    normalize_company_name,
)
from moatrader.pipeline import CanonicalFinancialDocumentPipeline
from moatrader.quality import assess_parser_quality


SCHEMA_VERSION = "moatrader-source-ablation/2"
MANIFEST_FIELDS = (
    "ticker",
    "source",
    "input",
    "metadata",
    "issuer_id",
    "issuer_name",
    "current_price",
    "price_as_of",
    "dcf_assumptions",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ticker(value: object) -> str:
    raw = str(value or "").strip()
    return raw.zfill(6) if raw else ""


def _identity_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        ticker = _ticker(row.get("ticker"))
        if not ticker:
            raise ValueError("base manifest contains a blank ticker")
        prior = result.get(ticker)
        if prior is None:
            result[ticker] = row
            continue
        for field in ("issuer_id", "issuer_name", "current_price", "price_as_of"):
            if str(prior.get(field) or "").strip() != str(row.get(field) or "").strip():
                raise ValueError(f"ticker {ticker}: inconsistent {field} in base manifest")
    return result


def _selection_key(seed: str, ticker: str) -> str:
    return hashlib.sha256(f"{seed}:{ticker}".encode("utf-8")).hexdigest()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    base_manifest = Path(args.base_manifest).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.materials_per_company_per_year <= 0:
        raise ValueError("materials_per_company_per_year must be positive")
    if args.max_years_per_company <= 0:
        raise ValueError("max_years_per_company must be positive")
    if args.minimum_years_per_company <= 0:
        raise ValueError("minimum_years_per_company must be positive")
    if args.minimum_years_per_company > args.max_years_per_company:
        raise ValueError(
            "minimum_years_per_company must not exceed max_years_per_company"
        )
    base_rows = _read_csv(base_manifest)
    identity_rows = _identity_rows(base_rows)
    identities = {
        ticker: KindCompanyIdentity(
            ticker=ticker,
            issuer_id=str(row.get("issuer_id") or "").strip(),
            issuer_name=str(row.get("issuer_name") or "").strip(),
        )
        for ticker, row in identity_rows.items()
    }
    if any(not item.issuer_id or not item.issuer_name for item in identities.values()):
        raise ValueError("base manifest must provide issuer_id and issuer_name for every ticker")

    http = ResilientHttpClient(
        user_agent="MoatRader source-ablation preparation",
        requests_per_second=args.requests_per_second,
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        default_max_bytes=int(args.max_download_mb * 1024 * 1024),
    )
    client = KindIrClient(http)
    materials = client.search_materials(
        begin_date=args.begin_date,
        end_date=args.end_date,
        page_size=3000,
    )
    identity_by_name = {
        normalize_company_name(identity.issuer_name): identity
        for identity in identities.values()
    }
    by_ticker: dict[str, list[Any]] = defaultdict(list)
    for material in materials:
        identity = identity_by_name.get(normalize_company_name(material.company_name))
        if identity is not None:
            by_ticker[identity.ticker].append(material)
    eligible = sorted(
        (
            ticker
            for ticker, ticker_materials in by_ticker.items()
            if len({material.listed_on.year for material in ticker_materials})
            >= args.minimum_years_per_company
        ),
        key=lambda ticker: (_selection_key(args.seed, ticker), ticker),
    )
    if args.fixed_cohort:
        fixed_rows = _read_csv(Path(args.fixed_cohort).resolve())
        fixed_selected = [_ticker(row.get("ticker")) for row in fixed_rows]
        if len(fixed_selected) != len(set(fixed_selected)) or any(
            not ticker for ticker in fixed_selected
        ):
            raise ValueError("fixed cohort must contain unique nonblank tickers")
        missing = sorted(set(fixed_selected) - set(eligible))
        if missing and not args.filter_fixed_cohort_by_ir_years:
            raise ValueError(f"fixed cohort lacks PIT-available KIND materials: {missing}")
        selected = (
            [ticker for ticker in fixed_selected if ticker in set(eligible)]
            if args.filter_fixed_cohort_by_ir_years
            else fixed_selected
        )
        if len(selected) != args.sample_size:
            raise ValueError(
                f"fixed cohort selection has {len(selected)} companies after the "
                f"availability rule; sample_size={args.sample_size}"
            )
    elif len(eligible) < args.sample_size:
        raise ValueError(
            f"only {len(eligible)} companies have a matching PIT-available KIND PDF; "
            f"sample_size={args.sample_size}"
        )
    else:
        selected = eligible[: args.sample_size]
    selected_identities = [identities[ticker] for ticker in selected]

    collector = KindIrCollector(
        client,
        BronzeFilingStore(output / "bronze"),
        max_download_bytes=int(args.max_download_mb * 1024 * 1024),
    )
    collection = collector.collect(
        begin_date=args.begin_date,
        end_date=args.end_date,
        companies=selected_identities,
        refresh=args.refresh,
        max_materials_per_company=(
            1 if args.ir_selection_mode == "latest" else None
        ),
        max_materials_per_company_per_year=(
            args.materials_per_company_per_year
            if args.ir_selection_mode == "annual-snapshots"
            else None
        ),
        max_years_per_company=(
            args.max_years_per_company
            if args.ir_selection_mode == "annual-snapshots"
            else None
        ),
    )
    _write_json(output / "collection-result.json", collection.model_dump(mode="json"))

    filings_by_ticker: dict[str, list[Any]] = defaultdict(list)
    for filing in collection.filings:
        filings_by_ticker[filing.ticker].append(filing)
    material_by_document_id = {
        material.source_document_id: material for material in materials
    }
    parse_rows: list[dict[str, Any]] = []
    prepared_metadata_by_document_id: dict[str, str] = {}
    ocr_adapter = (
        PaddlePdfOcrAdapter(
            device=args.ir_ocr_device,
            cpu_threads=args.ir_ocr_cpu_threads,
        )
        if args.ir_ocr_engine == "paddle"
        else None
    )
    pipeline = CanonicalFinancialDocumentPipeline(ir_ocr_adapter=ocr_adapter)
    for ticker in selected:
        filings = filings_by_ticker.get(ticker, [])
        if not filings:
            parse_rows.append(
                {
                    "ticker": ticker,
                    "issuer_name": identities[ticker].issuer_name,
                    "status": "MISSING_DOWNLOAD",
                }
            )
            continue
        for filing in filings:
            metadata = json.loads(
                Path(filing.metadata_path).read_text(encoding="utf-8-sig")
            )
            raw = RawDocument(
                content=Path(filing.input_path).read_bytes(),
                uri=str(metadata["primary_document_url"]),
                media_type="application/pdf",
                hints=metadata,
            )
            try:
                parsed_dir = output / "parsed" / ticker / filing.source_document_id
                bundle_path = parsed_dir / "bundle.json"
                reuse_bundle = False
                if args.resume_prepared and bundle_path.is_file():
                    from moatrader.canonical.models import CanonicalDocumentBundle

                    bundle = CanonicalDocumentBundle.model_validate_json(
                        bundle_path.read_text(encoding="utf-8-sig")
                    )
                    if bundle.metadata.raw_sha256 != hashlib.sha256(raw.content).hexdigest():
                        raise ValueError(
                            f"ticker {ticker}: prepared bundle raw hash mismatch"
                        )
                    reuse_bundle = (
                        bundle.metadata.parser_version == IR_PDF_PARSER_VERSION
                    )
                if reuse_bundle:
                    bundle = enrich_ir_table_semantics(bundle)
                    _write_json(bundle_path, bundle.model_dump(mode="json"))
                else:
                    bundle = pipeline.ingest(raw)
                    _write_json(bundle_path, bundle.model_dump(mode="json"))
                    parsed_dir.mkdir(parents=True, exist_ok=True)
                    (parsed_dir / "document.md").write_text(
                        pipeline.renderer.render_document(bundle),
                        encoding="utf-8",
                    )
                quality_assessment = assess_parser_quality(bundle)
                run_metadata = {
                    **metadata,
                    "canonical_bundle_path": str(bundle_path.resolve()),
                    "canonical_bundle_raw_sha256": bundle.metadata.raw_sha256,
                    "canonical_bundle_parser_version": bundle.metadata.parser_version,
                }
                run_metadata_path = (
                    output
                    / "run-metadata"
                    / f"{ticker}-{filing.source_document_id}-ir.json"
                )
                _write_json(run_metadata_path, run_metadata)
                prepared_metadata_by_document_id[filing.source_document_id] = str(
                    run_metadata_path
                )
                material = material_by_document_id.get(filing.source_document_id)
                parse_rows.append(
                    {
                        "ticker": ticker,
                        "issuer_name": identities[ticker].issuer_name,
                        "status": "PASS",
                        "source_document_id": bundle.metadata.source_document_id,
                        "listed_on": (
                            material.listed_on.isoformat() if material else ""
                        ),
                        "ir_year": material.listed_on.year if material else "",
                        "page_count": bundle.metadata.source_specific.get("pdf_page_count"),
                        "raw_visible_chars": bundle.quality.raw_visible_chars,
                        "text_retention": bundle.quality.text_retention,
                        "table_count": bundle.quality.ast_table_count,
                        "numeric_cell_count": bundle.quality.numeric_cell_count,
                        "numeric_retention": bundle.quality.numeric_retention,
                        "quality_gate_passed": quality_assessment.passed,
                        "quality_gate_failure_count": len(
                            quality_assessment.failures
                        ),
                        "quality_gate_failures": " | ".join(
                            quality_assessment.failures
                        ),
                        "warning_count": len(bundle.quality.warnings),
                        "ocr_required": any(
                            "OCR_REQUIRED" in warning
                            for warning in bundle.quality.warnings
                        ),
                    }
                )
            except Exception as exc:
                material = material_by_document_id.get(filing.source_document_id)
                parse_rows.append(
                    {
                        "ticker": ticker,
                        "issuer_name": identities[ticker].issuer_name,
                        "status": "FAILED",
                        "source_document_id": filing.source_document_id,
                        "listed_on": (
                            material.listed_on.isoformat() if material else ""
                        ),
                        "ir_year": material.listed_on.year if material else "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    _write_csv(
        output / "parse-audit.csv",
        parse_rows,
        (
            "ticker",
            "issuer_name",
            "status",
            "source_document_id",
            "listed_on",
            "ir_year",
            "page_count",
            "raw_visible_chars",
            "text_retention",
            "table_count",
            "numeric_cell_count",
            "numeric_retention",
            "quality_gate_passed",
            "quality_gate_failure_count",
            "quality_gate_failures",
            "warning_count",
            "ocr_required",
            "error",
        ),
    )

    selected_set = set(selected)
    dart_rows = [
        {field: row.get(field, "") for field in MANIFEST_FIELDS}
        for row in base_rows
        if _ticker(row.get("ticker")) in selected_set
    ]
    combined_rows = list(dart_rows)
    for ticker in selected:
        identity = identity_rows[ticker]
        for filing in filings_by_ticker.get(ticker, []):
            combined_rows.append(
                {
                    "ticker": ticker,
                    "source": "IR",
                    "input": filing.input_path,
                    "metadata": prepared_metadata_by_document_id.get(
                        filing.source_document_id,
                        filing.metadata_path,
                    ),
                    "issuer_id": filing.issuer_id,
                    "issuer_name": filing.issuer_name
                    or identity.get("issuer_name", ""),
                    "current_price": identity.get("current_price", ""),
                    "price_as_of": identity.get("price_as_of", ""),
                    "dcf_assumptions": identity.get("dcf_assumptions", ""),
                }
            )
    _write_csv(output / "dart-only.csv", dart_rows, MANIFEST_FIELDS)
    _write_csv(output / "dart-plus-ir.csv", combined_rows, MANIFEST_FIELDS)
    for shard_index, start in enumerate(range(0, len(selected), 5), start=1):
        shard_tickers = set(selected[start : start + 5])
        _write_csv(
            output / "shards" / f"dart-only-{shard_index:02d}.csv",
            [row for row in dart_rows if _ticker(row.get("ticker")) in shard_tickers],
            MANIFEST_FIELDS,
        )
        _write_csv(
            output / "shards" / f"dart-plus-ir-{shard_index:02d}.csv",
            [row for row in combined_rows if _ticker(row.get("ticker")) in shard_tickers],
            MANIFEST_FIELDS,
        )

    cohort_rows = []
    for ticker in selected:
        selected_materials = [
            material_by_document_id[filing.source_document_id]
            for filing in filings_by_ticker.get(ticker, [])
            if filing.source_document_id in material_by_document_id
        ]
        latest = max(
            selected_materials or by_ticker[ticker],
            key=lambda material: (
                material.listed_on,
                int(material.ir_seq),
                material.attachment_index,
            ),
        )
        ticker_parse_rows = [row for row in parse_rows if row["ticker"] == ticker]
        parse_status = (
            "PASS"
            if ticker_parse_rows
            and all(row["status"] == "PASS" for row in ticker_parse_rows)
            else "FAILED"
        )
        quality_pass_rows = [
            row
            for row in ticker_parse_rows
            if row.get("quality_gate_passed") is True
        ]
        quality_usable_years = sorted(
            {
                int(row["ir_year"])
                for row in quality_pass_rows
                if row.get("ir_year") not in (None, "")
            }
        )
        cohort_rows.append(
            {
                "ticker": ticker,
                "issuer_id": identities[ticker].issuer_id,
                "issuer_name": identities[ticker].issuer_name,
                "selection_key": _selection_key(args.seed, ticker),
                "latest_ir_listed_on": latest.listed_on.isoformat(),
                "latest_ir_seq": latest.ir_seq,
                "latest_ir_filename": latest.attachment_name,
                "ir_document_count": len(selected_materials),
                "ir_year_count": len(
                    {material.listed_on.year for material in selected_materials}
                ),
                "ir_years": ";".join(
                    str(year)
                    for year in sorted(
                        {material.listed_on.year for material in selected_materials}
                    )
                ),
                "parse_status": parse_status,
                "parse_pass_count": sum(
                    row["status"] == "PASS" for row in ticker_parse_rows
                ),
                "parse_fail_count": sum(
                    row["status"] != "PASS" for row in ticker_parse_rows
                ),
                "quality_gate_status": (
                    "PASS"
                    if ticker_parse_rows
                    and len(quality_pass_rows) == len(ticker_parse_rows)
                    else "FAILED"
                ),
                "quality_pass_count": len(quality_pass_rows),
                "quality_fail_count": len(ticker_parse_rows)
                - len(quality_pass_rows),
                "quality_usable_year_count": len(quality_usable_years),
                "quality_usable_years": ";".join(
                    str(year) for year in quality_usable_years
                ),
                "ocr_required": any(
                    row.get("ocr_required") is True for row in ticker_parse_rows
                ),
            }
        )
    _write_csv(
        output / "cohort.csv",
        cohort_rows,
        (
            "ticker",
            "issuer_id",
            "issuer_name",
            "selection_key",
            "latest_ir_listed_on",
            "latest_ir_seq",
            "latest_ir_filename",
            "ir_document_count",
            "ir_year_count",
            "ir_years",
            "parse_status",
            "parse_pass_count",
            "parse_fail_count",
            "quality_gate_status",
            "quality_pass_count",
            "quality_fail_count",
            "quality_usable_year_count",
            "quality_usable_years",
            "ocr_required",
        ),
    )

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": (
            "DART_ONLY_VS_DART_PLUS_LONGITUDINAL_IR"
            if args.ir_selection_mode == "annual-snapshots"
            else "DART_ONLY_VS_DART_PLUS_IR"
        ),
        "as_of": args.as_of,
        "ir_window": {
            "begin_date": args.begin_date.isoformat(),
            "end_date": args.end_date.isoformat(),
            "conservative_availability_rule": "KIND list date + 1 calendar day at 00:00 Asia/Seoul",
        },
        "selection": {
            "policy": "KIND availability + exact normalized issuer identity + seeded SHA-256 only",
            "fixed_cohort": str(Path(args.fixed_cohort).resolve()) if args.fixed_cohort else None,
            "fixed_cohort_filtered_by_ir_years": bool(
                args.fixed_cohort and args.filter_fixed_cohort_by_ir_years
            ),
            "fixed_cohort_excluded_tickers": missing if args.fixed_cohort else [],
            "return_data_used": False,
            "seed": args.seed,
            "eligible_company_count": len(eligible),
            "sample_size": args.sample_size,
            "tickers": selected,
            "ir_selection_mode": args.ir_selection_mode,
            "latest_materials_per_company": (
                1 if args.ir_selection_mode == "latest" else None
            ),
            "materials_per_company_per_year": (
                args.materials_per_company_per_year
                if args.ir_selection_mode == "annual-snapshots"
                else None
            ),
            "max_years_per_company": (
                args.max_years_per_company
                if args.ir_selection_mode == "annual-snapshots"
                else None
            ),
            "minimum_years_per_company": args.minimum_years_per_company,
        },
        "frozen_controls": {
            "same_company_set": True,
            "same_as_of": True,
            "same_prompt": True,
            "same_models": True,
            "same_reducer": True,
            "only_treatment_difference": (
                "addition of one annual PIT-available KIND IR PDF per year"
                if args.ir_selection_mode == "annual-snapshots"
                else "addition of one latest PIT-available KIND IR PDF"
            ),
            "shard_size": 5,
            "incremental_ir_required": True,
            "longitudinal_ir_required": (
                args.ir_selection_mode == "annual-snapshots"
            ),
            "dart_base_byte_identity_required": True,
            "treatment_requires_accepted_ir": True,
        },
        "primary_metrics": [
            "evidence_sufficiency",
            "mechanism_coverage",
            "outcome_coverage",
            "persistence_coverage",
            "counterevidence_coverage",
            "bridge_fail_rate",
            "score_distribution",
            "repeat_stability",
        ],
        "deferred_metrics": ["forward_return", "rank_ic", "q5_minus_q1"],
        "inputs": {
            "base_manifest": str(base_manifest),
            "base_manifest_sha256": _sha256(base_manifest),
            "dart_only_manifest": str(output / "dart-only.csv"),
            "dart_plus_ir_manifest": str(output / "dart-plus-ir.csv"),
        },
        "collection": {
            "downloaded": collection.downloaded_count,
            "unchanged": collection.unchanged_count,
            "failures": [failure.model_dump(mode="json") for failure in collection.failures],
        },
        "parse_status_counts": {
            status: sum(row["status"] == status for row in parse_rows)
            for status in sorted({str(row["status"]) for row in parse_rows})
        },
        "quality_gate": {
            "passed_document_count": sum(
                row.get("quality_gate_passed") is True for row in parse_rows
            ),
            "failed_document_count": sum(
                row.get("status") == "PASS"
                and row.get("quality_gate_passed") is not True
                for row in parse_rows
            ),
            "company_count_with_minimum_usable_years": sum(
                row["quality_usable_year_count"]
                >= args.minimum_years_per_company
                for row in cohort_rows
            ),
        },
        "ocr": {
            "engine": args.ir_ocr_engine,
            "device": args.ir_ocr_device,
            "cpu_threads": args.ir_ocr_cpu_threads,
        },
    }
    _write_json(output / "protocol.json", protocol)
    if collection.failures or any(row["status"] != "PASS" for row in parse_rows):
        raise RuntimeError(
            "source-ablation preparation completed with missing/failed PDFs; inspect collection-result.json and parse-audit.csv"
        )
    return protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and materialize a result-blind DART-only vs DART+IR source ablation"
    )
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--from", dest="begin_date", type=date.fromisoformat, required=True)
    parser.add_argument("--to", dest="end_date", type=date.fromisoformat, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument(
        "--fixed-cohort",
        help="cohort CSV whose ticker order is preserved exactly",
    )
    parser.add_argument(
        "--filter-fixed-cohort-by-ir-years",
        action="store_true",
        help=(
            "preserve fixed-cohort order but exclude companies that fail the "
            "minimum-years-per-company availability rule"
        ),
    )
    parser.add_argument("--seed", default="source-ablation-v1")
    parser.add_argument(
        "--ir-selection-mode",
        choices=["latest", "annual-snapshots"],
        default="latest",
        help="select one latest IR document or one latest document from each year",
    )
    parser.add_argument(
        "--materials-per-company-per-year",
        type=int,
        default=1,
    )
    parser.add_argument("--max-years-per-company", type=int, default=5)
    parser.add_argument(
        "--minimum-years-per-company",
        type=int,
        default=1,
        help="exclude companies with fewer distinct PIT-available IR years",
    )
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-download-mb", type=float, default=256.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--resume-prepared",
        action="store_true",
        help="reuse only raw-hash-matching canonical bundles already created in this output root",
    )
    parser.add_argument("--ir-ocr-engine", choices=["none", "paddle"], default="paddle")
    parser.add_argument("--ir-ocr-device", default="cpu")
    parser.add_argument("--ir-ocr-cpu-threads", type=int, default=6)
    return parser


if __name__ == "__main__":
    result = prepare(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
