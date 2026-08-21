from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
from zoneinfo import ZoneInfo

from lxml import etree, html
from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.canonical.models import StatementType
from moatrader.expectations.future_eri import (
    EvidenceObservation,
    EvidenceState,
    EvidenceVectorStatus,
    FcffEvidenceVectorV1,
    OperatingEvidenceAxis,
    build_fcff_evidence_vector,
    next_usable_signal_timestamp,
)


SEOUL = ZoneInfo("Asia/Seoul")
REGULAR_REPORT_CODES = {
    3: "11013",
    6: "11012",
    9: "11014",
    12: "11011",
}
REGULAR_REPORT_LABELS = ("사업보고서", "반기보고서", "분기보고서")
PERIOD_PATTERN = re.compile(r"(?P<year>20\d{2})[._](?P<month>03|06|09|12)")
REPORT_PERIOD_PATTERN = re.compile(r"\((?P<year>20\d{2})[.](?P<month>03|06|09|12)\)")
DATE_PATTERNS = (
    re.compile(r"\b20\d{2}[-./년]\s*\d{1,2}(?:[-./월]\s*\d{1,2}일?)?\b"),
    re.compile(r"\b20\d{2}\s*년\b"),
    re.compile(r"\b\d{14}\b"),
    re.compile(r"\b\d{6}\b"),
)


AXIS_KEYWORDS: dict[OperatingEvidenceAxis, tuple[str, ...]] = {
    OperatingEvidenceAxis.DEMAND: (
        "수요", "판매", "매출", "출하", "고객 주문", "시장 성장", "시장 침체", "판매량",
    ),
    OperatingEvidenceAxis.PRICE_MIX: (
        "가격", "판매가격", "평균판매가격", "단가", "ASP", "믹스", "mix", "프리미엄",
    ),
    OperatingEvidenceAxis.BACKLOG: (
        "수주", "수주잔고", "수주상황", "계약잔액", "계약금액", "신규 계약", "order backlog",
    ),
    OperatingEvidenceAxis.MARGIN: (
        "마진", "수익성", "영업이익률", "원가", "비용 절감", "채산성", "이익률", "원재료 가격",
    ),
    OperatingEvidenceAxis.INVENTORY_MISMATCH: (
        "재고", "재고조정", "재고자산", "과잉재고", "inventory", "재고 부담", "재고 수준",
    ),
    OperatingEvidenceAxis.CAPACITY_CAPEX: (
        "생산능력", "생산설비", "가동률", "시설투자", "설비투자", "증설", "CAPA", "CAPEX", "생산실적",
    ),
}
FINANCE_STATEMENT_EXTRA_AXIS_KEYWORDS: dict[
    OperatingEvidenceAxis, tuple[str, ...]
] = {
    # Finance-statement tables usually expose capex through account headings,
    # not the narrative terms used by business-info and finance-comment.
    OperatingEvidenceAxis.CAPACITY_CAPEX: (
        "유형자산의 취득",
        "유형자산 취득",
        "건설중인자산",
        "기계장치",
        "시설장치",
    ),
}
_KEYWORD_AXES: dict[str, set[OperatingEvidenceAxis]] = defaultdict(set)
for _axis, _keywords in AXIS_KEYWORDS.items():
    for _keyword in _keywords:
        _KEYWORD_AXES[_keyword.lower()].add(_axis)
_ALL_AXIS_KEYWORD_PATTERN = re.compile(
    "|".join(re.escape(item) for item in sorted(_KEYWORD_AXES, key=len, reverse=True)),
    flags=re.I,
)


class HistoricalSourceOrigin(StrEnum):
    ARCANA_BUSINESS_HTML = "ARCANA_BUSINESS_HTML"
    ARCANA_FINANCE_COMMENT_HTML = "ARCANA_FINANCE_COMMENT_HTML"
    ARCANA_FINANCE_STATEMENT_HTML = "ARCANA_FINANCE_STATEMENT_HTML"
    MOATRADER_OPENDART_ARCHIVE = "MOATRADER_OPENDART_ARCHIVE"


ARCANA_SOURCE_ORIGINS = frozenset(
    {
        HistoricalSourceOrigin.ARCANA_BUSINESS_HTML,
        HistoricalSourceOrigin.ARCANA_FINANCE_COMMENT_HTML,
        HistoricalSourceOrigin.ARCANA_FINANCE_STATEMENT_HTML,
    }
)
ARCANA_DART_SECTION_SPECS: tuple[tuple[str, str, HistoricalSourceOrigin], ...] = (
    ("business-info", "business_info", HistoricalSourceOrigin.ARCANA_BUSINESS_HTML),
    (
        "finance-comment",
        "finance_statement_comment",
        HistoricalSourceOrigin.ARCANA_FINANCE_COMMENT_HTML,
    ),
    (
        "finance-statement",
        "finance_statement",
        HistoricalSourceOrigin.ARCANA_FINANCE_STATEMENT_HTML,
    ),
)


class ReceiptLinkage(StrEnum):
    EXACT_METADATA = "EXACT_METADATA"
    INFERRED_TICKER_PERIOD = "INFERRED_TICKER_PERIOD"


class AxisClassificationStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS = "AMBIGUOUS"


class HistoricalSourceVariant(ContractModel):
    origin: HistoricalSourceOrigin
    path: str = Field(min_length=1)
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(gt=0)
    receipt_linkage: ReceiptLinkage
    metadata_path: str | None = None
    metadata_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    immutable_source: bool = True

    @model_validator(mode="after")
    def source_is_read_only(self) -> "HistoricalSourceVariant":
        if not self.immutable_source:
            raise ValueError("historical source variants are always immutable")
        if (self.metadata_path is None) != (self.metadata_sha256 is None):
            raise ValueError("metadata_path and metadata_sha256 must be supplied together")
        return self


class HistoricalRegularFiling(ContractModel):
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    issuer_name: str = ""
    rcept_no: str = Field(pattern=r"^[0-9]{14}$")
    report_name: str = Field(min_length=1)
    report_code: str = Field(pattern=r"^1101[1234]$")
    fiscal_period_end: date
    published_at: datetime
    available_at: datetime
    signal_timestamp: datetime
    is_amendment: bool = False
    source_variants: list[HistoricalSourceVariant] = Field(min_length=1)

    @model_validator(mode="after")
    def regular_point_in_time_filing(self) -> "HistoricalRegularFiling":
        for field, value in (
            ("published_at", self.published_at),
            ("available_at", self.available_at),
            ("signal_timestamp", self.signal_timestamp),
        ):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field} must be timezone-aware")
        if self.published_at > self.available_at or self.available_at > self.signal_timestamp:
            raise ValueError("filing timestamps violate point-in-time order")
        if self.is_amendment:
            raise ValueError("regular filing catalog cannot contain amendments")
        expected = REGULAR_REPORT_CODES.get(self.fiscal_period_end.month)
        if self.report_code != expected:
            raise ValueError("report_code does not match fiscal-period month")
        if len({item.raw_sha256 for item in self.source_variants}) != len(self.source_variants):
            raise ValueError("duplicate source content must be deduplicated by SHA-256")
        return self


class HistoricalFilingPair(ContractModel):
    pair_id: str = Field(pattern=r"^PAIR_[0-9a-f]{24}$")
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    previous: HistoricalRegularFiling
    current: HistoricalRegularFiling

    @model_validator(mode="after")
    def consecutive_regular_filings(self) -> "HistoricalFilingPair":
        if self.ticker != self.previous.ticker or self.ticker != self.current.ticker:
            raise ValueError("filing pair ticker mismatch")
        if self.current.fiscal_period_end <= self.previous.fiscal_period_end:
            raise ValueError("filing pair must be chronological")
        if not consecutive_fiscal_periods(
            self.previous.fiscal_period_end,
            self.current.fiscal_period_end,
        ):
            raise ValueError("filing pair periods are not consecutive")
        expected = historical_pair_id(
            self.ticker,
            self.previous.rcept_no,
            self.current.rcept_no,
        )
        if self.pair_id != expected:
            raise ValueError("pair_id does not match immutable receipt identifiers")
        return self


class BlindedExcerpt(ContractModel):
    source_id: str = Field(pattern=r"^SRC_[0-9a-f]{20}$")
    text: str = Field(min_length=1, max_length=1600)


class PairedAxisPacket(ContractModel):
    schema_version: str = "moatrader-blinded-evidence-pair-v1/1"
    packet_id: str = Field(pattern=r"^PKT_[0-9a-f]{24}$")
    axis: OperatingEvidenceAxis
    previous_excerpts: list[BlindedExcerpt]
    current_excerpts: list[BlindedExcerpt]
    identifiers_masked: bool = True
    dates_masked: bool = True
    future_context_included: bool = False
    market_data_included: bool = False

    @model_validator(mode="after")
    def contamination_controls(self) -> "PairedAxisPacket":
        if not self.identifiers_masked or not self.dates_masked:
            raise ValueError("historical packets must mask identifiers and dates")
        if self.future_context_included or self.market_data_included:
            raise ValueError("historical packets cannot contain future or market context")
        return self


class AxisPairClassification(ContractModel):
    packet_id: str = Field(pattern=r"^PKT_[0-9a-f]{24}$")
    axis: OperatingEvidenceAxis
    status: AxisClassificationStatus = AxisClassificationStatus.COMPLETE
    previous_state: EvidenceState | None = None
    current_state: EvidenceState | None = None
    previous_source_id: str | None = Field(default=None, pattern=r"^SRC_[0-9a-f]{20}$")
    current_source_id: str | None = Field(default=None, pattern=r"^SRC_[0-9a-f]{20}$")
    previous_source_span: str | None = Field(default=None, min_length=1, max_length=600)
    current_source_span: str | None = Field(default=None, min_length=1, max_length=600)
    confidence: float = Field(ge=0, le=1)
    classification_only: bool = True
    outlook_prediction_made: bool = False

    @model_validator(mode="after")
    def parser_role_only(self) -> "AxisPairClassification":
        if not self.classification_only or self.outlook_prediction_made:
            raise ValueError("LLM role is limited to paired fact classification")
        evidence_fields = (
            self.previous_state,
            self.current_state,
            self.previous_source_id,
            self.current_source_id,
            self.previous_source_span,
            self.current_source_span,
        )
        if self.status == AxisClassificationStatus.COMPLETE and any(
            value is None for value in evidence_fields
        ):
            raise ValueError("complete classification requires both grounded states")
        if self.status != AxisClassificationStatus.COMPLETE and any(
            value is not None for value in evidence_fields
        ):
            raise ValueError("abstained classification cannot contain inferred states")
        return self

    @property
    def delta(self) -> int | None:
        if self.previous_state is None or self.current_state is None:
            return None
        raw = self.current_state.value - self.previous_state.value
        return (raw > 0) - (raw < 0)


class HistoricalSourceIntegrityRecord(ContractModel):
    path: str = Field(min_length=1)
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(gt=0)
    modified_time_ns: int = Field(gt=0)


class HistoricalSourceIntegrityManifest(ContractModel):
    schema_version: str = "moatrader-historical-source-integrity-v1/1"
    created_at: datetime
    mutation_policy: str = "ARCANA_AND_MOATRADER_SOURCE_FILES_READ_ONLY"
    records: list[HistoricalSourceIntegrityRecord]

    @model_validator(mode="after")
    def timestamp_is_aware(self) -> "HistoricalSourceIntegrityManifest":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return self


class HistoricalEvidenceFeatureRowV1(ContractModel):
    schema_version: str = "moatrader-historical-evidence-feature-row-v1/1"
    observation_id: str = Field(pattern=r"^OBS_[0-9a-f]{24}$")
    pair_id: str = Field(pattern=r"^PAIR_[0-9a-f]{24}$")
    issuer_id: str = Field(pattern=r"^[0-9]{6}$")
    previous_rcept_no: str = Field(pattern=r"^[0-9]{14}$")
    current_rcept_no: str = Field(pattern=r"^[0-9]{14}$")
    previous_available_at: datetime
    current_available_at: datetime
    signal_timestamp: datetime
    previous_observations: list[EvidenceObservation] = Field(min_length=6)
    current_observations: list[EvidenceObservation] = Field(min_length=6)
    evidence: FcffEvidenceVectorV1
    coverage_sector: str = "UNMAPPED"
    sector_is_feature: Literal[False] = False
    materiality_policy: Literal["QUALITATIVE_AXIS_STATE_EQUAL_UNIT_V1"] = (
        "QUALITATIVE_AXIS_STATE_EQUAL_UNIT_V1"
    )
    feature_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_data_accessed: Literal[False] = False
    return_data_accessed: Literal[False] = False
    primary_ranking_policy: Literal["NONE_MECHANISM_ONLY"] = "NONE_MECHANISM_ONLY"

    @model_validator(mode="after")
    def complete_and_self_hashed(self) -> "HistoricalEvidenceFeatureRowV1":
        for field_name, value in (
            ("previous_available_at", self.previous_available_at),
            ("current_available_at", self.current_available_at),
            ("signal_timestamp", self.signal_timestamp),
        ):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field_name} must be timezone-aware")
        if self.previous_available_at > self.current_available_at:
            raise ValueError("historical feature periods are not chronological")
        if self.current_available_at > self.signal_timestamp:
            raise ValueError("historical feature was unavailable at signal timestamp")
        if self.evidence.status != EvidenceVectorStatus.COMPLETE:
            raise ValueError("historical feature rows require six complete axes")
        if self.evidence.issuer_id != self.issuer_id:
            raise ValueError("historical feature issuer mismatch")
        if self.evidence.signal_timestamp != self.signal_timestamp:
            raise ValueError("historical feature signal mismatch")
        for observations in (self.previous_observations, self.current_observations):
            if {item.axis for item in observations} != set(OperatingEvidenceAxis):
                raise ValueError("each historical feature side must cover exactly six axes")
            if any(item.issuer_id != self.issuer_id for item in observations):
                raise ValueError("historical observation issuer mismatch")
        payload = self.model_dump(mode="json")
        actual_hash = str(payload.pop("feature_hash"))
        if actual_hash != canonical_payload_sha256(payload):
            raise ValueError("historical feature_hash does not match row payload")
        return self


class HistoricalEvidenceDatasetSealV1(ContractModel):
    schema_version: str = "moatrader-historical-evidence-dataset-seal-v1/1"
    sealed_at: datetime
    feature_count: int = Field(gt=0)
    observation_ids: list[str] = Field(min_length=1)
    feature_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_row_sha256: dict[str, str]
    label_quality_gate_passed: Literal[True] = True
    outcome_source_opened_before_seal: Literal[False] = False
    return_data_accessed: Literal[False] = False
    primary_ranking_policy: Literal["NONE_MECHANISM_ONLY"] = "NONE_MECHANISM_ONLY"

    @model_validator(mode="after")
    def valid_feature_only_seal(self) -> "HistoricalEvidenceDatasetSealV1":
        if self.sealed_at.tzinfo is None or self.sealed_at.utcoffset() is None:
            raise ValueError("sealed_at must be timezone-aware")
        if self.observation_ids != sorted(set(self.observation_ids)):
            raise ValueError("sealed observation IDs must be sorted and unique")
        if self.feature_count != len(self.observation_ids):
            raise ValueError("sealed feature count mismatch")
        if set(self.feature_row_sha256) != set(self.observation_ids):
            raise ValueError("sealed row-hash keys mismatch")
        return self


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def historical_observation_id(pair_id: str) -> str:
    digest = hashlib.sha256(f"HISTORICAL_FEATURE|{pair_id}".encode("utf-8")).hexdigest()
    return f"OBS_{digest[:24]}"


def build_historical_evidence_feature_row(
    *,
    pair: HistoricalFilingPair,
    previous_observations: Sequence[EvidenceObservation],
    current_observations: Sequence[EvidenceObservation],
    coverage_sector: str = "UNMAPPED",
) -> HistoricalEvidenceFeatureRowV1:
    evidence = build_fcff_evidence_vector(
        issuer_id=pair.ticker,
        signal_timestamp=pair.current.signal_timestamp,
        current=current_observations,
        prior=previous_observations,
    )
    draft = HistoricalEvidenceFeatureRowV1.model_construct(
        observation_id=historical_observation_id(pair.pair_id),
        pair_id=pair.pair_id,
        issuer_id=pair.ticker,
        previous_rcept_no=pair.previous.rcept_no,
        current_rcept_no=pair.current.rcept_no,
        previous_available_at=pair.previous.available_at,
        current_available_at=pair.current.available_at,
        signal_timestamp=pair.current.signal_timestamp,
        previous_observations=list(previous_observations),
        current_observations=list(current_observations),
        evidence=evidence,
        coverage_sector=coverage_sector,
        feature_hash="0" * 64,
    )
    payload = draft.model_dump(mode="json", exclude={"feature_hash"})
    return HistoricalEvidenceFeatureRowV1.model_validate(
        {**payload, "feature_hash": canonical_payload_sha256(payload)}
    )


def seal_historical_evidence_features(
    rows: Sequence[HistoricalEvidenceFeatureRowV1],
    *,
    sealed_at: datetime,
) -> HistoricalEvidenceDatasetSealV1:
    if not rows:
        raise ValueError("cannot seal an empty historical evidence dataset")
    ordered = sorted(rows, key=lambda item: item.observation_id)
    ids = [item.observation_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("historical feature observation IDs must be unique")
    if any(item.signal_timestamp > sealed_at for item in ordered):
        raise ValueError("sealed_at cannot precede a historical feature signal")
    payload = [item.model_dump(mode="json") for item in ordered]
    return HistoricalEvidenceDatasetSealV1(
        sealed_at=sealed_at,
        feature_count=len(ordered),
        observation_ids=ids,
        feature_dataset_sha256=canonical_payload_sha256(payload),
        feature_row_sha256={item.observation_id: item.feature_hash for item in ordered},
    )


def source_integrity_record(path: str | Path) -> HistoricalSourceIntegrityRecord:
    source = Path(path).resolve()
    stat = source.stat()
    return HistoricalSourceIntegrityRecord(
        path=str(source),
        raw_sha256=sha256_file(source),
        byte_count=stat.st_size,
        modified_time_ns=stat.st_mtime_ns,
    )


def verify_source_integrity(manifest: HistoricalSourceIntegrityManifest) -> None:
    for expected in manifest.records:
        actual = source_integrity_record(expected.path)
        if actual != expected:
            raise ValueError(f"immutable historical source changed: {expected.path}")


def build_source_integrity_manifest(
    filings: Sequence[HistoricalRegularFiling],
    *,
    created_at: datetime,
) -> HistoricalSourceIntegrityManifest:
    expected_hashes: dict[str, str] = {}
    for filing in filings:
        for variant in filing.source_variants:
            expected_hashes[variant.path] = variant.raw_sha256
            if variant.metadata_path and variant.metadata_sha256:
                expected_hashes[variant.metadata_path] = variant.metadata_sha256
    records: list[HistoricalSourceIntegrityRecord] = []
    for path, expected_hash in sorted(expected_hashes.items()):
        source = Path(path).resolve()
        stat = source.stat()
        records.append(
            HistoricalSourceIntegrityRecord(
                path=str(source),
                raw_sha256=expected_hash,
                byte_count=stat.st_size,
                modified_time_ns=stat.st_mtime_ns,
            )
        )
    return HistoricalSourceIntegrityManifest(created_at=created_at, records=records)


def historical_pair_id(ticker: str, previous_rcept_no: str, current_rcept_no: str) -> str:
    digest = hashlib.sha256(
        f"{ticker.zfill(6)}|{previous_rcept_no}|{current_rcept_no}".encode("utf-8")
    ).hexdigest()
    return f"PAIR_{digest[:24]}"


def packet_id(pair_id: str, axis: OperatingEvidenceAxis) -> str:
    digest = hashlib.sha256(f"{pair_id}|{axis.value}".encode("utf-8")).hexdigest()
    return f"PKT_{digest[:24]}"


def opaque_source_id(raw_sha256: str, pair_id: str, side: str) -> str:
    digest = hashlib.sha256(f"{raw_sha256}|{pair_id}|{side}".encode("utf-8")).hexdigest()
    return f"SRC_{digest[:20]}"


def consecutive_fiscal_periods(previous: date, current: date) -> bool:
    if previous.day != _month_end(previous.year, previous.month):
        return False
    if current.day != _month_end(current.year, current.month):
        return False
    lookup = {
        3: (previous.year, 6),
        6: (previous.year, 9),
        9: (previous.year, 12),
        12: (previous.year + 1, 3),
    }
    return lookup.get(previous.month) == (current.year, current.month)


def _month_end(year: int, month: int) -> int:
    next_month = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    return (next_month - timedelta(days=1)).day


def _period_end(year: int, month: int) -> date:
    return date(year, month, _month_end(year, month))


def _regular_report_name(value: str) -> bool:
    normalized = " ".join(str(value or "").split())
    return any(label in normalized for label in REGULAR_REPORT_LABELS) and "정정" not in normalized


def _normalize_ticker(value: object) -> str | None:
    raw = str(value or "").strip()
    if not re.fullmatch(r"\d{1,6}", raw):
        return None
    normalized = raw.zfill(6)
    return normalized if normalized != "000000" else None


def _published_at_from_receipt(rcept_no: str) -> datetime:
    receipt_date = date.fromisoformat(f"{rcept_no[:4]}-{rcept_no[4:6]}-{rcept_no[6:8]}")
    return datetime.combine(receipt_date, time.max, tzinfo=SEOUL)


def discover_arcana_regular_sources(
    *,
    metadata_path: str | Path,
    business_html_root: str | Path,
    finance_comment_html_root: str | Path | None = None,
    finance_statement_html_root: str | Path | None = None,
    trading_sessions: Sequence[date],
    begin_year: int,
    end_year: int,
    tickers: set[str] | None = None,
    included_sections: Sequence[str] | None = None,
) -> tuple[list[HistoricalRegularFiling], list[dict[str, Any]], dict[str, Any]]:
    metadata_file = Path(metadata_path)
    business_root = Path(business_html_root)
    section_roots = {
        "business-info": business_root,
        "finance-comment": (
            Path(finance_comment_html_root)
            if finance_comment_html_root is not None
            else business_root.parent / "finance-comment"
        ),
        "finance-statement": (
            Path(finance_statement_html_root)
            if finance_statement_html_root is not None
            else business_root.parent / "finance-statement"
        ),
    }
    known_sections = {item[0] for item in ARCANA_DART_SECTION_SPECS}
    selected_sections = (
        set(known_sections) if included_sections is None else set(included_sections)
    )
    if not selected_sections or not selected_sections.issubset(known_sections):
        raise ValueError(f"invalid Arcana DART section selection: {sorted(selected_sections)}")
    section_specs = tuple(
        item for item in ARCANA_DART_SECTION_SPECS if item[0] in selected_sections
    )
    candidates: dict[tuple[str, date], list[dict[str, str]]] = defaultdict(list)
    amendments: list[dict[str, Any]] = []
    with metadata_file.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("source_type") or "") != "comment":
                continue
            period_raw = str(row.get("period_end_date") or "")[:10]
            if not period_raw:
                continue
            period = date.fromisoformat(period_raw)
            if period.year < begin_year or period.year > end_year or period.month not in REGULAR_REPORT_CODES:
                continue
            report_name = str(row.get("report_name") or "")
            ticker = _normalize_ticker(row.get("stock_code"))
            rcept_no = str(row.get("rcept_no") or "")
            if not re.fullmatch(r"\d{14}", rcept_no) or ticker is None:
                continue
            if tickers is not None and ticker not in tickers:
                continue
            if "정정" in report_name:
                amendments.append(dict(row))
                continue
            if not _regular_report_name(report_name):
                continue
            candidates[(ticker, period)].append(dict(row))

    metadata_sha256 = sha256_file(metadata_file)
    filings: list[HistoricalRegularFiling] = []
    section_counts: dict[str, Counter[str]] = {
        section: Counter() for section, _, _ in section_specs
    }
    all_required = 0
    skipped_without_business = 0
    for (ticker, period), rows in sorted(candidates.items()):
        row = min(rows, key=lambda item: str(item["rcept_no"]))
        available: list[tuple[Path, HistoricalSourceOrigin, str]] = []
        for section, prefix, origin in section_specs:
            source = (
                section_roots[section]
                / ticker
                / f"{prefix}_({period.year}.{period.month:02d}).html"
            )
            if not source.is_file():
                section_counts[section]["missing"] += 1
                continue
            if source.stat().st_size == 0:
                section_counts[section]["empty"] += 1
                continue
            section_counts[section]["discovered_nonempty"] += 1
            available.append((source, origin, section))

        if not any(
            origin == HistoricalSourceOrigin.ARCANA_BUSINESS_HTML
            for _, origin, _ in available
        ):
            skipped_without_business += 1
            continue
        if len(available) == len(section_specs):
            all_required += 1

        variants: list[HistoricalSourceVariant] = []
        seen_hashes: set[str] = set()
        for source, origin, section in available:
            raw_sha256 = sha256_file(source)
            if raw_sha256 in seen_hashes:
                section_counts[section]["duplicate_content_hash"] += 1
                continue
            variants.append(
                HistoricalSourceVariant(
                    origin=origin,
                    path=str(source.resolve()),
                    raw_sha256=raw_sha256,
                    byte_count=source.stat().st_size,
                    receipt_linkage=ReceiptLinkage.INFERRED_TICKER_PERIOD,
                    metadata_path=str(metadata_file.resolve()),
                    metadata_sha256=metadata_sha256,
                )
            )
            seen_hashes.add(raw_sha256)
            section_counts[section]["attached"] += 1
        rcept_no = str(row["rcept_no"])
        published = _published_at_from_receipt(rcept_no)
        signal = next_usable_signal_timestamp(published, trading_sessions=trading_sessions)
        filings.append(
            HistoricalRegularFiling(
                ticker=ticker,
                issuer_name=str(row.get("corp_name") or ""),
                rcept_no=rcept_no,
                report_name=" ".join(str(row["report_name"]).split()),
                report_code=REGULAR_REPORT_CODES[period.month],
                fiscal_period_end=period,
                published_at=published,
                available_at=published,
                signal_timestamp=signal,
                source_variants=variants,
            )
        )
    candidate_count = len(candidates)
    audit = {
        "schema_version": "moatrader-arcana-dart-section-audit-v1/1",
        "required_sections": [item[0] for item in section_specs],
        "anchor_policy": "BUSINESS_INFO_REQUIRED_FOR_BACKWARD_COMPATIBLE_FILING_UNIVERSE",
        "candidate_regular_filing_count": candidate_count,
        "regular_filing_count": len(filings),
        "filing_count_with_all_required_sections": all_required,
        "filing_count_with_all_three_sections": (
            all_required if selected_sections == known_sections else 0
        ),
        "filing_count_skipped_without_business_info": skipped_without_business,
        "sections": {
            section: {
                "root": str(section_roots[section].resolve()),
                "discovered_nonempty_count": section_counts[section]["discovered_nonempty"],
                "missing_count": section_counts[section]["missing"],
                "empty_count": section_counts[section]["empty"],
                "duplicate_content_hash_count": section_counts[section][
                    "duplicate_content_hash"
                ],
                "attached_source_variant_count": section_counts[section]["attached"],
                "discovery_coverage": (
                    section_counts[section]["discovered_nonempty"] / candidate_count
                    if candidate_count
                    else 0.0
                ),
            }
            for section, _, _ in section_specs
        },
        "source_files_modified": False,
    }
    return filings, amendments, audit


def discover_arcana_business_sources(
    *,
    metadata_path: str | Path,
    business_html_root: str | Path,
    trading_sessions: Sequence[date],
    begin_year: int,
    end_year: int,
    tickers: set[str] | None = None,
) -> tuple[list[HistoricalRegularFiling], list[dict[str, Any]]]:
    """Backward-compatible wrapper around the three-section Arcana discovery."""

    filings, amendments, _ = discover_arcana_regular_sources(
        metadata_path=metadata_path,
        business_html_root=business_html_root,
        trading_sessions=trading_sessions,
        begin_year=begin_year,
        end_year=end_year,
        tickers=tickers,
        included_sections=("business-info",),
    )
    return filings, amendments


def _archive_candidate(directory: Path) -> Path | None:
    for name in ("original-document.zip", "original.zip"):
        candidate = directory / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _metadata_archive_hash(payload: dict[str, Any], archive: Path) -> str | None:
    if archive.name == "original-document.zip":
        value = payload.get("original_archive_sha256")
        if value:
            return str(value)
    source_specific = payload.get("source_specific") or {}
    value = source_specific.get("archive_sha256")
    if value:
        return str(value)
    files = (payload.get("storage") or {}).get("files") or {}
    for key, item in files.items():
        if Path(str(key)).name == archive.name:
            return str(item)
    return None


def _moatrader_source_priority(path: Path) -> tuple[int, int, str]:
    value = str(path).lower()
    if "historical-validation-v7-2020-2025" in value and "smoke" not in value:
        rank = 0
    elif "data-lake\\bronze" in value or "data-lake/bronze" in value:
        rank = 1
    else:
        rank = 2
    return rank, len(value), value


def discover_moatrader_original_sources(
    *,
    data_lake_root: str | Path,
    trading_sessions: Sequence[date],
    begin_year: int,
    end_year: int,
    tickers: set[str] | None = None,
) -> tuple[list[HistoricalRegularFiling], dict[str, int]]:
    root = Path(data_lake_root)
    by_receipt: dict[
        str,
        list[tuple[Path, Path, dict[str, Any], dict[str, Any]]],
    ] = defaultdict(list)
    metadata_seen = 0
    for directory, _, filenames in os.walk(root):
        if "metadata.json" not in filenames:
            continue
        folder = Path(directory)
        archive = _archive_candidate(folder)
        if archive is None:
            continue
        metadata_seen += 1
        try:
            payload = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        report_name = str(payload.get("report_name") or payload.get("title") or "")
        if not _regular_report_name(report_name) or bool(payload.get("is_amendment", False)):
            continue
        rcept_no = str(
            payload.get("rcept_no")
            or payload.get("source_document_id")
            or (payload.get("source_specific") or {}).get("rcept_no")
            or ""
        )
        ticker = _normalize_ticker(
            payload.get("ticker")
            or payload.get("stock_code")
            or (payload.get("source_specific") or {}).get("stock_code")
        )
        period_raw = str(
            payload.get("fiscal_period_end")
            or payload.get("period_end")
            or payload.get("report_date")
            or ""
        )[:10]
        if not re.fullmatch(r"\d{14}", rcept_no) or ticker is None:
            continue
        if tickers is not None and ticker not in tickers:
            continue
        try:
            period = date.fromisoformat(period_raw)
        except ValueError:
            matched = REPORT_PERIOD_PATTERN.search(report_name)
            if matched is None:
                continue
            period = _period_end(int(matched.group("year")), int(matched.group("month")))
        if period.year < begin_year or period.year > end_year or period.month not in REGULAR_REPORT_CODES:
            continue
        by_receipt[rcept_no].append(
            (
                archive,
                folder / "metadata.json",
                payload,
                {
                    "ticker": ticker,
                    "period": period,
                    "report_name": " ".join(report_name.split()),
                    "issuer_name": str(
                        payload.get("corp_name") or payload.get("issuer_name") or ""
                    ),
                },
            )
        )

    filings: list[HistoricalRegularFiling] = []
    duplicate_copies = 0
    hash_mismatch = 0
    for rcept_no, copies in sorted(by_receipt.items()):
        copies.sort(key=lambda item: _moatrader_source_priority(item[0]))
        duplicate_copies += len(copies) - 1
        archive, metadata_file, payload, canonical = copies[0]
        report_name = str(canonical["report_name"])
        ticker = str(canonical["ticker"])
        period = canonical["period"]
        if not isinstance(period, date):
            raise TypeError("canonical MoatRader filing period must be a date")
        actual_hash = sha256_file(archive)
        expected_hash = _metadata_archive_hash(payload, archive)
        if expected_hash and expected_hash != actual_hash:
            hash_mismatch += 1
            continue
        published = _published_at_from_receipt(rcept_no)
        signal = next_usable_signal_timestamp(published, trading_sessions=trading_sessions)
        filings.append(
            HistoricalRegularFiling(
                ticker=ticker,
                issuer_name=str(canonical["issuer_name"]),
                rcept_no=rcept_no,
                report_name=" ".join(report_name.split()),
                report_code=REGULAR_REPORT_CODES[period.month],
                fiscal_period_end=period,
                published_at=published,
                available_at=published,
                signal_timestamp=signal,
                source_variants=[
                    HistoricalSourceVariant(
                        origin=HistoricalSourceOrigin.MOATRADER_OPENDART_ARCHIVE,
                        path=str(archive.resolve()),
                        raw_sha256=actual_hash,
                        byte_count=archive.stat().st_size,
                        receipt_linkage=ReceiptLinkage.EXACT_METADATA,
                        metadata_path=str(metadata_file.resolve()),
                        metadata_sha256=sha256_file(metadata_file),
                    )
                ],
            )
        )
    return filings, {
        "metadata_with_original_archive": metadata_seen,
        "unique_regular_receipts": len(by_receipt),
        "duplicate_archive_copies_removed": duplicate_copies,
        "archive_hash_mismatch_count": hash_mismatch,
    }


def merge_historical_sources(
    arcana: Sequence[HistoricalRegularFiling],
    moatrader: Sequence[HistoricalRegularFiling],
) -> tuple[list[HistoricalRegularFiling], dict[str, int]]:
    by_period: dict[tuple[str, date], list[HistoricalRegularFiling]] = defaultdict(list)
    for filing in (*moatrader, *arcana):
        by_period[(filing.ticker, filing.fiscal_period_end)].append(filing)
    merged: list[HistoricalRegularFiling] = []
    dual_source = 0
    rcept_conflicts = 0
    for _, rows in sorted(by_period.items()):
        exact = [
            row
            for row in rows
            if any(
                variant.receipt_linkage == ReceiptLinkage.EXACT_METADATA
                for variant in row.source_variants
            )
        ]
        base = min(exact or rows, key=lambda row: row.rcept_no)
        if len({row.rcept_no for row in rows}) > 1:
            rcept_conflicts += 1
        variants: list[HistoricalSourceVariant] = []
        seen_hashes: set[str] = set()
        for row in sorted(rows, key=lambda item: (item.rcept_no, item.source_variants[0].origin.value)):
            for variant in row.source_variants:
                if variant.raw_sha256 not in seen_hashes:
                    variants.append(variant)
                    seen_hashes.add(variant.raw_sha256)
        origins = {item.origin for item in variants}
        if origins.intersection(ARCANA_SOURCE_ORIGINS) and (
            HistoricalSourceOrigin.MOATRADER_OPENDART_ARCHIVE in origins
        ):
            dual_source += 1
        merged_payload = base.model_dump()
        merged_payload["source_variants"] = variants
        merged_payload["issuer_name"] = base.issuer_name or next(
            (row.issuer_name for row in rows if row.issuer_name),
            "",
        )
        merged.append(HistoricalRegularFiling.model_validate(merged_payload))
    return merged, {
        "merged_filing_count": len(merged),
        "dual_source_filing_count": dual_source,
        "receipt_conflict_count": rcept_conflicts,
    }


def build_regular_filing_pairs(
    filings: Sequence[HistoricalRegularFiling],
) -> list[HistoricalFilingPair]:
    by_ticker: dict[str, list[HistoricalRegularFiling]] = defaultdict(list)
    for filing in filings:
        by_ticker[filing.ticker].append(filing)
    pairs: list[HistoricalFilingPair] = []
    for ticker, rows in sorted(by_ticker.items()):
        ordered = sorted(rows, key=lambda row: (row.fiscal_period_end, row.rcept_no))
        for previous, current in zip(ordered, ordered[1:]):
            if not consecutive_fiscal_periods(previous.fiscal_period_end, current.fiscal_period_end):
                continue
            pairs.append(
                HistoricalFilingPair(
                    pair_id=historical_pair_id(ticker, previous.rcept_no, current.rcept_no),
                    ticker=ticker,
                    previous=previous,
                    current=current,
                )
            )
    return pairs


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _visible_document_text(content: bytes, suffix: str) -> str:
    decoded = _decode_text(content)
    if suffix.lower() == ".txt":
        return "\n".join(" ".join(line.split()) for line in decoded.splitlines() if line.strip())
    decoded = re.sub(r"^\s*<\?xml[^>]*\?>", "", decoded, count=1, flags=re.I)
    try:
        root = html.fromstring(decoded)
    except (etree.ParserError, ValueError):
        return ""
    for node in root.xpath("//script|//style|//noscript"):
        node.drop_tree()
    return "\n".join(
        normalized
        for value in root.xpath("//text()")
        if (normalized := " ".join(str(value).replace("\xa0", " ").split()))
    )


def source_variant_text(variant: HistoricalSourceVariant) -> str:
    path = Path(variant.path)
    if variant.origin in ARCANA_SOURCE_ORIGINS:
        return _visible_document_text(path.read_bytes(), path.suffix)
    texts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            suffix = Path(info.filename).suffix.lower()
            if suffix not in {".html", ".htm", ".xhtml", ".xml", ".txt"}:
                continue
            if info.file_size > 128 * 1024 * 1024:
                continue
            text = _visible_document_text(archive.read(info), suffix)
            if text:
                texts.append(text)
    return "\n".join(texts)


def anonymize_historical_text(
    text: str,
    *,
    issuer_name: str,
    ticker: str,
    rcept_no: str,
    brand_terms: Sequence[str] = (),
) -> str:
    result = str(text)
    sensitive = [issuer_name, ticker, rcept_no, *brand_terms]
    for term in sorted({item.strip() for item in sensitive if item and item.strip()}, key=len, reverse=True):
        result = re.sub(re.escape(term), "[ENTITY]", result, flags=re.I)
    for pattern in DATE_PATTERNS:
        result = pattern.sub("[DATE]", result)
    return " ".join(result.split())


def _keyword_windows(
    text: str,
    *,
    keywords: Sequence[str],
    maximum: int = 8,
    window: int = 520,
) -> list[str]:
    # ``source_variant_text`` already emits whitespace-normalized visible-text
    # lines. Re-normalizing a full filing here made the same multi-megabyte
    # archive text split/join six times per pair (once per axis).
    normalized = text
    lowered = normalized.lower()
    centers: list[int] = []
    for keyword in keywords:
        start = 0
        needle = keyword.lower()
        while len(centers) < maximum * 6:
            index = lowered.find(needle, start)
            if index < 0:
                break
            centers.append(index)
            start = index + max(len(needle), 1)
    excerpts: list[str] = []
    seen: set[str] = set()
    for center in sorted(centers):
        start = max(0, center - window // 2)
        end = min(len(normalized), center + window // 2)
        left_break = normalized.rfind("\n", start, center)
        right_break = normalized.find("\n", center, end)
        if left_break >= 0:
            start = left_break + 1
        if right_break >= 0:
            end = right_break
        excerpt = " ".join(normalized[start:end].split())[:1500]
        digest = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        if excerpt and digest not in seen:
            excerpts.append(excerpt)
            seen.add(digest)
        if len(excerpts) >= maximum:
            break
    return excerpts


def _all_axis_keyword_windows(
    text: str,
    *,
    maximum: int = 8,
    window: int = 520,
) -> dict[OperatingEvidenceAxis, list[str]]:
    """Find every axis candidate in one regex pass over a normalized filing."""

    centers: dict[OperatingEvidenceAxis, list[int]] = {
        axis: [] for axis in OperatingEvidenceAxis
    }
    maximum_centers = maximum * 6
    for matched in _ALL_AXIS_KEYWORD_PATTERN.finditer(text):
        for axis in _KEYWORD_AXES[matched.group(0).lower()]:
            if len(centers[axis]) < maximum_centers:
                centers[axis].append(matched.start())
        if all(len(values) >= maximum_centers for values in centers.values()):
            break
    result: dict[OperatingEvidenceAxis, list[str]] = {}
    for axis, axis_centers in centers.items():
        excerpts: list[str] = []
        seen: set[str] = set()
        for center in axis_centers:
            start = max(0, center - window // 2)
            end = min(len(text), center + window // 2)
            left_break = text.rfind("\n", start, center)
            right_break = text.find("\n", center, end)
            if left_break >= 0:
                start = left_break + 1
            if right_break >= 0:
                end = right_break
            excerpt = " ".join(text[start:end].split())[:1500]
            digest = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
            if excerpt and digest not in seen:
                excerpts.append(excerpt)
                seen.add(digest)
            if len(excerpts) >= maximum:
                break
        result[axis] = excerpts
    return result


def source_variant_axis_windows(
    variant: HistoricalSourceVariant,
) -> tuple[str, dict[OperatingEvidenceAxis, list[str]]]:
    text = source_variant_text(variant)
    result = _all_axis_keyword_windows(text)
    if variant.origin == HistoricalSourceOrigin.ARCANA_FINANCE_STATEMENT_HTML:
        for axis, extra_keywords in FINANCE_STATEMENT_EXTRA_AXIS_KEYWORDS.items():
            result[axis] = _keyword_windows(
                text,
                keywords=(*AXIS_KEYWORDS[axis], *extra_keywords),
            )
    return source_variant_window_cache_key(variant), result


def source_variant_window_cache_key(variant: HistoricalSourceVariant) -> str:
    """Keep source-specific extraction rules isolated even for identical bytes."""

    return f"{variant.origin.value}|{variant.raw_sha256}"


def build_blinded_packets(
    pair: HistoricalFilingPair,
    *,
    brand_terms: Sequence[str] = (),
    text_cache: dict[str, str] | None = None,
    window_cache: dict[str, dict[OperatingEvidenceAxis, list[str]]] | None = None,
) -> tuple[list[PairedAxisPacket], dict[str, Any]]:
    cache = text_cache if text_cache is not None else {}
    cached_windows = window_cache if window_cache is not None else {}
    protected_issuer_name = pair.current.issuer_name or pair.previous.issuer_name

    def excerpts(
        filing: HistoricalRegularFiling,
        side: str,
        axis: OperatingEvidenceAxis,
    ) -> list[BlindedExcerpt]:
        by_source: list[list[BlindedExcerpt]] = []
        for variant in filing.source_variants:
            window_key = source_variant_window_cache_key(variant)
            windows_by_axis = cached_windows.get(window_key)
            if windows_by_axis is None:
                text = cache.get(variant.raw_sha256)
                if text is None:
                    text = source_variant_text(variant)
                    if window_cache is None:
                        cache[variant.raw_sha256] = text
                windows_by_axis = _all_axis_keyword_windows(text)
                if variant.origin == HistoricalSourceOrigin.ARCANA_FINANCE_STATEMENT_HTML:
                    for extra_axis, extra_keywords in (
                        FINANCE_STATEMENT_EXTRA_AXIS_KEYWORDS.items()
                    ):
                        windows_by_axis[extra_axis] = _keyword_windows(
                            text,
                            keywords=(*AXIS_KEYWORDS[extra_axis], *extra_keywords),
                        )
                cached_windows[window_key] = windows_by_axis
            source_rows: list[BlindedExcerpt] = []
            for window_text in windows_by_axis[axis]:
                blinded = anonymize_historical_text(
                    window_text,
                    issuer_name=protected_issuer_name,
                    ticker=filing.ticker,
                    rcept_no=filing.rcept_no,
                    brand_terms=brand_terms,
                )
                source_rows.append(
                    BlindedExcerpt(
                        source_id=opaque_source_id(variant.raw_sha256, pair.pair_id, side),
                        text=blinded,
                    )
                )
            if source_rows:
                by_source.append(source_rows)
        unique: list[BlindedExcerpt] = []
        seen: set[str] = set()
        maximum_depth = max((len(rows) for rows in by_source), default=0)
        for index in range(maximum_depth):
            for source_rows in by_source:
                if index >= len(source_rows):
                    continue
                item = source_rows[index]
                digest = hashlib.sha256(item.text.encode("utf-8")).hexdigest()
                if digest not in seen:
                    unique.append(item)
                    seen.add(digest)
                if len(unique) >= 8:
                    return unique
        return unique[:8]

    packets: list[PairedAxisPacket] = []
    private_sources: dict[str, Any] = {}
    for axis in OperatingEvidenceAxis:
        previous = excerpts(pair.previous, "PREVIOUS", axis)
        current = excerpts(pair.current, "CURRENT", axis)
        packets.append(
            PairedAxisPacket(
                packet_id=packet_id(pair.pair_id, axis),
                axis=axis,
                previous_excerpts=previous,
                current_excerpts=current,
            )
        )
    for side, filing in (("previous", pair.previous), ("current", pair.current)):
        for variant in filing.source_variants:
            private_sources[opaque_source_id(variant.raw_sha256, pair.pair_id, side.upper())] = {
                "side": side,
                "ticker": filing.ticker,
                "rcept_no": filing.rcept_no,
                "fiscal_period_end": filing.fiscal_period_end.isoformat(),
                "available_at": filing.available_at.isoformat(),
                "signal_timestamp": filing.signal_timestamp.isoformat(),
                "origin": variant.origin.value,
                "path": variant.path,
                "raw_sha256": variant.raw_sha256,
            }
    return packets, {
        "pair_id": pair.pair_id,
        "ticker": pair.ticker,
        "issuer_name": protected_issuer_name,
        "previous_rcept_no": pair.previous.rcept_no,
        "current_rcept_no": pair.current.rcept_no,
        "previous_period_end": pair.previous.fiscal_period_end.isoformat(),
        "current_period_end": pair.current.fiscal_period_end.isoformat(),
        "signal_timestamp": pair.current.signal_timestamp.isoformat(),
        "sources": private_sources,
    }


def validate_classification_grounding(
    classification: AxisPairClassification,
    packet: PairedAxisPacket,
) -> None:
    if classification.packet_id != packet.packet_id or classification.axis != packet.axis:
        raise ValueError("classification packet identity mismatch")
    if classification.status != AxisClassificationStatus.COMPLETE:
        return
    assert classification.previous_source_id is not None
    assert classification.current_source_id is not None
    assert classification.previous_source_span is not None
    assert classification.current_source_span is not None
    previous: dict[str, list[str]] = defaultdict(list)
    current: dict[str, list[str]] = defaultdict(list)
    for item in packet.previous_excerpts:
        previous[item.source_id].append(item.text)
    for item in packet.current_excerpts:
        current[item.source_id].append(item.text)
    if classification.previous_source_id not in previous:
        raise ValueError("previous classification source is absent from packet")
    if classification.current_source_id not in current:
        raise ValueError("current classification source is absent from packet")
    if not any(
        classification.previous_source_span in text
        for text in previous[classification.previous_source_id]
    ):
        raise ValueError("previous source span is not verbatim packet evidence")
    if not any(
        classification.current_source_span in text
        for text in current[classification.current_source_id]
    ):
        raise ValueError("current source span is not verbatim packet evidence")


def validate_packet_anonymization(
    packet: PairedAxisPacket,
    *,
    issuer_name: str,
    ticker: str,
    rcept_numbers: Sequence[str],
    brand_terms: Sequence[str] = (),
) -> None:
    excerpt_texts = [
        item.text for item in (*packet.previous_excerpts, *packet.current_excerpts)
    ]
    combined = " ".join(excerpt_texts)
    sensitive = [issuer_name, ticker, *rcept_numbers, *brand_terms]
    for term in sensitive:
        if term and term.strip() and term.lower() in combined.lower():
            raise ValueError(f"blinded packet leaked a protected identifier: {packet.packet_id}")
    # Validate each independently anonymized excerpt. Joining excerpt A and B
    # can fabricate a date at the boundary (for example ``2018`` + ``. 05``)
    # that was never present in either LLM input field.
    if any(pattern.search(text) for text in excerpt_texts for pattern in DATE_PATTERNS):
        raise ValueError(f"blinded packet leaked a historical date: {packet.packet_id}")


def packet_coverage_report(
    packets: Sequence[PairedAxisPacket],
    *,
    private_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    private_by_pair = {str(item["pair_id"]): item for item in private_rows}
    packet_pair = {
        packet_id(pair_id, axis): pair_id
        for pair_id in private_by_pair
        for axis in OperatingEvidenceAxis
    }
    by_axis = {
        axis.value: {
            "packet_count": sum(packet.axis == axis for packet in packets),
            "both_periods_have_candidate_spans": sum(
                packet.axis == axis and bool(packet.previous_excerpts) and bool(packet.current_excerpts)
                for packet in packets
            ),
        }
        for axis in OperatingEvidenceAxis
    }
    complete_pairs = 0
    by_year: Counter[int] = Counter()
    packets_by_pair: dict[str, list[PairedAxisPacket]] = defaultdict(list)
    for packet in packets:
        pair_id_value = packet_pair.get(packet.packet_id)
        if pair_id_value:
            packets_by_pair[pair_id_value].append(packet)
    for pair_id_value, values in packets_by_pair.items():
        if len(values) == 6 and all(item.previous_excerpts and item.current_excerpts for item in values):
            complete_pairs += 1
            by_year[date.fromisoformat(private_by_pair[pair_id_value]["current_period_end"]).year] += 1
    pair_count = len(private_rows)
    return {
        "schema_version": "moatrader-historical-packet-coverage-v1/1",
        "total_filing_pairs": pair_count,
        "axis_packet_count": len(packets),
        "six_axis_candidate_complete": complete_pairs,
        "candidate_coverage": complete_pairs / pair_count if pair_count else 0.0,
        "unique_issuers": len({str(item["ticker"]) for item in private_rows}),
        "by_axis": by_axis,
        "complete_by_fiscal_year": {str(key): value for key, value in sorted(by_year.items())},
        "outcomes_opened": False,
        "returns_opened": False,
    }
