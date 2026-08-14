from __future__ import annotations

from pathlib import Path

import pytest

from moatrader.universe import load_universe_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_loads_and_selects_tickers(tmp_path: Path) -> None:
    source = ROOT / "examples" / "sample-dart.html"
    metadata = ROOT / "examples" / "sample-dart-metadata.json"
    manifest = tmp_path / "universe.csv"
    manifest.write_text(
        "ticker,source,input,metadata,issuer_name\n"
        f"AAA,DART,{source},{metadata},Alpha\n"
        f"AAA,DART,{source},{metadata},Alpha\n"
        f"BBB,SEC,{source},{metadata},Beta\n",
        encoding="utf-8",
    )

    loaded = load_universe_manifest(manifest)

    assert [company.ticker for company in loaded.companies] == ["AAA", "BBB"]
    assert len(loaded.companies[0].documents) == 2
    assert loaded.companies[1].documents[0].source.value == "SEC_EDGAR"
    assert [company.ticker for company in loaded.select({"BBB"})] == ["BBB"]
    with pytest.raises(ValueError, match="not found"):
        loaded.select({"MISSING"})


def test_manifest_rejects_naive_price_timestamp(tmp_path: Path) -> None:
    source = ROOT / "examples" / "sample-dart.html"
    metadata = ROOT / "examples" / "sample-dart-metadata.json"
    manifest = tmp_path / "universe.csv"
    manifest.write_text(
        "ticker,source,input,metadata,current_price,price_as_of\n"
        f"AAA,DART,{source},{metadata},10,2025-01-01T09:00:00\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="timezone"):
        load_universe_manifest(manifest)


def test_manifest_rejects_path_traversal_ticker(tmp_path: Path) -> None:
    source = ROOT / "examples" / "sample-dart.html"
    metadata = ROOT / "examples" / "sample-dart-metadata.json"
    manifest = tmp_path / "universe.csv"
    manifest.write_text(
        "ticker,source,input,metadata\n"
        f"../escape,DART,{source},{metadata}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ticker"):
        load_universe_manifest(manifest)
