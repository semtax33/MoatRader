from __future__ import annotations

import io
import zipfile
from datetime import date
from decimal import Decimal

from moatrader.financial.historical_xbrl import parse_dart_ifrs_archive


def _archive(instance: str) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("sample.xbrl", instance.encode("utf-8"))
    return target.getvalue()


def test_parse_dart_ifrs_archive_extracts_consolidated_annual_metrics() -> None:
    instance = """<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
 xmlns:xbrldi="http://xbrl.org/2006/xbrldi"
 xmlns:ifrs="http://xbrl.ifrs.org/taxonomy/2020-03-16/ifrs-full"
 xmlns:dart="http://dart.fss.or.kr/taxonomy/2020-01-01/dart">
  <xbrli:context id="D2020Consolidated">
    <xbrli:entity><xbrli:identifier scheme="test">1</xbrli:identifier>
      <xbrli:segment><xbrldi:explicitMember dimension="dart:ConsolidationAxis">dart:ConsolidatedMember</xbrldi:explicitMember></xbrli:segment>
    </xbrli:entity>
    <xbrli:period><xbrli:startDate>2020-01-01</xbrli:startDate><xbrli:endDate>2020-12-31</xbrli:endDate></xbrli:period>
  </xbrli:context>
  <xbrli:context id="I2020Consolidated">
    <xbrli:entity><xbrli:identifier scheme="test">1</xbrli:identifier>
      <xbrli:segment><xbrldi:explicitMember dimension="dart:ConsolidationAxis">dart:ConsolidatedMember</xbrldi:explicitMember></xbrli:segment>
    </xbrli:entity>
    <xbrli:period><xbrli:instant>2020-12-31</xbrli:instant></xbrli:period>
  </xbrli:context>
  <xbrli:context id="D2020Separate">
    <xbrli:entity><xbrli:identifier scheme="test">1</xbrli:identifier>
      <xbrli:segment><xbrldi:explicitMember dimension="dart:ConsolidationAxis">dart:SeparateMember</xbrldi:explicitMember></xbrli:segment>
    </xbrli:entity>
    <xbrli:period><xbrli:startDate>2020-01-01</xbrli:startDate><xbrli:endDate>2020-12-31</xbrli:endDate></xbrli:period>
  </xbrli:context>
  <ifrs:Revenue contextRef="D2020Consolidated">1000</ifrs:Revenue>
  <ifrs:Revenue contextRef="D2020Separate">777</ifrs:Revenue>
  <ifrs:OperatingIncomeLoss contextRef="D2020Consolidated">130</ifrs:OperatingIncomeLoss>
  <ifrs:PurchaseOfPropertyPlantAndEquipment contextRef="D2020Consolidated">-40</ifrs:PurchaseOfPropertyPlantAndEquipment>
  <ifrs:PurchaseOfIntangibleAssets contextRef="D2020Consolidated">-10</ifrs:PurchaseOfIntangibleAssets>
  <ifrs:DepreciationAndAmortisation contextRef="D2020Consolidated">25</ifrs:DepreciationAndAmortisation>
  <ifrs:CashAndCashEquivalents contextRef="I2020Consolidated">90</ifrs:CashAndCashEquivalents>
  <ifrs:ShortTermBorrowings contextRef="I2020Consolidated">30</ifrs:ShortTermBorrowings>
  <ifrs:LongTermBorrowings contextRef="I2020Consolidated">70</ifrs:LongTermBorrowings>
  <ifrs:TradeReceivables contextRef="I2020Consolidated">50</ifrs:TradeReceivables>
  <ifrs:Inventories contextRef="I2020Consolidated">20</ifrs:Inventories>
  <ifrs:TradePayables contextRef="I2020Consolidated">35</ifrs:TradePayables>
</xbrli:xbrl>"""

    metrics = parse_dart_ifrs_archive(_archive(instance), fiscal_year=2020)

    assert metrics.revenue == Decimal("1000")
    assert metrics.ebit == Decimal("130")
    assert metrics.capex == Decimal("50")
    assert metrics.depreciation == Decimal("25")
    assert metrics.cash == Decimal("90")
    assert metrics.debt == Decimal("100")
    assert metrics.nwc == Decimal("35")
    assert metrics.metric_coverage_count == 7
    assert metrics.instance_member == "sample.xbrl"


def test_parse_dart_ifrs_archive_rejects_missing_instance() -> None:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as output:
        output.writestr("readme.txt", "no instance")

    try:
        parse_dart_ifrs_archive(target.getvalue(), fiscal_year=2020)
    except ValueError as exc:
        assert "no XBRL instance" in str(exc)
    else:
        raise AssertionError("missing XBRL instance should be rejected")


def test_parse_dart_ifrs_archive_supports_non_december_fiscal_year_end() -> None:
    instance = """<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
 xmlns:ifrs="http://xbrl.ifrs.org/taxonomy/2020-03-16/ifrs-full">
  <xbrli:context id="D2021"><xbrli:entity><xbrli:identifier scheme="t">1</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:startDate>2020-04-01</xbrli:startDate><xbrli:endDate>2021-03-31</xbrli:endDate></xbrli:period>
  </xbrli:context>
  <xbrli:context id="I2021"><xbrli:entity><xbrli:identifier scheme="t">1</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:instant>2021-03-31</xbrli:instant></xbrli:period>
  </xbrli:context>
  <ifrs:Revenue contextRef="D2021">500</ifrs:Revenue>
  <ifrs:OperatingIncomeLoss contextRef="D2021">55</ifrs:OperatingIncomeLoss>
  <ifrs:CashAndCashEquivalents contextRef="I2021">20</ifrs:CashAndCashEquivalents>
</xbrli:xbrl>"""

    metrics = parse_dart_ifrs_archive(
        _archive(instance),
        fiscal_year=2021,
        period_end=date(2021, 3, 31),
    )

    assert metrics.revenue == Decimal("500")
    assert metrics.ebit == Decimal("55")
    assert metrics.cash == Decimal("20")
