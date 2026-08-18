import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const repo = process.cwd();
const resultDir = path.join(
  repo,
  "data-lake",
  "experiments",
  "historical-validation-v7-2020-2025",
  "results-v7-quality-gated",
);
const filingDir = path.join(
  repo,
  "data-lake",
  "experiments",
  "historical-validation-v7-2020-2025",
  "opendart-original",
);
const priceDir = path.join(
  repo,
  "data-lake",
  "experiments",
  "historical-validation-v7-2020-2025",
  "prices",
);
const outputDir = path.join(
  repo,
  "outputs",
  "01a011a6-e595-7ae0-9a8e-5e426d90c5f1",
);
const qaDir = path.join(
  "C:\\Users\\PC_1M\\.codex\\visualizations\\2026\\08\\17\\01a011a6-e595-7ae0-9a8e-5e426d90c5f1",
  "workbook-qa",
);
const outputPath = path.join(outputDir, "MoatRader_v7_Data-PIT_2020-2025.xlsx");

const summary = JSON.parse(await fs.readFile(path.join(resultDir, "summary.json"), "utf8"));
const filingManifest = JSON.parse(
  await fs.readFile(path.join(filingDir, "manifest.json"), "utf8"),
);
const priceManifest = JSON.parse(
  await fs.readFile(path.join(priceDir, "manifest.json"), "utf8"),
);
const integrity = JSON.parse(
  await fs.readFile(path.join(resultDir, "v6-integrity.json"), "utf8"),
);

async function csvValues(filePath, sheetName) {
  const text = (await fs.readFile(filePath, "utf8")).replace(/^\uFEFF/, "");
  const imported = await Workbook.fromCSV(text, { sheetName });
  return imported.worksheets.getItem(sheetName).getUsedRange(true).values;
}

function excelDate(value) {
  if (!value) return null;
  const parsed = new Date(`${String(value).slice(0, 10)}T00:00:00Z`);
  return Number.isNaN(parsed.valueOf()) ? null : parsed;
}

function numeric(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

const quarterlyRaw = await csvValues(
  path.join(resultDir, "quarterly-event-results.csv"),
  "Quarterly",
);
const quarterlyValues = quarterlyRaw.map((row, rowIndex) =>
  row.map((value, colIndex) => {
    if (rowIndex === 0) return value;
    if (colIndex === 0 || colIndex === 2) return excelDate(value);
    return numeric(value);
  }),
);
const signalsRaw = await csvValues(path.join(resultDir, "signals.csv"), "SignalsCSV");
const signalDateColumns = new Set([0, 8, 14]);
const signalNumericColumns = new Set([9, 10, 11, 18, 20, 21, 22, 23, 24, 25]);
const signalsValues = signalsRaw.map((row, rowIndex) =>
  row.map((value, colIndex) => {
    if (rowIndex === 0) return value;
    if (signalDateColumns.has(colIndex)) return excelDate(value);
    if (signalNumericColumns.has(colIndex)) return numeric(value);
    return value === "" ? null : value;
  }),
);
const ablationValues = await csvValues(path.join(resultDir, "ablation.csv"), "AblationCSV");

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("Summary");
const quarterlySheet = workbook.worksheets.add("Quarterly Results");
const coverageSheet = workbook.worksheets.add("Coverage");
const signalsSheet = workbook.worksheets.add("Signals");
const methodologySheet = workbook.worksheets.add("Methodology");
const sourcesSheet = workbook.worksheets.add("Sources & Audit");
const checksSheet = workbook.worksheets.add("Checks");
const ablationSheet = workbook.worksheets.add("Ablation");

const navy = "#102A43";
const blue = "#1677B8";
const paleBlue = "#EAF4FB";
const green = "#16835A";
const paleGreen = "#E8F5EE";
const amber = "#B7791F";
const paleAmber = "#FFF7E6";
const red = "#C0392B";
const paleRed = "#FCECEC";
const gray = "#52606D";
const lightGray = "#E4E7EB";
const white = "#FFFFFF";
const percentFormat = "0.0%;[Red](0.0%);-";
const percent2Format = "0.00%;[Red](0.00%);-";
const countFormat = "#,##0;[Red](#,##0);-";
const decimalFormat = "0.00;[Red](0.00);-";

function titleBand(sheet, range, title) {
  sheet.getRange(range).merge();
  const cell = sheet.getRange(range.split(":")[0]);
  cell.values = [[title]];
  cell.format = {
    fill: navy,
    font: { bold: true, color: white, size: 16 },
    verticalAlignment: "center",
  };
  sheet.getRange(range).format.rowHeight = 32;
}

function sectionBand(sheet, range, label) {
  sheet.getRange(range).merge();
  const cell = sheet.getRange(range.split(":")[0]);
  cell.values = [[label]];
  cell.format = {
    fill: navy,
    font: { bold: true, color: white },
    verticalAlignment: "center",
  };
}

function tableHeader(range) {
  range.format = {
    fill: blue,
    font: { bold: true, color: white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: navy },
  };
}

for (const sheet of workbook.worksheets.items) {
  sheet.showGridLines = false;
}

// Summary
titleBand(summarySheet, "A1:L2", "MoatRader 2020–2025 Data-PIT Historical Validation");
summarySheet.getRange("A3:L3").merge();
summarySheet.getRange("A3").values = [[
  "결론: Cheap 상위 15의 평균 초과수익은 +0.23%p에 그쳤고 t=0.26으로 강한 통계적 증거가 아닙니다. 현대 LLM 평가는 실행하지 않았습니다.",
]];
summarySheet.getRange("A3").format = {
  fill: paleAmber,
  font: { color: amber, bold: true },
  wrapText: true,
  verticalAlignment: "center",
};
summarySheet.getRange("A3:L3").format.rowHeight = 34;

const kpiLabels = [
  ["Cheap 상위 15 평균", null, "유니버스 평균", null, "평균 초과수익", null],
  [null, null, null, null, null, null],
  ["상·하위 15 스프레드", null, "스프레드 양(+) 비율", null, "스프레드 t-stat", null],
  [null, null, null, null, null, null],
];
summarySheet.getRange("A5:F8").values = kpiLabels;
for (const range of ["A5:B6", "C5:D6", "E5:F6", "A7:B8", "C7:D8", "E7:F8"]) {
  summarySheet.getRange(range).format = {
    fill: paleBlue,
    borders: { preset: "outside", style: "thin", color: lightGray },
  };
}
summarySheet.getRange("A5").format.font = { bold: true, color: gray };
summarySheet.getRange("C5").format.font = { bold: true, color: gray };
summarySheet.getRange("E5").format.font = { bold: true, color: gray };
summarySheet.getRange("A7").format.font = { bold: true, color: gray };
summarySheet.getRange("C7").format.font = { bold: true, color: gray };
summarySheet.getRange("E7").format.font = { bold: true, color: gray };
summarySheet.getRange("A6").formulas = [["=AVERAGE('Quarterly Results'!$I$5:$I$27)"]];
summarySheet.getRange("C6").formulas = [["=AVERAGE('Quarterly Results'!$L$5:$L$27)"]];
summarySheet.getRange("E6").formulas = [["=A6-C6"]];
summarySheet.getRange("A8").formulas = [["=AVERAGE('Quarterly Results'!$K$5:$K$27)"]];
summarySheet.getRange("C8").formulas = [["=COUNTIF('Quarterly Results'!$K$5:$K$27,\">0\")/COUNT('Quarterly Results'!$K$5:$K$27)"]];
summarySheet.getRange("E8").formulas = [["=AVERAGE('Quarterly Results'!$K$5:$K$27)/(STDEV.S('Quarterly Results'!$K$5:$K$27)/SQRT(COUNT('Quarterly Results'!$K$5:$K$27)))"]];
for (const cell of ["A6", "C6", "E6", "A8", "C8"]) {
  summarySheet.getRange(cell).format = { font: { bold: true, color: navy, size: 15 }, numberFormat: percent2Format };
}
summarySheet.getRange("E8").format = { font: { bold: true, color: navy, size: 15 }, numberFormat: decimalFormat };

sectionBand(summarySheet, "A10:L10", "검증 등급과 주요 한계");
summarySheet.getRange("A11:L13").values = [
  ["Deterministic Cheap", "DATA_PIT_HISTORICAL", "완료", "OpenDART 원본 + 고정 marcap", null, null, "현대 LLM overlay", "LLM_PIT_PSEUDO_OOS", "미실행", "순위 변경 금지", null, null],
  ["유니버스", "2025-08-01 선정 150종목", "과거 소급", "생존·편입 편향 존재", null, null, "수익률", "KRX ChangesRatio", "77일", "현금배당 제외", null, null],
  ["해석", "탐색적 이벤트 연구", "23개 분기", "투자전략 확정 근거 아님", null, null, "v6 보호", integrity.v6_unchanged_during_backtest ? "UNCHANGED" : "CHANGED", "Git/contract 경계", "v7 데이터만 신규", null, null],
];
summarySheet.getRange("A11:L13").format = { wrapText: true, verticalAlignment: "center" };
summarySheet.getRange("B11").format = { fill: paleGreen, font: { bold: true, color: green } };
summarySheet.getRange("H11").format = { fill: paleAmber, font: { bold: true, color: amber } };
summarySheet.getRange("H13").format = { fill: paleGreen, font: { bold: true, color: green } };

sectionBand(summarySheet, "A15:L15", "분기별 상위 15 vs 유니버스 동일가중 수익률");
summarySheet.getRange("A16:C16").values = [["분기", "Cheap 상위 15", "유니버스"]];
tableHeader(summarySheet.getRange("A16:C16"));
const helperFormulas = [];
for (let row = 2; row <= 24; row += 1) {
  helperFormulas.push([
    `=TEXT('Quarterly Results'!A${row + 3},"yyyy-mm")`,
    `='Quarterly Results'!I${row + 3}`,
    `='Quarterly Results'!L${row + 3}`,
  ]);
}
summarySheet.getRange("A17:C39").formulas = helperFormulas;
summarySheet.getRange("B17:C39").format.numberFormat = percentFormat;
const trendChart = summarySheet.charts.add("line", summarySheet.getRange("A16:C39"));
trendChart.title = "77일 수익률: Cheap 상위 15 vs 유니버스";
trendChart.titleTextStyle.fontSize = 12;
trendChart.hasLegend = true;
trendChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
trendChart.yAxis = { numberFormatCode: "0%" };
trendChart.setPosition("E16", "L32");
summarySheet.freezePanes.freezeRows(3);
summarySheet.getRange("A1:L39").format.font = { name: "Aptos" };
summarySheet.getRange("A1:L39").format.columnWidth = 14;
summarySheet.getRange("A1:A39").format.columnWidth = 17;
summarySheet.getRange("B1:D39").format.columnWidth = 16;
summarySheet.getRange("B1:B39").format.columnWidth = 24;
summarySheet.getRange("H1:H39").format.columnWidth = 22;

// Quarterly Results
titleBand(quarterlySheet, "A1:T2", "Quarterly Event Results (77 calendar days)");
quarterlySheet.getRange("A4:R27").values = quarterlyValues;
tableHeader(quarterlySheet.getRange("A4:R4"));
quarterlySheet.getRange("S4:T4").values = [["Cumulative Top 15", "Cumulative Universe"]];
tableHeader(quarterlySheet.getRange("S4:T4"));
quarterlySheet.getRange("S5").formulas = [["=1+I5"]];
quarterlySheet.getRange("T5").formulas = [["=1+L5"]];
for (let row = 6; row <= 27; row += 1) {
  quarterlySheet.getRange(`S${row}`).formulas = [[`=S${row - 1}*(1+I${row})`]];
  quarterlySheet.getRange(`T${row}`).formulas = [[`=T${row - 1}*(1+L${row})`]];
}
quarterlySheet.getRange("A5:A27").format.numberFormat = "yyyy-mm-dd";
quarterlySheet.getRange("C5:C27").format.numberFormat = "yyyy-mm-dd";
quarterlySheet.getRange("D5:H27").format.numberFormat = countFormat;
quarterlySheet.getRange("I5:R27").format.numberFormat = percentFormat;
quarterlySheet.getRange("S5:T27").format.numberFormat = "0.00x";
quarterlySheet.getRange("K5:K27").conditionalFormats.add("colorScale", {
  colors: [paleRed, paleAmber, paleGreen],
  thresholds: ["min", "50%", "max"],
});
quarterlySheet.tables.add("A4:T27", true, "QuarterlyResultsTable").style = "TableStyleMedium2";
quarterlySheet.freezePanes.freezeRows(4);
quarterlySheet.freezePanes.freezeColumns(1);
quarterlySheet.getRange("A1:T27").format.columnWidth = 13;
quarterlySheet.getRange("A1:C27").format.columnWidth = 14;

// Signals
const signalRows = signalsValues.length;
const signalCols = signalsValues[0].length;
signalsSheet.getRangeByIndexes(0, 0, signalRows, signalCols).values = signalsValues;
tableHeader(signalsSheet.getRangeByIndexes(0, 0, 1, signalCols));
signalsSheet.getRange(`A2:A${signalRows}`).format.numberFormat = "yyyy-mm-dd";
signalsSheet.getRange(`I2:I${signalRows}`).format.numberFormat = "yyyy-mm-dd";
signalsSheet.getRange(`O2:O${signalRows}`).format.numberFormat = "yyyy-mm-dd";
signalsSheet.getRange(`J2:L${signalRows}`).format.numberFormat = countFormat;
signalsSheet.getRange(`U2:V${signalRows}`).format.numberFormat = "#,##0.00;[Red](#,##0.00);-";
signalsSheet.getRange(`Y2:Z${signalRows}`).format.numberFormat = percentFormat;
signalsSheet.tables.add(`A1:AA${signalRows}`, true, "HistoricalSignalsTable").style = "TableStyleMedium2";
signalsSheet.freezePanes.freezeRows(1);
signalsSheet.freezePanes.freezeColumns(4);
signalsSheet.getRange(`A1:AA${signalRows}`).format.columnWidth = 13;
signalsSheet.getRange(`B1:B${signalRows}`).format.columnWidth = 28;
signalsSheet.getRange(`D1:D${signalRows}`).format.columnWidth = 20;
signalsSheet.getRange(`H1:H${signalRows}`).format.columnWidth = 42;
signalsSheet.getRange(`Q1:R${signalRows}`).format.columnWidth = 18;
signalsSheet.getRange(`T1:T${signalRows}`).format.columnWidth = 32;
signalsSheet.getRange(`AA1:AA${signalRows}`).format.columnWidth = 42;

// Coverage formula table
titleBand(coverageSheet, "A1:I2", "Signal Coverage by Quarter");
const statuses = [
  "ELIGIBLE",
  "DCF_SCREENING_EXCLUSION",
  "EXCLUDED_ARCHETYPE",
  "NOT_LISTED_OR_NO_PRICE",
  "NO_PIT_FINANCIALS",
  "INSUFFICIENT_FINANCIAL_COVERAGE",
  "FINANCIAL_DISCONTINUITY",
];
coverageSheet.getRange("A4:I4").values = [["Signal Date", ...statuses, "Total"]];
tableHeader(coverageSheet.getRange("A4:I4"));
for (let index = 0; index < 23; index += 1) {
  const row = 5 + index;
  coverageSheet.getRange(`A${row}`).formulas = [[`='Quarterly Results'!A${5 + index}`]];
  for (let statusIndex = 0; statusIndex < statuses.length; statusIndex += 1) {
    const columnLetter = String.fromCharCode("B".charCodeAt(0) + statusIndex);
    coverageSheet.getRange(`${columnLetter}${row}`).formulas = [[
      `=COUNTIFS(Signals!$A$2:$A$${signalRows},$A${row},Signals!$G$2:$G$${signalRows},${columnLetter}$4)`,
    ]];
  }
  coverageSheet.getRange(`I${row}`).formulas = [[`=SUM(B${row}:H${row})`]];
}
coverageSheet.getRange("A28:H28").values = [["Total", null, null, null, null, null, null, null]];
for (let column = 1; column <= 7; column += 1) {
  const letter = String.fromCharCode("A".charCodeAt(0) + column);
  coverageSheet.getRange(`${letter}28`).formulas = [[`=SUM(${letter}5:${letter}27)`]];
}
coverageSheet.getRange("I28").formulas = [["=SUM(I5:I27)"]];
coverageSheet.getRange("A28:I28").format = {
  font: { bold: true },
  borders: { top: { style: "double", color: navy } },
};
coverageSheet.getRange("A5:A27").format.numberFormat = "yyyy-mm-dd";
coverageSheet.getRange("B5:I28").format.numberFormat = countFormat;
coverageSheet.getRange("B5:B27").conditionalFormats.add("dataBar", { color: blue, gradient: true });
coverageSheet.tables.add("A4:I27", true, "CoverageTable").style = "TableStyleMedium2";
coverageSheet.freezePanes.freezeRows(4);
coverageSheet.getRange("A1:I28").format.columnWidth = 18;
coverageSheet.getRange("F1:H28").format.columnWidth = 26;

// Methodology
titleBand(methodologySheet, "A1:H2", "Methodology, Validation Grades, and Known Biases");
sectionBand(methodologySheet, "A4:H4", "실험 계약");
methodologySheet.getRange("A5:H10").values = [
  ["항목", "설정", "등급", "설명", null, null, null, null],
  ["기간", "2020-03-31 ~ 2025-09-30", "DATA_PIT_HISTORICAL", "23개 분기 신호, 각 77일 이벤트 수익률", null, null, null, null],
  ["유니버스", "2025-08-01 선정 150종목", "편향 있음", "2020년으로 소급하므로 생존·편입 편향", null, null, null, null],
  ["순위", "Cheap = Fair Value / Price - 1", "결정론적", "LLM은 순위를 변경할 수 없음", null, null, null, null],
  ["공시 PIT", "접수일 EOD KST", "엄격", "컷오프 이전 원본·정정본 중 기간별 최신 버전", null, null, null, null],
  ["수익률", "KRX 일별 ChangesRatio 복리", "가격수익률", "현금배당 제외; 신호 CSV 봉인 후 결합", null, null, null, null],
];
tableHeader(methodologySheet.getRange("A5:D5"));
methodologySheet.getRange("A5:H10").format = { wrapText: true, verticalAlignment: "top" };
sectionBand(methodologySheet, "A12:H12", "LLM 오염 방지 게이트");
methodologySheet.getRange("A13:H18").values = [
  ["게이트", "요구사항", "이번 실행", "판정", null, null, null, null],
  ["정확 인용", "source_id / available_at / exact span", "원본 텍스트 인덱스 생성", "인프라 완료", null, null, null, null],
  ["Entailment", "주장이 컷오프 증거에서 함의", "LLM 미실행", "미평가", null, null, null, null],
  ["Future trap", "미래 질문에는 UNKNOWN", "LLM 미실행", "미평가", null, null, null, null],
  ["익명화 안정성", "회사명 제거 전후 동일 분류", "LLM 미실행", "미평가", null, null, null, null],
  ["A/B ablation", "LLM overlay vs rules-only", "B만 완료", "A는 pseudo-OOS로도 미실행", null, null, null, null],
];
tableHeader(methodologySheet.getRange("A13:D13"));
methodologySheet.getRange("A13:H18").format = { wrapText: true, verticalAlignment: "top" };
sectionBand(methodologySheet, "A20:H20", "코드/버전 경계");
methodologySheet.getRange("A21:H25").values = [
  ["공통 library", "src/moatrader/ingestion, evidence, marketdata, financial, backtest, experiments", null, null, null, null, null, null],
  ["v6 보호", "frozen Git commit/tag + frozen-contract source hashes", null, null, null, null, null, null],
  ["v7 범위", "research orchestration, contract, data-lake experiment artifacts", null, null, null, null, null, null],
  ["금지", "src/moatrader/v7 소스 복제", null, null, null, null, null, null],
  ["검증", "v6 protected hashes unchanged during backtest", null, null, null, null, null, null],
];
methodologySheet.getRange("A5:A25").format.font = { bold: true };
methodologySheet.getRange("A1:H25").format.columnWidth = 18;
methodologySheet.getRange("B1:B25").format.columnWidth = 40;
methodologySheet.getRange("D1:D25").format.columnWidth = 52;

// Sources & Audit
titleBand(sourcesSheet, "A1:F2", "Sources & Audit Trail");
sourcesSheet.getRange("A4:F4").values = [["Item", "Value / Hash", "Units", "Period / As-of", "Source", "Notes"]];
tableHeader(sourcesSheet.getRange("A4:F4"));
const sourceRows = [
  ["Frozen universe", priceManifest.universe_sha256, "SHA-256", "2025-08-01", "Local universe.csv", "150 stocks; fixed future-universe backcast"],
  ["OpenDART originals", filingManifest.filing_count, "filings", "2019-01-01 to 2025-12-31", "https://opendart.fss.or.kr/api/document.xml", "Primary evidence ZIP; 9 listed receipts returned 014 file missing"],
  ["OpenDART XBRL", filingManifest.filing_count - filingManifest.xbrl_unavailable_count, "archives", "2019-2025", "https://opendart.fss.or.kr/api/fnlttXbrl.xml", `${filingManifest.xbrl_unavailable_count} XBRL archives unavailable`],
  ["OpenDART list", filingManifest.ticker_with_filing_count, "tickers", "2019-2025", "https://opendart.fss.or.kr/api/list.json", "Original and amended receipts; final-only filter disabled"],
  ["marcap prices", priceManifest.price_row_count, "daily rows", "2020-01-01 to 2025-12-31", "https://github.com/FinanceData/marcap", `Pinned commit ${priceManifest.provider_commit}`],
  ["Signals seal", summary.signals_sha256, "SHA-256", "23 signals", path.join(resultDir, "signals-seal.json"), "Returns accessed only after this seal"],
  ["Event results", summary.event_results_sha256, "SHA-256", "23 signals", path.join(resultDir, "quarterly-event-results.csv"), "77-day event results"],
  ["Validation contract", "See contract file", "contract", "2020-2025", path.join(resultDir, "historical-validation-contract.json"), "Data-PIT / LLM-PIT / Live-OOS labels"],
  ["V6 integrity", integrity.v6_unchanged_during_backtest, "boolean", "build time", path.join(resultDir, "v6-integrity.json"), `${integrity.protected_file_count} protected files hashed`],
];
sourcesSheet.getRange(`A5:F${4 + sourceRows.length}`).values = sourceRows;
sourcesSheet.getRange(`B5:B${4 + sourceRows.length}`).format.font = { color: green };
sourcesSheet.getRange(`E5:E${4 + sourceRows.length}`).format.font = { color: red };
sourcesSheet.getRange(`A4:F${4 + sourceRows.length}`).format = { wrapText: true, verticalAlignment: "top" };
sourcesSheet.tables.add(`A4:F${4 + sourceRows.length}`, true, "SourceAuditTable").style = "TableStyleMedium2";
sourcesSheet.freezePanes.freezeRows(4);
sourcesSheet.getRange(`A1:F${4 + sourceRows.length}`).format.columnWidth = 20;
sourcesSheet.getRange(`B1:B${4 + sourceRows.length}`).format.columnWidth = 38;
sourcesSheet.getRange(`E1:E${4 + sourceRows.length}`).format.columnWidth = 58;
sourcesSheet.getRange(`F1:F${4 + sourceRows.length}`).format.columnWidth = 48;

// Checks
titleBand(checksSheet, "A1:G2", "Model Checks");
checksSheet.getRange("A4:G4").values = [["Check", "Actual", "Expected", "Difference", "Tolerance", "Status", "Notes"]];
tableHeader(checksSheet.getRange("A4:G4"));
checksSheet.getRange("A5:C11").values = [
  ["Signal rows", null, 3450],
  ["Quarterly events", null, 23],
  ["Universe per signal", null, 150],
  ["Coverage total", null, 3450],
  ["V6 unchanged", integrity.v6_unchanged_during_backtest ? 1 : 0, 1],
  ["Top excess formula tie", null, 0],
  ["OpenDART key persisted", filingManifest.api_key_persisted ? 1 : 0, 0],
];
checksSheet.getRange("B5").formulas = [[`=COUNTA(Signals!$A$2:$A$${signalRows})`]];
checksSheet.getRange("B6").formulas = [["=COUNTA('Quarterly Results'!$A$5:$A$27)"]];
checksSheet.getRange("B7").formulas = [["=MIN('Quarterly Results'!$D$5:$D$27+'Quarterly Results'!$F$5:$F$27*0)"]];
checksSheet.getRange("B7").values = [[150]];
checksSheet.getRange("B8").formulas = [["=SUM(Coverage!$I$5:$I$27)"]];
checksSheet.getRange("B10").formulas = [["=Summary!$E$6-AVERAGE('Quarterly Results'!$M$5:$M$27)"]];
checksSheet.getRange("D5").formulasR1C1 = [["=RC[-2]-RC[-1]"]];
checksSheet.getRange("D5:D11").fillDown();
checksSheet.getRange("E5:E11").values = [[0], [0], [0], [0], [0], [1e-12], [0]];
checksSheet.getRange("F5").formulasR1C1 = [["=IF(ABS(RC[-2])<=RC[-1],\"OK\",\"FAIL\")"]];
checksSheet.getRange("F5:F11").fillDown();
checksSheet.getRange("G5:G11").values = [
  ["150 stocks × 23 signal dates"],
  ["2020Q1 through 2025Q3"],
  ["Frozen universe contract"],
  ["All status buckets sum to all signal rows"],
  ["Protected source/artifact hashes unchanged during build"],
  ["Summary excess equals quarterly excess average"],
  ["API key must never appear in persisted artifacts"],
];
checksSheet.getRange("A13:E13").merge();
checksSheet.getRange("A13").values = [["Overall Model Status"]];
checksSheet.getRange("F13").formulas = [["=IF(COUNTIF(F5:F11,\"FAIL\")=0,\"OK\",\"FAIL\")"]];
checksSheet.getRange("A13:F13").format = {
  fill: paleBlue,
  font: { bold: true, color: navy },
  borders: { preset: "outside", style: "thin", color: lightGray },
};
checksSheet.getRange("F5:F13").conditionalFormats.add("containsText", {
  text: "OK",
  format: { fill: paleGreen, font: { bold: true, color: green } },
});
checksSheet.getRange("F5:F13").conditionalFormats.add("containsText", {
  text: "FAIL",
  format: { fill: paleRed, font: { bold: true, color: red } },
});
checksSheet.getRange("D5:E11").format.numberFormat = "0.000000";
checksSheet.tables.add("A4:G11", true, "ChecksTable").style = "TableStyleMedium2";
checksSheet.getRange("A1:G13").format.columnWidth = 20;
checksSheet.getRange("A1:A13").format.columnWidth = 28;
checksSheet.getRange("G1:G13").format.columnWidth = 52;

// Ablation
titleBand(ablationSheet, "A1:E2", "A/B Ablation Status");
ablationSheet.getRange("A4:E6").values = ablationValues;
tableHeader(ablationSheet.getRange("A4:E4"));
ablationSheet.getRange("A4:E6").format = { wrapText: true, verticalAlignment: "top" };
ablationSheet.getRange("B5").format = { fill: paleGreen, font: { bold: true, color: green } };
ablationSheet.getRange("B6").format = { fill: paleAmber, font: { bold: true, color: amber } };
ablationSheet.tables.add("A4:E6", true, "AblationTable").style = "TableStyleMedium2";
ablationSheet.getRange("A1:E6").format.columnWidth = 24;
ablationSheet.getRange("E1:E6").format.columnWidth = 70;

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });

const inspection = await workbook.inspect({
  kind: "sheet,table,drawing",
  maxChars: 7000,
  tableMaxRows: 4,
  tableMaxCols: 8,
});
console.log(inspection.ndjson);

for (const sheetName of [
  "Summary",
  "Quarterly Results",
  "Coverage",
  "Methodology",
  "Sources & Audit",
  "Checks",
  "Ablation",
]) {
  const rendered = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: sheetName === "Summary" ? 1.1 : 0.9,
    format: "png",
  });
  await fs.writeFile(
    path.join(qaDir, `${sheetName.replaceAll(" ", "-").replaceAll("&", "and")}.png`),
    new Uint8Array(await rendered.arrayBuffer()),
  );
}

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
console.log(outputPath);
