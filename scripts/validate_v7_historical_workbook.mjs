import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { FileBlob, SpreadsheetFile } = require("@oai/artifact-tool");

const repo = process.cwd();
const workbookPath = path.join(
  repo,
  "outputs",
  "01a011a6-e595-7ae0-9a8e-5e426d90c5f1",
  "MoatRader_v7_Data-PIT_2020-2025.xlsx",
);
const summaryPath = path.join(
  repo,
  "data-lake",
  "experiments",
  "historical-validation-v7-2020-2025",
  "results-v7-quality-gated",
  "summary.json",
);

const expectedSheets = [
  "Summary",
  "Quarterly Results",
  "Coverage",
  "Signals",
  "Methodology",
  "Sources & Audit",
  "Checks",
  "Ablation",
];
const excelErrorPattern = /^#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!)/i;

function fail(message) {
  throw new Error(message);
}

function assertClose(actual, expected, label, tolerance = 1e-12) {
  if (typeof actual !== "number" || Math.abs(actual - expected) > tolerance) {
    fail(`${label}: expected ${expected}, received ${actual}`);
  }
}

const fileStat = await fs.stat(workbookPath);
if (fileStat.size <= 0) fail("Workbook is empty");

const summary = JSON.parse(await fs.readFile(summaryPath, "utf8"));
const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);

for (const sheetName of expectedSheets) {
  if (!workbook.worksheets.getItem(sheetName)) {
    fail(`Missing worksheet: ${sheetName}`);
  }
}

const summarySheet = workbook.worksheets.getItem("Summary");
const summaryValues = summarySheet.getRange("A5:F8").values;
const summaryFormulas = summarySheet.getRange("A5:F8").formulas;
assertClose(summaryValues[1][0], summary.top15_statistics.mean, "Top-15 mean return");
assertClose(
  summaryValues[1][4],
  summary.top15_excess_statistics.mean,
  "Top-15 excess mean",
);
assertClose(
  summaryValues[3][0],
  summary.bottom_spread_statistics.mean,
  "Top-minus-bottom mean",
);
assertClose(
  summaryValues[1][2],
  summary.universe_equal_weight_statistics.mean,
  "Universe mean return",
);
if (!String(summaryFormulas[1][4]).includes("=A6-C6")) {
  fail("Summary excess KPI is not formula-driven");
}

const checksSheet = workbook.worksheets.getItem("Checks");
const checkStatuses = [
  ...checksSheet.getRange("F5:F11").values.flat(),
  checksSheet.getRange("F13").values[0][0],
];
if (checkStatuses.some((status) => status !== "OK")) {
  fail(`Model checks failed: ${JSON.stringify(checkStatuses)}`);
}

const formulaErrors = [];
for (const sheetName of expectedSheets) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const usedRange = sheet.getUsedRange();
  if (!usedRange) continue;
  const values = usedRange.values;
  for (let rowIndex = 0; rowIndex < values.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < values[rowIndex].length; columnIndex += 1) {
      const value = values[rowIndex][columnIndex];
      if (typeof value === "string" && excelErrorPattern.test(value.trim())) {
        formulaErrors.push(`${sheetName}!R${rowIndex + 1}C${columnIndex + 1}=${value}`);
      }
    }
  }
}
if (formulaErrors.length > 0) {
  fail(`Excel errors detected: ${formulaErrors.slice(0, 20).join(", ")}`);
}

const report = {
  workbook: workbookPath,
  bytes: fileStat.size,
  sheets: expectedSheets,
  summary_kpis: {
    top15_mean: summaryValues[1][0],
    top15_excess_mean: summaryValues[1][4],
    top_minus_bottom_mean: summaryValues[3][0],
    universe_mean: summaryValues[1][2],
  },
  model_checks: checkStatuses,
  excel_formula_errors: formulaErrors.length,
  status: "PASS",
};

process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
