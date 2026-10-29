import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

async function loadArtifactTool() {
  try {
    return await import("@oai/artifact-tool");
  } catch (originalError) {
    const candidates = [
      process.env.CODEX_WORKSPACE_NODE_MODULES,
      path.join(
        os.homedir(),
        ".cache",
        "codex-runtimes",
        "codex-primary-runtime",
        "dependencies",
        "node",
        "node_modules",
      ),
    ].filter(Boolean);
    for (const nodeModules of candidates) {
      try {
        const require = createRequire(path.join(nodeModules, "artifact-loader.cjs"));
        const resolved = require.resolve("@oai/artifact-tool");
        return await import(pathToFileURL(resolved).href);
      } catch {
        // Try the next supported dependency location.
      }
    }
    throw originalError;
  }
}

const { FileBlob, SpreadsheetFile, Workbook } = await loadArtifactTool();

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, "../../..");
const resultsDir = path.join(projectRoot, "results", "new_eval_2026_results");
const sourcePath = process.argv[2] ?? path.join(resultsDir, "source", "mapped_surface_points.csv");
const outputPath =
  process.argv[3] ??
  path.join(resultsDir, "eni_to_ithaca_energy", "highlighted_facilities.xlsx");
const manifestPath =
  process.argv[4] ?? path.join(resultsDir, "source", "mapped_surface_points_manifest.json");

function columnLetter(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function widthForHeader(header) {
  if (!header) return 3;
  if (header === "feature_id" || header.endsWith("_object_id")) return 36;
  if (header.includes("infrastructure_name")) return 26;
  if (header.includes("reporting_group_name")) return 24;
  if (header.includes("description") || header.includes("comments")) return 34;
  if (header.includes("pipeline_system_name")) return 28;
  if (header.includes("crs_name") || header.includes("reason_for_")) return 28;
  if (header.includes("date") || header.includes("longitude") || header.includes("latitude")) {
    return 18;
  }
  if (header.startsWith("changed_")) return 14;
  return 16;
}

const csvText = (await fs.readFile(sourcePath, "utf8")).replace(/^\uFEFF/, "");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "facility_mapping" });
const sheet = workbook.worksheets.getItemAt(0);
const usedRange = sheet.getUsedRange();
const rowCount = usedRange.values.length;
const columnCount = usedRange.values[0].length;

if (rowCount < 2 || columnCount < 2) {
  throw new Error(`Mapping table is unexpectedly small: ${rowCount} x ${columnCount}`);
}

const sourceManifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
function latestLabel(year) {
  const candidates = sourceManifest.sources.filter((source) => source.period.startsWith(`${year}-`));
  if (candidates.length === 0) throw new Error(`No ${year} NSTA snapshot in ${manifestPath}`);
  candidates.sort((left, right) => left.period.localeCompare(right.period));
  return candidates.at(-1).label;
}
const previousLabel = latestLabel(2025);
const currentLabel = latestLabel(2026);

sheet.showGridLines = false;
sheet.tabColor = "#1F4E78";
sheet.freezePanes.freezeRows(1);

const lastColumn = columnLetter(columnCount - 1);
const headerRange = sheet.getRange(`A1:${lastColumn}1`);
headerRange.format.fill = "#1F4E78";
headerRange.format.font = { bold: true, color: "#FFFFFF" };
headerRange.format.wrapText = true;
headerRange.format.verticalAlignment = "center";
headerRange.format.rowHeight = 36;

const bodyRange = sheet.getRange(`A2:${lastColumn}${rowCount}`);
bodyRange.format.font = { name: "Aptos", size: 10, color: "#222222" };
bodyRange.format.verticalAlignment = "top";

const headers = headerRange.values[0];
for (let index = 0; index < headers.length; index += 1) {
  const letter = columnLetter(index);
  sheet.getRange(`${letter}1:${letter}${rowCount}`).format.columnWidth = widthForHeader(
    String(headers[index] ?? ""),
  );
}

const oldOwnerIndex = headers.indexOf(`${previousLabel}_reporting_group_name`);
const newOwnerIndex = headers.indexOf(`${currentLabel}_reporting_group_name`);
if (oldOwnerIndex < 0 || newOwnerIndex < 0) {
  throw new Error("Mapping is missing comparison ownership columns");
}
const highlightedRows = [];
for (let rowIndex = 1; rowIndex < rowCount; rowIndex += 1) {
  const values = usedRange.values[rowIndex];
  if (
    String(values[oldOwnerIndex] ?? "").trim().toUpperCase() === "ENI UK LIMITED" &&
    String(values[newOwnerIndex] ?? "").trim().toUpperCase() === "ITHACA ENERGY"
  ) {
    const excelRow = rowIndex + 1;
    const range = sheet.getRange(`A${excelRow}:${lastColumn}${excelRow}`);
    range.format.fill = "#F4B183";
    range.format.font = { bold: true, color: "#222222" };
    highlightedRows.push(`A${excelRow}`);
  }
}
if (highlightedRows.length === 0) {
  throw new Error(`No ENI to Ithaca transfer rows found between ${previousLabel} and ${currentLabel}`);
}

workbook.recalculate();

const errorCheck = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "formula error scan",
});
if (!errorCheck.ndjson.includes("matched 0 entries")) {
  throw new Error(`Workbook error-token check failed:\n${errorCheck.ndjson}`);
}

// Exercise the renderer before export as a layout smoke test.
await workbook.render({
  sheetName: sheet.name,
  range: `A1:${columnLetter(Math.min(columnCount - 1, 16))}${Math.min(rowCount, 25)}`,
  scale: 1,
  format: "png",
});

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

// Re-open the saved workbook and assert that all dynamic transfer rows remain highlighted.
const savedWorkbook = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const savedSheet = savedWorkbook.worksheets.getItemAt(0);
const savedHeader = savedSheet.getRange("A1").values[0][0];
if (savedHeader !== "feature_id") {
  throw new Error(`Unexpected first header after export: ${JSON.stringify(savedHeader)}`);
}

const styles = await savedWorkbook.inspect({
  kind: "computedStyle",
  sheetId: savedSheet.name,
  range: `A1:A${rowCount}`,
  maxChars: Math.max(16000, rowCount * 200),
});
const savedHighlightedRows = styles.ndjson
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line))
  .filter((record) => record.kind === "computedStyle")
  .filter((record) => record.style?.fill?.color?.value === "F4B183")
  .map((record) => record.for);
if (JSON.stringify(savedHighlightedRows) !== JSON.stringify(highlightedRows)) {
  throw new Error(`Unexpected highlighted rows: ${JSON.stringify(savedHighlightedRows)}`);
}

// artifact-tool may persist large inspect output beside an imported workbook;
// it is a QA scratch file rather than a case deliverable.
await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });

console.log(
  JSON.stringify({
    outputPath,
    dimensions: `${rowCount}x${columnCount}`,
    highlightedRows: savedHighlightedRows,
    formulaErrors: 0,
  }),
);
