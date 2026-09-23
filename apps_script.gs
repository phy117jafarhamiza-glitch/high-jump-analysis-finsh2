/**
 * خادم حفظ البيانات لتطبيق المحلل الذكي للوثب العالي
 * يُلصق في: Google Sheets ← الإضافات (Extensions) ← Apps Script
 * ثم: Deploy ← New deployment ← Web app ← Execute as: Me ← Who has access: Anyone
 *
 * غيّر كلمة السر أدناه، واكتب نفسها في Secrets داخل Streamlit (SHEETS_TOKEN).
 */
const TOKEN = "ضع-كلمة-سر-طويلة-هنا";

function doPost(e) {
  try {
    const req = JSON.parse(e.postData.contents);
    if (req.token !== TOKEN) return out({ ok: false, error: "كلمة السر غير صحيحة" });
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const lock = LockService.getScriptLock();
    lock.waitLock(20000);
    try {
      if (req.action === "ping") return out({ ok: true });
      if (req.action === "read") return out({ ok: true, rows: readSheet(ss, req.sheet) });
      if (req.action === "append") { appendRows(ss, req.sheet, req.rows || []); return out({ ok: true }); }
      if (req.action === "delete") return out({ ok: true, deleted: deleteRows(ss, req.sheet, req.key, req.value) });
      return out({ ok: false, error: "إجراء غير معروف" });
    } finally {
      lock.releaseLock();
    }
  } catch (err) {
    return out({ ok: false, error: String(err) });
  }
}

function getSheet(ss, name) {
  return ss.getSheetByName(name) || ss.insertSheet(name);
}

function headers(sh) {
  if (sh.getLastColumn() === 0) return [];
  return sh.getRange(1, 1, 1, sh.getLastColumn()).getValues()[0].map(String);
}

function appendRows(ss, name, rows) {
  if (!rows.length) return;
  const sh = getSheet(ss, name);
  let head = headers(sh);
  // أضف أي عمود جديد غير موجود
  rows.forEach(r => Object.keys(r).forEach(k => { if (head.indexOf(k) < 0) head.push(k); }));
  sh.getRange(1, 1, 1, head.length).setValues([head]).setFontWeight("bold");
  const values = rows.map(r => head.map(k => (r[k] === undefined || r[k] === null) ? "" : r[k]));
  sh.getRange(sh.getLastRow() + 1, 1, values.length, head.length).setValues(values);
}

function readSheet(ss, name) {
  const sh = ss.getSheetByName(name);
  if (!sh || sh.getLastRow() < 2) return [];
  const data = sh.getRange(1, 1, sh.getLastRow(), sh.getLastColumn()).getValues();
  const head = data[0].map(String);
  return data.slice(1).map(row => {
    const o = {};
    head.forEach((k, i) => { o[k] = row[i] instanceof Date ? row[i].toISOString() : row[i]; });
    return o;
  });
}

function deleteRows(ss, name, key, value) {
  const sh = ss.getSheetByName(name);
  if (!sh || sh.getLastRow() < 2) return 0;
  const head = headers(sh);
  const col = head.indexOf(key);
  if (col < 0) return 0;
  const vals = sh.getRange(2, col + 1, sh.getLastRow() - 1, 1).getValues();
  let n = 0;
  for (let i = vals.length - 1; i >= 0; i--) {
    if (String(vals[i][0]) === String(value)) { sh.deleteRow(i + 2); n++; }
  }
  return n;
}

function out(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
