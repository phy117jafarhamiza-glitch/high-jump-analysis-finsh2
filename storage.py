"""
حفظ البيانات في Google Sheets عبر Apps Script، مع بديل مؤقت داخل الجلسة إذا لم يُعدّ الربط.

الأوراق المستخدمة:
  players   : بيانات اللاعبين (رمز، اسم، مجموعة، طول، وزن...)
  analyses  : نتائج التحليل الحركي لكل محاولة (قبلي/بعدي)
  tests     : نتائج الاختبارات البدنية والإنجاز
  programs  : سجل البرامج التدريبية المولّدة
"""
import io
import json
import time
import uuid
from datetime import datetime

import pandas as pd
import requests

SHEETS = ["players", "analyses", "tests", "programs"]


def new_id():
    return uuid.uuid4().hex[:10]


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


class StoreError(RuntimeError):
    pass


class Store:
    """
    url/token: رابط تطبيق Apps Script وكلمة السر. إذا كانا فارغين يُستخدم local_db
    (قاموس داخل جلسة Streamlit) كبديل مؤقت.
    """

    def __init__(self, url="", token="", local_db=None, timeout=30):
        self.url = (url or "").strip()
        self.token = (token or "").strip()
        self.timeout = timeout
        self.local = local_db if local_db is not None else {}
        for s in SHEETS:
            self.local.setdefault(s, [])
        self._cache = {}

    @property
    def remote(self):
        return bool(self.url and self.token)

    # ---------------- الاتصال ----------------
    def _call(self, payload):
        payload = dict(payload, token=self.token)
        last = None
        for attempt in range(3):
            try:
                r = requests.post(self.url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                  headers={"Content-Type": "text/plain;charset=utf-8"},
                                  timeout=self.timeout, allow_redirects=True)
                if r.status_code >= 500:
                    last = StoreError(f"خطأ من خادم Google ({r.status_code})")
                    time.sleep(1.5 * (attempt + 1))
                    continue
                try:
                    data = r.json()
                except ValueError:
                    raise StoreError(
                        "رد غير متوقع من Google Sheets. تأكد أن النشر بصلاحية Anyone وأن الرابط ينتهي بـ /exec."
                    )
                if not data.get("ok"):
                    raise StoreError(data.get("error", "خطأ غير معروف من Google Sheets"))
                return data
            except requests.RequestException as e:
                last = StoreError(f"تعذّر الاتصال بـ Google Sheets: {e}")
                time.sleep(1.5 * (attempt + 1))
        raise last

    def ping(self):
        if not self.remote:
            return False
        self._call({"action": "ping"})
        return True

    # ---------------- العمليات ----------------
    def read(self, sheet):
        if sheet in self._cache:
            return self._cache[sheet].copy()
        if self.remote:
            rows = self._call({"action": "read", "sheet": sheet}).get("rows", [])
        else:
            rows = list(self.local.get(sheet, []))
        df = pd.DataFrame(rows)
        self._cache[sheet] = df
        return df.copy()

    def append(self, sheet, rows):
        if isinstance(rows, dict):
            rows = [rows]
        clean = [{k: _jsonable(v) for k, v in r.items()} for r in rows]
        if self.remote:
            self._call({"action": "append", "sheet": sheet, "rows": clean})
        else:
            self.local.setdefault(sheet, []).extend(clean)
        self._cache.pop(sheet, None)

    def delete(self, sheet, key, value):
        if self.remote:
            n = self._call({"action": "delete", "sheet": sheet, "key": key, "value": value}).get("deleted", 0)
        else:
            before = len(self.local.get(sheet, []))
            self.local[sheet] = [r for r in self.local.get(sheet, []) if str(r.get(key)) != str(value)]
            n = before - len(self.local[sheet])
        self._cache.pop(sheet, None)
        return n

    def clear_cache(self):
        self._cache = {}

    # ---------------- نسخ احتياطي (للوضع المؤقت أو للأرشفة) ----------------
    def export_excel(self):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            for s in SHEETS:
                df = self.read(s)
                (df if not df.empty else pd.DataFrame({"فارغ": []})).to_excel(w, sheet_name=s, index=False)
        return buf.getvalue()

    def import_excel(self, file_bytes):
        """يستورد نسخة احتياطية إلى الوضع المؤقت فقط (لتجنب تكرار البيانات في Google Sheets)."""
        if self.remote:
            raise StoreError("الاستيراد متاح في الوضع المؤقت فقط.")
        x = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
        for s in SHEETS:
            if s in x and "فارغ" not in x[s].columns:
                self.local[s] = x[s].where(pd.notna(x[s]), None).to_dict("records")
        self.clear_cache()


def _jsonable(v):
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN
        return ""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v
