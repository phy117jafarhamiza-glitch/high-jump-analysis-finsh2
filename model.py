"""
الأنموذج التعليمي: قراءة بنك الأخطاء والتمارين، كشف الأخطاء، توليد البرنامج التدريبي
المخصص لكل لاعب، تصديره إلى Word، والإحصاء للمقارنة القبلية/البعدية بين المجموعات.
"""
import io
import math

import numpy as np
import pandas as pd

PHASES = ["تأسيس", "تطوير", "تثبيت"]
PHYS_TYPES = {"قوة", "بلايومتري", "سرعة", "مرونة"}
SCALED_TYPES = {"قوة", "بلايومتري"}

# أسماء المتغيرات البايوميكانيكية بالعربية
METRIC_LABELS = {
    "approach_speed_mps": ("سرعة الاقتراب النهائية", "م/ث"),
    "com_lowering_cm": ("خفض مركز الثقل قبل الارتقاء", "سم"),
    "knee_at_plant_deg": ("زاوية ركبة الارتقاء لحظة وضع القدم", "°"),
    "knee_min_deg": ("أقل زاوية لركبة الارتقاء أثناء الارتكاز", "°"),
    "knee_at_takeoff_deg": ("زاوية ركبة الارتقاء لحظة ترك الأرض", "°"),
    "contact_time_s": ("زمن الارتقاء", "ث"),
    "trunk_lean_plant_deg": ("ميل الجذع لحظة وضع القدم", "°"),
    "trunk_lean_takeoff_deg": ("ميل الجذع لحظة الارتقاء", "°"),
    "free_knee_lift_pct": ("ارتفاع ركبة الرجل الحرة عن الورك", "%"),
    "free_knee_angle_deg": ("زاوية ركبة الرجل الحرة لحظة الارتقاء", "°"),
    "arms_raised_count": ("عدد الذراعين المرفوعتين لحظة الارتقاء", ""),
    "takeoff_angle_deg": ("زاوية الانطلاق", "°"),
    "hip_rise_cm": ("ارتفاع الورك من الارتقاء إلى القمة", "سم"),
    "time_to_peak_s": ("الزمن من الارتقاء إلى القمة", "ث"),
    "hip_angle_at_peak_deg": ("زاوية الورك عند القمة", "°"),
    "knee_angle_at_peak_deg": ("زاوية الركبتين عند القمة", "°"),
}


# ============================================================================
# 1) بنك الأنموذج
# ============================================================================
def load_bank(path_or_bytes):
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    x = pd.read_excel(src, sheet_name=None)
    need = ["الأخطاء", "التمارين", "الاختبارات", "الإعدادات"]
    missing = [s for s in need if s not in x]
    if missing:
        raise ValueError("ملف الأنموذج ينقصه الأوراق: " + "، ".join(missing))
    errors = x["الأخطاء"].dropna(subset=["رمز_الخطأ"]).copy()
    errors["الأولوية"] = pd.to_numeric(errors["الأولوية"], errors="coerce").fillna(9)
    exercises = x["التمارين"].dropna(subset=["رمز_التمرين"]).copy()
    tests = x["الاختبارات"].dropna(subset=["رمز_الاختبار"]).copy()
    settings = {}
    for _, r in x["الإعدادات"].dropna(subset=["المفتاح"]).iterrows():
        try:
            settings[str(r["المفتاح"]).strip()] = float(r["القيمة"])
        except (TypeError, ValueError):
            settings[str(r["المفتاح"]).strip()] = r["القيمة"]
    defaults = dict(weeks=8, units_per_week=3, unit_minutes=90, foundation_weeks=3, development_weeks=3,
                    max_errors=3, tech_per_unit=2, phys_per_unit=2, bmi_mid=23, bmi_high=25,
                    factor_mid=0.9, factor_high=0.8, hurdle_ratio=0.25, box_ratio=0.2)
    for k, v in defaults.items():
        settings.setdefault(k, v)
    return {"errors": errors, "exercises": exercises, "tests": tests, "settings": settings}


# ============================================================================
# 2) كشف الأخطاء من القياسات
# ============================================================================
def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def detect_errors(metrics, gender, bank):
    """يعيد الأخطاء الآلية المتحققة مرتبة حسب الأولوية."""
    found = []
    for _, e in bank["errors"].iterrows():
        if str(e.get("طريقة_الكشف", "")).strip() != "آلي":
            continue
        var = str(e.get("المتغير", "")).strip()
        val = _num(metrics.get(var))
        thr = _num(e.get("الحد_إناث") if gender == "أنثى" else e.get("الحد_ذكور"))
        op = str(e.get("الشرط", "")).strip()
        if val is None or thr is None or op not in ("<", ">", "<=", ">="):
            continue
        hit = {"<": val < thr, ">": val > thr, "<=": val <= thr, ">=": val >= thr}[op]
        if hit:
            found.append({
                "id": e["رمز_الخطأ"], "name": e["الخطأ"], "phase": e.get("المرحلة", ""),
                "variable": var, "value": val, "op": op, "threshold": thr,
                "priority": float(e["الأولوية"]), "description": e.get("الوصف", ""),
                "cause": e.get("السبب_المحتمل", ""),
            })
    return sorted(found, key=lambda d: d["priority"])


def error_info(bank, ids):
    df = bank["errors"]
    rows = df[df["رمز_الخطأ"].isin(ids)].sort_values("الأولوية")
    return rows


# ============================================================================
# 3) البرنامج التدريبي المخصص
# ============================================================================
def personalization(height_cm, weight_kg, s):
    h = float(height_cm) / 100.0
    bmi = float(weight_kg) / (h * h)
    if bmi >= s["bmi_high"]:
        factor = s["factor_high"]
    elif bmi >= s["bmi_mid"]:
        factor = s["factor_mid"]
    else:
        factor = 1.0
    hurdle = int(round(float(height_cm) * s["hurdle_ratio"] / 5.0) * 5)
    box = int(round(float(height_cm) * s["box_ratio"] / 5.0) * 5)
    return {"bmi": round(bmi, 1), "factor": factor, "hurdle_cm": hurdle, "box_cm": box}


def phase_of_week(week, s):
    f, d = int(s["foundation_weeks"]), int(s["development_weeks"])
    if week <= f:
        return "تأسيس"
    if week <= f + d:
        return "تطوير"
    return "تثبيت"


def _dose(ex_row, phase, factor):
    sets = _num(ex_row.get(f"مجموعات_{phase}")) or 1
    reps = _num(ex_row.get(f"تكرار_{phase}")) or 1
    if ex_row.get("النوع") in SCALED_TYPES:
        reps = max(1, round(reps * factor))
    return int(sets), int(reps)


def _fill(text, pers):
    return str(text).replace("{hurdle}", str(pers["hurdle_cm"])).replace("{box}", str(pers["box_cm"]))


def build_program(player, error_ids, bank):
    """
    player: dict فيه code, name, height, weight, gender
    error_ids: رموز الأخطاء المستهدفة (آلية أو بصرية)
    """
    s = bank["settings"]
    ex = bank["exercises"]
    pers = personalization(player["height"], player["weight"], s)

    errs = error_info(bank, list(error_ids)).head(int(s["max_errors"]))
    targets = list(errs["رمز_الخطأ"])

    def pool(err, kinds):
        rows = ex[(ex["رمز_الخطأ"] == err) & (ex["النوع"].isin(kinds))]
        return [r for _, r in rows.iterrows()]

    tech = {e: pool(e, {"تكنيك"}) for e in targets}
    phys = {e: pool(e, PHYS_TYPES) for e in targets}
    gen_phys = pool("GEN", PHYS_TYPES)
    warm = pool("GEN", {"إحماء"})
    cool = pool("GEN", {"تهدئة"})
    full = [r for r in pool("GEN", {"تكنيك"})]

    weeks, upw = int(s["weeks"]), int(s["units_per_week"])
    t_per, p_per = int(s["tech_per_unit"]), int(s["phys_per_unit"])

    def item(section, r, phase):
        sets, reps = _dose(r, phase, pers["factor"])
        return {
            "القسم": section, "رمز": r["رمز_التمرين"], "الخطأ": r["رمز_الخطأ"],
            "التمرين": _fill(r["التمرين"], pers), "النوع": r["النوع"],
            "الجرعة": (f"{reps} {r.get('وحدة_التكرار', '')}" if sets == 1 else
                       f"{sets} × {reps} {r.get('وحدة_التكرار', '')}").strip(),
            "الراحة": f"{int(_num(r.get('الراحة_ث')) or 0)} ث" if _num(r.get("الراحة_ث")) else "—",
            "النقطة التعليمية": _fill(r.get("النقطة_التعليمية", "") if pd.notna(r.get("النقطة_التعليمية")) else "", pers),
        }

    def pick(pools_by_err, count, u, fallback):
        chosen, used = [], set()
        order = [e for e in targets if pools_by_err.get(e)]
        k = 0
        while order and len(chosen) < count and k < count * 4:
            e = order[(u + k) % len(order)]
            lst = pools_by_err[e]
            r = lst[(u // max(1, len(order)) + k) % len(lst)]
            if r["رمز_التمرين"] not in used:
                chosen.append(r); used.add(r["رمز_التمرين"])
            k += 1
        i = 0
        while len(chosen) < count and fallback and i < len(fallback) * 2:
            r = fallback[(u + i) % len(fallback)]
            if r["رمز_التمرين"] not in used:
                chosen.append(r); used.add(r["رمز_التمرين"])
            i += 1
        return chosen

    plan = []
    u = 0
    for w in range(1, weeks + 1):
        phase = phase_of_week(w, s)
        units = []
        for j in range(1, upw + 1):
            items = [item("الإحماء", r, phase) for r in warm[:1]]
            items += [item("الجزء الرئيسي: تكنيك تصحيحي", r, phase) for r in pick(tech, t_per, u, [])]
            if phase != "تأسيس" and j == upw and full:
                items.append(item("الجزء الرئيسي: الأداء الكامل", full[0], phase))
            items += [item("الجزء الرئيسي: إعداد بدني خاص", r, phase) for r in pick(phys, p_per, u, gen_phys)]
            items += [item("الختام", r, phase) for r in cool[:1]]
            units.append({"unit": j, "items": items})
            u += 1
        plan.append({"week": w, "phase": phase, "units": units})

    return {
        "player": player,
        "personalization": pers,
        "targets": errs[["رمز_الخطأ", "الخطأ", "المرحلة", "الوصف", "الأولوية"]].to_dict("records"),
        "weeks": plan,
        "settings": {k: s[k] for k in ("weeks", "units_per_week", "unit_minutes")},
    }


def program_table(program):
    rows = []
    for w in program["weeks"]:
        for u in w["units"]:
            for it in u["items"]:
                rows.append({"الأسبوع": w["week"], "المرحلة": w["phase"], "الوحدة": u["unit"], **it})
    return pd.DataFrame(rows)


# ============================================================================
# 4) تصدير البرنامج إلى Word (من اليمين لليسار)
# ============================================================================
def _rtl_paragraph(p):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    pPr = p._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    bidi.set(qn("w:val"), "1")
    pPr.append(bidi)
    for run in p.runs:
        rPr = run._r.get_or_add_rPr()
        rtl = OxmlElement("w:rtl")
        rtl.set(qn("w:val"), "1")
        rPr.append(rtl)
        fonts = rPr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts"); rPr.append(fonts)
        for a in ("w:ascii", "w:hAnsi", "w:cs"):
            fonts.set(qn(a), "Arial")


def _para(doc_or_cell, text, bold=False, size=None, align="right", style=None):
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    p = doc_or_cell.add_paragraph(style=style) if style else doc_or_cell.add_paragraph()
    run = p.add_run(str(text))
    run.bold = bold
    rPr = run._r.get_or_add_rPr()
    if bold:
        rPr.append(OxmlElement("w:bCs"))
    if size:
        run.font.size = Pt(size)
        szcs = OxmlElement("w:szCs"); szcs.set(qn("w:val"), str(int(size * 2))); rPr.append(szcs)
    p.alignment = {"right": WD_ALIGN_PARAGRAPH.RIGHT, "center": WD_ALIGN_PARAGRAPH.CENTER}[align]
    _rtl_paragraph(p)
    return p


def _table_rtl(table):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    tblPr = table._tbl.tblPr
    b = OxmlElement("w:bidiVisual")
    b.set(qn("w:val"), "1")
    tblPr.append(b)


def program_to_docx(program, researcher=""):
    from docx import Document
    from docx.shared import Pt, Cm
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    doc = Document()
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Cm(1.8)
        sectPr = sec._sectPr
        b = OxmlElement("w:bidi"); b.set(qn("w:val"), "1"); sectPr.append(b)
    st = doc.styles["Normal"]
    st.font.name = "Arial"
    st.font.size = Pt(11)

    pl, pers = program["player"], program["personalization"]
    _para(doc, "البرنامج التدريبي التصحيحي — الوثب العالي (فوسبري فلوب)", bold=True, size=16, align="center")
    _para(doc, f"اللاعب: {pl.get('name', '')}  |  الرمز: {pl.get('code', '')}  |  المجموعة: {pl.get('group', '')}")
    _para(doc, f"الطول: {pl['height']} سم  |  الوزن: {pl['weight']} كغم  |  مؤشر كتلة الجسم: {pers['bmi']}  |  "
               f"معامل حجم التمارين: {pers['factor']}")
    st_ = program["settings"]
    _para(doc, f"مدة البرنامج: {int(st_['weeks'])} أسابيع × {int(st_['units_per_week'])} وحدات أسبوعياً "
               f"(زمن الوحدة {int(st_['unit_minutes'])} دقيقة)  |  ارتفاع الحواجز ≈ {pers['hurdle_cm']} سم  |  "
               f"ارتفاع الصندوق ≈ {pers['box_cm']} سم")
    if researcher:
        _para(doc, f"إعداد: {researcher}")

    _para(doc, "الأخطاء الفنية المستهدفة (حسب الأولوية)", bold=True, size=13)
    for t in program["targets"]:
        _para(doc, f"• {t['الخطأ']} ({t['رمز_الخطأ']} — مرحلة {t['المرحلة']}): {t['الوصف']}")
    if not program["targets"]:
        _para(doc, "لم تُحدد أخطاء؛ البرنامج مبني على التمارين العامة.")

    cols = ["القسم", "التمرين", "الجرعة", "الراحة", "النقطة التعليمية"]
    widths = [Cm(3.2), Cm(6.0), Cm(2.6), Cm(1.6), Cm(4.8)]
    for w in program["weeks"]:
        _para(doc, f"الأسبوع {w['week']} — مرحلة {w['phase']}", bold=True, size=13)
        for u in w["units"]:
            _para(doc, f"الوحدة التدريبية {u['unit']}", bold=True)
            tbl = doc.add_table(rows=1, cols=len(cols))
            tbl.style = "Table Grid"
            _table_rtl(tbl)
            for i, c in enumerate(cols):
                cell = tbl.rows[0].cells[i]
                cell.paragraphs[0].text = ""
                cell.width = widths[i]
                r = cell.paragraphs[0].add_run(c); r.bold = True
                r._r.get_or_add_rPr().append(OxmlElement("w:bCs"))
                _rtl_paragraph(cell.paragraphs[0])
            for it in u["items"]:
                row = tbl.add_row().cells
                for i, c in enumerate(cols):
                    row[i].width = widths[i]
                    row[i].paragraphs[0].add_run(str(it.get(c, "")))
                    _rtl_paragraph(row[i].paragraphs[0])
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ============================================================================
# 5) تجميع البيانات والإحصاء
# ============================================================================
def long_data(players, analyses, tests, bank, how="mean"):
    """جدول طويل: الرمز، المجموعة، القياس (قبلي/بعدي)، المتغير، القيمة."""
    frames = []
    groups = players.set_index("code")["group"].to_dict() if not players.empty and "code" in players else {}
    if not analyses.empty:
        keep = [k for k in METRIC_LABELS if k in analyses.columns]
        a = analyses.melt(id_vars=["code", "phase"], value_vars=keep, var_name="var", value_name="value")
        a["value"] = pd.to_numeric(a["value"], errors="coerce")
        a["label"] = a["var"].map(lambda k: METRIC_LABELS[k][0])
        a["unit"] = a["var"].map(lambda k: METRIC_LABELS[k][1])
        a["kind"] = "بايوميكانيكي"
        a["better"] = ""
        frames.append(a)
    if not tests.empty:
        tmeta = bank["tests"].set_index("رمز_الاختبار")
        t = tests[["code", "phase", "test_id", "value"]].rename(columns={"test_id": "var"}).copy()
        t["value"] = pd.to_numeric(t["value"], errors="coerce")
        t["label"] = t["var"].map(lambda k: tmeta["الاختبار"].get(k, k))
        t["unit"] = t["var"].map(lambda k: tmeta["الوحدة"].get(k, ""))
        t["kind"] = t["var"].map(lambda k: "الإنجاز" if tmeta["القابلية"].get(k, "") == "الإنجاز" else "قابلية بدنية")
        t["better"] = t["var"].map(lambda k: tmeta["الأفضل"].get(k, ""))
        frames.append(t)
    if not frames:
        return pd.DataFrame(columns=["code", "group", "phase", "var", "label", "unit", "kind", "better", "value"])
    df = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    agg = {"mean": "mean", "max": "max", "min": "min", "last": "last"}[how]
    df = (df.groupby(["code", "phase", "var", "label", "unit", "kind", "better"], as_index=False)["value"].agg(agg))
    df["group"] = df["code"].map(groups).fillna("غير محددة")
    return df


def spss_wide(long_df):
    """صف لكل لاعب، وعمود لكل متغير × قياس (جاهز لـ SPSS)."""
    if long_df.empty:
        return pd.DataFrame()
    d = long_df.copy()
    d["col"] = d["var"] + "_" + d["phase"].map({"قبلي": "pre", "بعدي": "post"}).fillna(d["phase"])
    w = d.pivot_table(index=["code", "group"], columns="col", values="value", aggfunc="first")
    order = []
    for v in dict.fromkeys(d["var"]):
        order += [c for c in (f"{v}_pre", f"{v}_post") if c in w.columns]
    order += [c for c in w.columns if c not in order]
    w = w[order].reset_index()
    w.columns.name = None
    return w


def _desc(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, 0
    return x.mean(), (x.std(ddof=1) if len(x) > 1 else np.nan), len(x)


def group_stats(long_df, var):
    """الإحصاء الوصفي، واختبار t للعينات المترابطة (قبلي/بعدي)، وحجم الأثر، ونسبة التحسن."""
    from scipy import stats

    d = long_df[long_df["var"] == var]
    if d.empty:
        return pd.DataFrame(), {}
    better = d["better"].iloc[0]
    rows = []
    for g, gd in d.groupby("group"):
        piv = gd.pivot_table(index="code", columns="phase", values="value", aggfunc="first")
        pre = piv["قبلي"] if "قبلي" in piv else pd.Series(dtype=float)
        post = piv["بعدي"] if "بعدي" in piv else pd.Series(dtype=float)
        m1, s1, n1 = _desc(pre)
        m2, s2, n2 = _desc(post)
        both = piv.dropna(subset=[c for c in ("قبلي", "بعدي") if c in piv]) if {"قبلي", "بعدي"} <= set(piv) else piv.iloc[0:0]
        t = p = dz = np.nan
        if len(both) >= 2:
            diff = both["بعدي"] - both["قبلي"]
            if diff.std(ddof=1) > 0:
                t, p = stats.ttest_rel(both["بعدي"], both["قبلي"])
                dz = diff.mean() / diff.std(ddof=1)
        imp = np.nan
        if m1 and not np.isnan(m1) and not np.isnan(m2) and m1 != 0:
            imp = (m2 - m1) / abs(m1) * 100 * (-1 if better == "أقل" else 1)
        rows.append({
            "المجموعة": g, "ن": int(max(n1, n2)),
            "قبلي: الوسط": m1, "قبلي: الانحراف": s1, "بعدي: الوسط": m2, "بعدي: الانحراف": s2,
            "قيمة t (مترابطة)": t, "الدلالة p": p, "حجم الأثر dz": dz, "نسبة التحسن %": imp,
        })
    table = pd.DataFrame(rows)

    between = {}
    groups = sorted(d["group"].unique())
    if len(groups) == 2:
        for phase in ("قبلي", "بعدي"):
            a = d[(d["group"] == groups[0]) & (d["phase"] == phase)]["value"].dropna()
            b = d[(d["group"] == groups[1]) & (d["phase"] == phase)]["value"].dropna()
            if len(a) >= 2 and len(b) >= 2 and (a.std() > 0 or b.std() > 0):
                t, p = stats.ttest_ind(a, b, equal_var=False)
                sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
                dd = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
                between[phase] = {"t": t, "p": p, "d": dd, "groups": groups}
    return table, between
