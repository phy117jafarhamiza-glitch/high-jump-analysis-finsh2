"""
المحلل الذكي للوثب العالي (AI Coach) — نظام متكامل لأطروحة:
تأثير تمرينات تصحيحية وفق أنموذج تعليمي مُعد باستخدام الشبكات العصبية الالتفافية (CNN)
في بعض القابليات البدنية والمتغيرات البايوميكانيكية والإنجاز للاعبي القفز العالي.

الصفحات: اللاعبون • التحليل الحركي • الاختبارات والإنجاز • المقارنة القبلية/البعدية
          • البرنامج التدريبي • الإحصاء والتصدير • الأنموذج التعليمي • الإعدادات
"""
import base64
import hashlib
import json
import os
import tempfile
from datetime import date

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import model
import pose_analysis as pa
import storage

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BANK_PATH = os.path.join(APP_DIR, "model_bank.xlsx")

st.set_page_config(page_title="المحلل الذكي للوثب العالي", page_icon="🧠", layout="wide")

if not hasattr(pa, "measure_vertical_jump") or not hasattr(model, "build_program"):
    st.error("⚠️ بعض ملفات التطبيق قديمة. ارفع كل الملفات الجديدة إلى GitHub ثم اضغط Manage app ← ⋮ ← Reboot app.")
    st.stop()

st.markdown(
    """
    <style>
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] {direction: rtl; text-align: right;}
    </style>
    """,
    unsafe_allow_html=True,
)

PHASES = ["قبلي", "بعدي"]
GROUPS = ["تجريبية", "ضابطة"]
EVENT_NAMES = {
    "approach": "نهاية الاقتراب", "plant": "وضع قدم الارتقاء", "takeoff": "ترك الأرض",
    "peak": "أعلى نقطة", "clearance": "اجتياز العارضة",
}
CLAUDE_MODELS = {
    "Claude Sonnet 5 (موصى به)": "claude-sonnet-5",
    "Claude Opus 5.5 (أدق وأغلى)": "claude-opus-5-5",
    "Claude Haiku 4.5 (أسرع وأرخص)": "claude-haiku-4-5-20251001",
}
SLOWMO_OPTS = [1, 2, 4, 8]


def slowmo_label(x):
    return "لا (سرعة عادية)" if x == 1 else f"نعم، أبطأ {x} مرات"


# ============================================================================
# الموارد المشتركة
# ============================================================================
def secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


@st.cache_resource(show_spinner="تحميل نموذج تقدير وضع الجسم (مرة واحدة فقط)...")
def get_model_path():
    return pa.download_model()


@st.cache_resource
def default_bank():
    return model.load_bank(BANK_PATH)


def get_bank():
    if st.session_state.get("bank_bytes"):
        return model.load_bank(st.session_state["bank_bytes"])
    if not os.path.exists(BANK_PATH):
        st.error("⚠️ ملف الأنموذج التعليمي **model_bank.xlsx** غير موجود في المستودع. "
                 "ارفعه إلى GitHub في نفس مكان app.py (بالاسم نفسه تماماً)، ثم اضغط Reboot app. "
                 "أو ارفعه هنا مؤقتاً لهذه الجلسة:")
        up = st.file_uploader("model_bank.xlsx", type=["xlsx"], key="bank_missing")
        if up is not None:
            try:
                model.load_bank(up.getvalue())
                st.session_state["bank_bytes"] = up.getvalue()
                st.rerun()
            except Exception as e:
                st.error(f"الملف غير صالح: {e}")
        st.stop()
    return default_bank()


def get_store():
    if "store" not in st.session_state:
        st.session_state["local_db"] = st.session_state.get("local_db", {})
        st.session_state["store"] = storage.Store(
            url=secret("SHEETS_URL"), token=secret("SHEETS_TOKEN"), local_db=st.session_state["local_db"])
    return st.session_state["store"]


def safe_read(store, sheet):
    try:
        return store.read(sheet)
    except storage.StoreError as e:
        st.error(f"تعذّرت قراءة البيانات من Google Sheets: {e}")
        return pd.DataFrame()


def players_df(store):
    df = safe_read(store, "players")
    if df.empty or "code" not in df:
        return pd.DataFrame(columns=["code", "name", "group", "gender", "height", "weight"])
    df["code"] = df["code"].astype(str)
    return df.drop_duplicates("code", keep="last")


def player_picker(store, key, groups=None, label="اللاعب"):
    df = players_df(store)
    if groups:
        df = df[df["group"].isin(groups)]
    if df.empty:
        st.warning("لا يوجد لاعبون بعد. أضفهم من صفحة «اللاعبون» أولاً.")
        return None
    opts = df["code"].tolist()
    names = dict(zip(df["code"], df["name"].astype(str)))
    grp = dict(zip(df["code"], df["group"].astype(str)))
    code = st.selectbox(label, opts, key=key, format_func=lambda c: f"{c} — {names.get(c, '')} ({grp.get(c, '')})")
    row = df[df["code"] == code].iloc[0].to_dict()
    row["height"] = float(row.get("height") or 180)
    row["weight"] = float(row.get("weight") or 70)
    return row


def save_video_temp(uploaded, prefix):
    """يحفظ الفيديو المرفوع مرة واحدة ويعيد مساره (يبقى صالحاً عبر إعادة التشغيل)."""
    data = uploaded.getvalue()
    key = prefix + hashlib.md5(data).hexdigest()
    paths = st.session_state.setdefault("_videos", {})
    if key not in paths or not os.path.exists(paths[key]):
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(data)
            paths[key] = f.name
    return paths[key], key


def extract(video_path, slowmo, label):
    bar = st.progress(0.0, text=label)
    landmarker = pa.create_landmarker(get_model_path())
    try:
        data = pa.extract_landmarks(video_path, landmarker, slowmo_factor=slowmo,
                                    progress=lambda x: bar.progress(x, text=f"{label} {int(x * 100)}%"))
    finally:
        landmarker.close()
    bar.empty()
    return data


def jpeg_b64(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return round(float(v), nd)


# ============================================================================
# الشريط الجانبي
# ============================================================================
store = get_store()
bank = get_bank()

with st.sidebar:
    st.markdown("## 🧠 AI Coach — الوثب العالي")
    page = st.radio("الصفحة", [
        "🏃 اللاعبون", "🎥 التحليل الحركي", "📏 الاختبارات والإنجاز", "📈 المقارنة القبلية/البعدية",
        "🗓️ البرنامج التدريبي", "📊 الإحصاء والتصدير", "📚 الأنموذج التعليمي", "⚙️ الإعدادات والمساعدة",
    ], label_visibility="collapsed")
    st.markdown("---")
    if store.remote:
        st.success("✅ الحفظ في Google Sheets")
        if st.button("🔄 تحديث البيانات"):
            store.clear_cache()
            st.rerun()
    else:
        st.warning("⚠️ وضع مؤقت: البيانات تضيع عند إغلاق الصفحة. اربط Google Sheets من صفحة الإعدادات، "
                   "أو حمّل نسخة احتياطية قبل الخروج.")
        st.download_button("⬇️ نسخة احتياطية (Excel)", store.export_excel(), "ai_coach_backup.xlsx")
        up = st.file_uploader("استعادة نسخة احتياطية", type=["xlsx"], key="restore")
        if up is not None and st.session_state.get("_restored") != up.name + str(up.size):
            store.import_excel(up.getvalue())
            st.session_state["_restored"] = up.name + str(up.size)
            st.rerun()
    if st.session_state.get("bank_bytes"):
        st.info("📚 يُستخدم ملف أنموذج مرفوع مؤقتاً.")


# ============================================================================
# 1) اللاعبون
# ============================================================================
def page_players():
    st.title("🏃 اللاعبون")
    with st.form("add_player", clear_on_submit=True):
        st.subheader("إضافة لاعب")
        c1, c2, c3 = st.columns(3)
        code = c1.text_input("الرمز (مثل E1 أو C1)").strip().upper()
        name = c2.text_input("الاسم")
        group = c3.selectbox("المجموعة", GROUPS)
        c4, c5, c6, c7 = st.columns(4)
        gender = c4.selectbox("الجنس", ["ذكر", "أنثى"])
        height = c5.number_input("الطول (سم)", 140, 240, 180)
        weight = c6.number_input("الوزن (كغم)", 35, 150, 70)
        birth = c7.number_input("سنة الميلاد", 1970, 2020, 2005)
        c8, c9 = st.columns(2)
        leg = c8.selectbox("رجل الارتقاء", ["اليسرى", "اليمنى"])
        best = c9.text_input("أفضل إنجاز سابق (م)", "")
        notes = st.text_input("ملاحظات", "")
        if st.form_submit_button("➕ حفظ اللاعب", type="primary"):
            existing = players_df(store)
            if not code:
                st.error("الرمز مطلوب.")
            elif code in existing["code"].tolist():
                st.error(f"الرمز {code} مستخدم. لتعديل بيانات لاعب احذفه ثم أضفه من جديد.")
            else:
                store.append("players", {"code": code, "name": name, "group": group, "gender": gender,
                                         "height": height, "weight": weight, "birth_year": birth,
                                         "takeoff_leg": leg, "best": best, "notes": notes,
                                         "created": storage.now_str()})
                st.success(f"تمت إضافة {code}.")

    df = players_df(store)
    st.subheader(f"قائمة اللاعبين ({len(df)})")
    if df.empty:
        st.info("لا يوجد لاعبون بعد.")
        return
    counts = df["group"].value_counts().to_dict()
    st.caption("  |  ".join(f"{g}: {n}" for g, n in counts.items()))
    show = df.rename(columns={"code": "الرمز", "name": "الاسم", "group": "المجموعة", "gender": "الجنس",
                              "height": "الطول", "weight": "الوزن", "birth_year": "الميلاد",
                              "takeoff_leg": "رجل الارتقاء", "best": "أفضل إنجاز", "notes": "ملاحظات"})
    st.dataframe(show.drop(columns=[c for c in ["created"] if c in show]), hide_index=True, width="stretch")

    with st.expander("🗑️ حذف لاعب"):
        code = st.selectbox("اللاعب", df["code"].tolist(), key="del_player")
        also = st.checkbox("احذف أيضاً كل تحليلاته واختباراته")
        if st.button("حذف نهائي", type="secondary"):
            store.delete("players", "code", code)
            if also:
                store.delete("analyses", "code", code)
                store.delete("tests", "code", code)
            st.success(f"حُذف {code}.")
            st.rerun()


# ============================================================================
# 2) التحليل الحركي
# ============================================================================
def claude_error_text(e):
    msg = str(e)
    low = msg.lower()
    if "credit balance" in low:
        return "💳 رصيد حساب Anthropic غير كافٍ. اشحن رصيداً من console.anthropic.com ← Billing."
    if "authentication" in low or "invalid x-api-key" in low or "401" in msg:
        return "🔑 مفتاح Claude غير صالح. تأكد من نسخه كاملاً (يبدأ بـ sk-ant-)."
    if "429" in msg or "rate_limit" in low:
        return "⏳ طلبات كثيرة خلال وقت قصير. انتظر دقيقة ثم أعد المحاولة."
    if "529" in msg or "overloaded" in low:
        return "⏳ خوادم Claude مشغولة حالياً. أعد المحاولة بعد قليل."
    if "not_found" in low or "404" in msg:
        return f"⚠️ النموذج المختار غير متاح لحسابك. جرّب نموذجاً آخر. التفاصيل: {msg}"
    return f"حدث خطأ أثناء الاتصال بـ Claude: {msg}"


def coach_prompt(player, res, errors_final):
    b = get_bank()
    ex = b["exercises"]
    einfo = model.error_info(b, errors_final)
    bank_text = []
    for _, e in einfo.iterrows():
        rows = ex[ex["رمز_الخطأ"] == e["رمز_الخطأ"]]
        lst = "\n".join(f"   - {r['التمرين']} ({r['النوع']}): {r['النقطة_التعليمية']}" for _, r in rows.iterrows())
        bank_text.append(f"- {e['رمز_الخطأ']} {e['الخطأ']}: {e['الوصف']} السبب المحتمل: {e['السبب_المحتمل']}\n{lst}")
    metrics_ar = {model.METRIC_LABELS[k][0] + (f" ({model.METRIC_LABELS[k][1]})" if model.METRIC_LABELS[k][1] else ""): v
                  for k, v in res["metrics"].items() if k in model.METRIC_LABELS}
    return f"""أنت مدرب وخبير بايوميكانيك في الوثب العالي (فوسبري فلوب). تكتب تقريراً للاعب ومدربه ضمن بحث علمي.

اللاعب: {player.get('name', '')} ({player.get('code', '')}) — {player.get('gender', '')}، الطول {player['height']} سم، الوزن {player['weight']} كغم، رجل الارتقاء المكتشفة {res['take_leg']}.

القياسات الآلية (MediaPipe Pose، كاميرا واحدة ثنائية الأبعاد؛ null = لم يُحسب):
{json.dumps(metrics_ar, ensure_ascii=False, indent=1)}
ملاحظات التصوير: {json.dumps(res.get('notes', []), ensure_ascii=False)}

الأخطاء المعتمدة في الأنموذج التعليمي لهذا اللاعب، مع التمارين التصحيحية المعتمدة لكل خطأ:
{chr(10).join(bank_text) if bank_text else "لا توجد أخطاء مسجلة."}

مرفق صور اللحظات المفتاحية مع الهيكل المرسوم.

قواعد صارمة:
- لا تقترح أي تمرين خارج القائمة أعلاه؛ الأنموذج التعليمي محكّم ومعتمد.
- لا تخترع قياسات. إن بدت صورة ما مخالفة للقياس فاذكر ذلك.

اكتب بالعربية الفصحى وبعناوين:
## 1. ملخص الأداء (3 جمل)
## 2. الاقتراب
## 3. الارتقاء
## 4. الطيران واجتياز العارضة
## 5. الأخطاء المعتمدة: شرح كل خطأ وأثره بلغة يفهمها اللاعب
## 6. كيف ينفّذ اللاعب التمارين المعتمدة (النقاط التعليمية والأخطاء الشائعة أثناء التنفيذ)
## 7. ملاحظات على جودة التصوير
"""


def page_analysis():
    st.title("🎥 التحليل الحركي")
    player = player_picker(store, "an_player")
    if player is None:
        return
    c1, c2, c3 = st.columns(3)
    phase = c1.selectbox("القياس", PHASES + ["تجربة (لا يُحفظ)"], key="an_phase")
    attempt = c2.number_input("رقم المحاولة", 1, 10, 1, key="an_attempt")
    slowmo = c3.selectbox("الحركة البطيئة", SLOWMO_OPTS, format_func=slowmo_label, key="an_slowmo",
                          help="120 إطاراً/ث = ×4، و240 إطاراً/ث = ×8. إن تركته على «لا» سيكتشفه النظام تقريبياً.")
    up = st.file_uploader("فيديو القفزة (كاميرا ثابتة من الجانب، يظهر فيها آخر 3 خطوات والارتقاء والطيران)",
                          type=["mp4", "mov", "avi", "m4v"], key="an_video")
    if up is not None:
        with st.expander("عرض الفيديو"):
            st.video(up)

    if st.button("🚀 بدء التحليل", type="primary", width="stretch", disabled=up is None):
        video_path, vkey = save_video_temp(up, "an")
        try:
            data = extract(video_path, slowmo, "📐 استخراج مفاصل الجسم...")
            res = pa.analyze(data, height_cm=player["height"], gender=player.get("gender", "ذكر"))
            frames = pa.key_frames(video_path, data["frame_idx"], res["points"], res["events"])
            auto = model.detect_errors(res["metrics"], player.get("gender", "ذكر"), get_bank())
            st.session_state["an"] = {
                "player": player, "phase": phase, "attempt": int(attempt), "slowmo": slowmo,
                "res": res, "frames": frames, "auto": auto, "video_path": video_path,
                "frame_idx": data["frame_idx"], "fps_eff": data["fps_eff"], "video_name": up.name,
                "saved": False, "report": None, "token": storage.new_id(),
            }
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"حدث خطأ غير متوقع أثناء التحليل: {e}")

    an = st.session_state.get("an")
    if not an:
        return
    if an["player"]["code"] != player["code"]:
        st.info("النتيجة المعروضة أدناه تخص لاعباً آخر. حلّل فيديو جديداً لهذا اللاعب.")
        return
    res, frames = an["res"], an["frames"]

    st.header("📐 القياسات البيوميكانيكية")
    st.caption(f"اللاعب {player['code']} | القياس: {an['phase']} | المحاولة {an['attempt']} | "
               f"رجل الارتقاء المكتشفة: {res['take_leg']} | نسبة الاكتشاف {res['detection_rate']}%")
    for n in res.get("notes", []):
        st.info(n)
    rows = [{"المؤشر": model.METRIC_LABELS[k][0], "القيمة": "غير متاح" if v is None else v,
             "الوحدة": model.METRIC_LABELS[k][1]} for k, v in res["metrics"].items() if k in model.METRIC_LABELS]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("🖼️ اللحظات المفتاحية")
    st.caption("تحقق أن كل صورة تطابق اسمها قبل الحفظ؛ إن لم تطابق فالقياسات غير دقيقة ولا تحفظها.")
    names = [n for n in EVENT_NAMES if n in frames]
    for c, n in zip(st.columns(len(names)) if names else [], names):
        c.image(frames[n], caption=f"{EVENT_NAMES[n]} ({res['event_times'][n]} ث)", width="stretch")

    s, et = res["series"], res["event_times"]
    cd = {"الزمن (ث)": s["t"], "زاوية ركبة الارتقاء (°)": s["knee_take"], "ميل الجذع عن العمودي (°)": s["trunk"]}
    if s.get("hip_height_cm") is not None:
        cd["ارتفاع الورك (سم)"] = s["hip_height_cm"]
    ch = pd.DataFrame(cd)
    ch = ch[(ch["الزمن (ث)"] >= et["plant"] - 1.5) & (ch["الزمن (ث)"] <= et["clearance"] + 0.5)].set_index("الزمن (ث)")
    g1, g2 = st.columns(2)
    g1.markdown(f"**زاوية ركبة الارتقاء** (الارتكاز {et['plant']}–{et['takeoff']} ث)")
    g1.line_chart(ch[["زاوية ركبة الارتقاء (°)"]])
    other = "ارتفاع الورك (سم)" if "ارتفاع الورك (سم)" in ch else "ميل الجذع عن العمودي (°)"
    g2.markdown(f"**{other}**")
    g2.line_chart(ch[[other]])

    # --- الأخطاء: آلية + بصرية يختارها الباحث ---
    st.subheader("⚠️ الأخطاء الفنية (وفق الأنموذج التعليمي)")
    b = get_bank()
    all_err = b["errors"]
    labels = dict(zip(all_err["رمز_الخطأ"], all_err["الخطأ"]))
    auto_ids = [e["id"] for e in an["auto"]]
    if an["auto"]:
        for e in an["auto"]:
            st.markdown(f"- **{e['name']}** ({e['id']}): القيمة {fmt(e['value'])} {e['op']} الحد {fmt(e['threshold'])}")
    else:
        st.success("لم يُكتشف خطأ آلي وفق حدود الأنموذج.")
    visual = all_err[all_err["طريقة_الكشف"] == "بصري"]
    final = st.multiselect(
        "الأخطاء المعتمدة لهذه المحاولة (عدّل بخبرتك: احذف الخطأ غير الصحيح وأضف الأخطاء البصرية)",
        list(labels), default=[i for i in auto_ids if i in labels], format_func=lambda c: f"{c} — {labels[c]}",
        key=f"an_final_{an.get('token', '')}")
    st.caption("الأخطاء البصرية المتاحة: " + "، ".join(f"{r['رمز_الخطأ']} {r['الخطأ']}" for _, r in visual.iterrows()))

    # --- الحفظ ---
    c1, c2 = st.columns([1, 2])
    if an["phase"] in PHASES:
        if c1.button("💾 حفظ هذه المحاولة", type="primary", disabled=an["saved"]):
            row = {"id": storage.new_id(), "date": storage.now_str(), "code": player["code"],
                   "group": player.get("group", ""), "phase": an["phase"], "attempt": an["attempt"],
                   **{k: v for k, v in res["metrics"].items()},
                   "errors_auto": ",".join(auto_ids), "errors_final": ",".join(final),
                   "slowmo": res.get("slowmo_applied", an["slowmo"]), "zoom_ratio": res["camera"]["zoom_ratio"],
                   "detection_rate": res["detection_rate"], "video": an["video_name"],
                   "events": json.dumps(res["event_times"], ensure_ascii=False)}
            try:
                store.append("analyses", row)
                an["saved"] = True
                st.success("تم الحفظ.")
            except storage.StoreError as e:
                st.error(str(e))
        if an["saved"]:
            c2.success("✅ محفوظة")
    else:
        c1.info("وضع التجربة: لا يُحفظ.")

    # --- التنزيلات ---
    st.subheader("⬇️ تنزيلات")
    d1, d2, d3 = st.columns(3)
    d1.download_button("القياسات (CSV)", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                       f"{player['code']}_metrics.csv", "text/csv")
    angles = pa.frame_angles(res["points"], res["series"]["t"], an["frame_idx"])
    d2.download_button("زوايا كل إطار (للمقارنة مع Kinovea)", angles.to_csv(index=False).encode("utf-8-sig"),
                       f"{player['code']}_angles.csv", "text/csv")
    if d3.button("🎬 إنشاء فيديو مُحلَّل"):
        out = os.path.join(tempfile.gettempdir(), f"{player['code']}_annotated.mp4")
        with st.spinner("جارٍ رسم الهيكل على الفيديو..."):
            ok = pa.annotated_video(an["video_path"], out, an["frame_idx"], res["points"], an["fps_eff"],
                                    events={EVENT_NAMES[k]: v for k, v in res["events"].items()})
        if ok:
            an["annotated"] = out
    if an.get("annotated") and os.path.exists(an["annotated"]):
        with open(an["annotated"], "rb") as f:
            d3.download_button("تحميل الفيديو المُحلَّل", f.read(), f"{player['code']}_annotated.mp4", "video/mp4")
        d3.caption("يُفتح ببرنامج VLC أو مشغّل Windows.")

    # --- تقرير Claude ---
    st.markdown("---")
    st.subheader("🏅 تقرير المدرب الذكي (Claude)")
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        key = st.text_input("مفتاح Claude API", type="password", key="claude_key")
    model_label = st.selectbox("النموذج", list(CLAUDE_MODELS), key="claude_model")
    st.caption("يشرح Claude الأداء والأخطاء المعتمدة وطريقة تنفيذ التمارين المعتمدة فقط، دون إضافة تمارين من عنده.")
    if st.button("✍️ اكتب التقرير", disabled=not key):
        import anthropic

        content = []
        for n in names:
            content.append({"type": "text", "text": f"الإطار: {EVENT_NAMES[n]}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": jpeg_b64(frames[n])}})
        content.append({"type": "text", "text": coach_prompt(player, res, final)})
        try:
            client = anthropic.Anthropic(api_key=key.strip())
            with client.messages.stream(model=CLAUDE_MODELS[model_label], max_tokens=8000,
                                        messages=[{"role": "user", "content": content}]) as stream:
                an["report"] = st.write_stream(stream.text_stream)
        except Exception as e:
            st.error(claude_error_text(e))
    elif an.get("report"):
        st.markdown(an["report"])
    if an.get("report"):
        st.download_button("⬇️ تحميل التقرير", str(an["report"]).encode("utf-8"),
                           f"{player['code']}_report.md", "text/markdown")


# ============================================================================
# 3) الاختبارات والإنجاز
# ============================================================================
def page_tests():
    st.title("📏 الاختبارات البدنية والإنجاز")
    player = player_picker(store, "t_player")
    if player is None:
        return
    b = get_bank()
    tmeta = b["tests"]
    c1, c2 = st.columns(2)
    phase = c1.selectbox("القياس", PHASES, key="t_phase")
    tdate = c2.date_input("تاريخ الاختبار", date.today(), key="t_date")

    tab1, tab2, tab3 = st.tabs(["✍️ إدخال النتائج", "🎥 الوثب العمودي بالفيديو", "📋 النتائج المسجلة"])
    with tab1:
        manual = tmeta[~tmeta["طريقة_القياس"].astype(str).str.contains("آلي")]
        grid = pd.DataFrame({"الرمز": manual["رمز_الاختبار"], "الاختبار": manual["الاختبار"],
                             "الوحدة": manual["الوحدة"], "النتيجة": [None] * len(manual)})
        edited = st.data_editor(grid, hide_index=True, width="stretch", key=f"grid_{player['code']}_{phase}",
                                disabled=["الرمز", "الاختبار", "الوحدة"],
                                column_config={"النتيجة": st.column_config.NumberColumn("النتيجة", format="%.2f")})
        if st.button("💾 حفظ النتائج", type="primary"):
            rows = [{"id": storage.new_id(), "date": str(tdate), "code": player["code"],
                     "group": player.get("group", ""), "phase": phase, "test_id": r["الرمز"],
                     "value": float(r["النتيجة"]), "method": "يدوي"}
                    for _, r in edited.iterrows() if pd.notna(r["النتيجة"])]
            if rows:
                store.append("tests", rows)
                st.success(f"حُفظت {len(rows)} نتيجة.")
            else:
                st.warning("لم تُدخل أي نتيجة.")

    with tab2:
        st.markdown("صوّر اللاعب **من الجانب بكاميرا ثابتة على حامل**، وتظهر القدمان كاملتين. "
                    "الأفضل 120 إطاراً/ث أو أكثر. الحساب: الارتفاع = g × (زمن الطيران)² ÷ 8.")
        st.caption("تنبيه منهجي: طريقة زمن الطيران تفترض الهبوط بنفس وضع الارتقاء (الرجلان ممدودتان)؛ "
                   "ثني الركبتين عند الهبوط يزيد النتيجة.")
        slowmo = st.selectbox("الحركة البطيئة", SLOWMO_OPTS, format_func=slowmo_label, key="vj_slowmo")
        up = st.file_uploader("فيديو الوثب العمودي", type=["mp4", "mov", "avi", "m4v"], key="vj_video")
        if st.button("📐 قياس الوثب", disabled=up is None):
            path, _ = save_video_temp(up, "vj")
            try:
                data = extract(path, slowmo, "📐 تتبع القدمين...")
                r = pa.measure_vertical_jump(data, player["height"])
                ev = {"takeoff": r["takeoff_idx"], "clearance": r["landing_idx"]}
                fr = pa.key_frames(path, data["frame_idx"], pa.clean_points(data["points"], data["real_fps"]), ev)
                st.session_state["vj"] = {"code": player["code"], "r": r, "frames": fr}
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"خطأ غير متوقع: {e}")
        vj = st.session_state.get("vj")
        if vj and vj["code"] == player["code"]:
            r = vj["r"]
            m1, m2 = st.columns(2)
            m1.metric("ارتفاع الوثب", f"{r['height_cm']} سم")
            m2.metric("زمن الطيران", f"{r['flight_s']} ث")
            for n in r["notes"]:
                st.warning(n)
            cc = st.columns(2)
            for c, (k, lbl) in zip(cc, [("takeoff", "لحظة ترك الأرض"), ("clearance", "لحظة الهبوط")]):
                if k in vj["frames"]:
                    c.image(vj["frames"][k], caption=lbl, width="stretch")
            st.caption("تحقق من الصورتين: الأولى يجب أن تكون آخر لحظة تلمس فيها القدم الأرض، والثانية أول لمسة.")
            if st.button("💾 حفظ النتيجة كاختبار T02"):
                store.append("tests", {"id": storage.new_id(), "date": str(tdate), "code": player["code"],
                                       "group": player.get("group", ""), "phase": phase, "test_id": "T02",
                                       "value": r["height_cm"], "method": "آلي (فيديو)"})
                st.success("تم الحفظ.")

    with tab3:
        t = safe_read(store, "tests")
        if t.empty or "code" not in t:
            st.info("لا توجد نتائج بعد.")
        else:
            t = t[t["code"].astype(str) == player["code"]].copy()
            names = dict(zip(tmeta["رمز_الاختبار"], tmeta["الاختبار"]))
            t["الاختبار"] = t["test_id"].map(names).fillna(t["test_id"])
            t = t.reindex(columns=["date", "phase", "الاختبار", "value", "method", "id"])
            st.dataframe(t.rename(
                columns={"date": "التاريخ", "phase": "القياس", "value": "النتيجة", "method": "الطريقة"}),
                hide_index=True, width="stretch")
            if not t.empty:
                del_id = st.selectbox("حذف نتيجة (بالمعرّف id)", t["id"].astype(str).tolist(), key="del_test")
                if st.button("حذف النتيجة"):
                    store.delete("tests", "id", del_id)
                    st.rerun()


# ============================================================================
# 4) المقارنة القبلية/البعدية للاعب
# ============================================================================
def page_compare():
    st.title("📈 المقارنة القبلية/البعدية")
    player = player_picker(store, "c_player")
    if player is None:
        return
    how = st.radio("عند تعدد المحاولات", ["mean", "max", "last"], horizontal=True,
                   format_func=lambda x: {"mean": "المتوسط", "max": "الأعلى", "last": "آخر محاولة"}[x])
    L = model.long_data(players_df(store), safe_read(store, "analyses"), safe_read(store, "tests"), get_bank(), how)
    L = L[L["code"] == player["code"]]
    if L.empty:
        st.info("لا توجد بيانات لهذا اللاعب بعد.")
        return
    piv = L.pivot_table(index=["kind", "label", "unit", "better"], columns="phase", values="value", aggfunc="first").reset_index()
    for p in PHASES:
        if p not in piv:
            piv[p] = np.nan
    piv["التغير"] = piv["بعدي"] - piv["قبلي"]
    piv["التحسن %"] = np.where(piv["قبلي"].abs() > 0, piv["التغير"] / piv["قبلي"].abs() * 100, np.nan)
    piv.loc[piv["better"] == "أقل", "التحسن %"] *= -1
    out = piv.rename(columns={"kind": "النوع", "label": "المتغير", "unit": "الوحدة"})[
        ["النوع", "المتغير", "الوحدة", "قبلي", "بعدي", "التغير", "التحسن %"]]
    st.dataframe(out.round(2), hide_index=True, width="stretch")

    a = safe_read(store, "analyses")
    if not a.empty and "errors_final" in a:
        a = a[a["code"].astype(str) == player["code"]]
        st.subheader("الأخطاء الفنية المعتمدة")
        labels = dict(zip(get_bank()["errors"]["رمز_الخطأ"], get_bank()["errors"]["الخطأ"]))
        for p in PHASES:
            ids = set()
            for v in a[a["phase"] == p]["errors_final"].fillna("").astype(str):
                ids |= {x for x in v.split(",") if x}
            st.markdown(f"**{p}:** " + ("، ".join(f"{labels.get(i, i)}" for i in sorted(ids)) or "لا توجد"))


# ============================================================================
# 5) البرنامج التدريبي
# ============================================================================
def page_program():
    st.title("🗓️ البرنامج التدريبي المخصص")
    st.caption("يُبنى البرنامج من بنك التمارين المعتمد في الأنموذج التعليمي، ويُخصَّص حسب أخطاء اللاعب وطوله ووزنه.")
    player = player_picker(store, "p_player")
    if player is None:
        return
    if player.get("group") == "ضابطة":
        st.warning("هذا اللاعب في المجموعة الضابطة؛ عادةً لا يطبّق البرنامج التجريبي.")
    b = get_bank()
    labels = dict(zip(b["errors"]["رمز_الخطأ"], b["errors"]["الخطأ"]))

    a = safe_read(store, "analyses")
    default = []
    if not a.empty and "errors_final" in a:
        pre = a[(a["code"].astype(str) == player["code"]) & (a["phase"] == "قبلي")]
        for v in pre["errors_final"].fillna("").astype(str):
            default += [x for x in v.split(",") if x and x in labels and x not in default]
    if not default:
        st.info("لا توجد أخطاء محفوظة من الاختبار القبلي لهذا اللاعب؛ اختر الأخطاء يدوياً.")
    errs = st.multiselect("الأخطاء المستهدفة (مرتبة آلياً حسب الأولوية)", list(labels), default=default,
                          format_func=lambda c: f"{c} — {labels[c]}")
    s = b["settings"]
    pers = model.personalization(player["height"], player["weight"], s)
    m = st.columns(4)
    m[0].metric("مؤشر كتلة الجسم", pers["bmi"])
    m[1].metric("معامل حجم القفزات والقوة", pers["factor"])
    m[2].metric("ارتفاع الحواجز", f"{pers['hurdle_cm']} سم")
    m[3].metric("ارتفاع الصندوق", f"{pers['box_cm']} سم")
    st.caption(f"البرنامج: {int(s['weeks'])} أسابيع × {int(s['units_per_week'])} وحدات، "
               f"ويعالج أهم {int(s['max_errors'])} أخطاء (يُغيّر من ورقة الإعدادات في ملف الأنموذج).")

    prog = model.build_program(player, errs, b)
    tbl = model.program_table(prog)
    for w in prog["weeks"]:
        with st.expander(f"الأسبوع {w['week']} — {w['phase']}", expanded=w["week"] == 1):
            wt = tbl[tbl["الأسبوع"] == w["week"]]
            for u in sorted(wt["الوحدة"].unique()):
                st.markdown(f"**الوحدة {u}**")
                st.dataframe(wt[wt["الوحدة"] == u][["القسم", "التمرين", "الجرعة", "الراحة", "النقطة التعليمية"]],
                             hide_index=True, width="stretch")
    researcher = st.text_input("اسم الباحث/المدرب (يظهر في الملف)", "", key="researcher")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ تحميل البرنامج (Word)", model.program_to_docx(prog, researcher),
                       f"program_{player['code']}.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary")
    if c2.button("💾 تسجيل البرنامج في قاعدة البيانات"):
        store.append("programs", {"id": storage.new_id(), "date": storage.now_str(), "code": player["code"],
                                  "errors": ",".join(t["رمز_الخطأ"] for t in prog["targets"]),
                                  "weeks": int(s["weeks"]), "units_per_week": int(s["units_per_week"]),
                                  "bmi": pers["bmi"], "factor": pers["factor"]})
        st.success("سُجّل البرنامج.")


# ============================================================================
# 6) الإحصاء والتصدير
# ============================================================================
def page_stats():
    st.title("📊 الإحصاء والتصدير")
    st.caption("مراجعة سريعة داخل التطبيق. للتحليل النهائي في الرسالة استخدم SPSS بعد التحقق من التوزيع الطبيعي.")
    how = st.radio("عند تعدد المحاولات", ["mean", "max", "last"], horizontal=True,
                   format_func=lambda x: {"mean": "المتوسط", "max": "الأعلى", "last": "آخر محاولة"}[x], key="s_how")
    L = model.long_data(players_df(store), safe_read(store, "analyses"), safe_read(store, "tests"), get_bank(), how)
    if L.empty:
        st.info("لا توجد بيانات بعد.")
        return
    wide = model.spss_wide(L)
    c1, c2 = st.columns(2)
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        wide.to_excel(w, sheet_name="SPSS", index=False)
        L.to_excel(w, sheet_name="long", index=False)
        key = L[["var", "label", "unit", "kind"]].drop_duplicates()
        key.to_excel(w, sheet_name="دليل المتغيرات", index=False)
    c1.download_button("⬇️ ملف SPSS (Excel)", buf.getvalue(), "spss_data.xlsx", type="primary")
    c2.caption("ورقة SPSS: صف لكل لاعب، وعمودان لكل متغير (pre و post). ورقة «دليل المتغيرات» تشرح الرموز.")

    kinds = ["بايوميكانيكي", "قابلية بدنية", "الإنجاز"]
    kind = st.radio("النوع", [k for k in kinds if k in set(L["kind"])], horizontal=True)
    vars_ = L[L["kind"] == kind][["var", "label"]].drop_duplicates()
    var = st.selectbox("المتغير", vars_["var"].tolist(), format_func=lambda v: vars_.set_index("var")["label"][v])
    table, between = model.group_stats(L, var)
    if table.empty:
        st.info("لا توجد بيانات كافية.")
        return
    st.subheader("داخل كل مجموعة (قبلي ← بعدي)")
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    st.caption("اختبار t للعينات المترابطة. حجم الأثر dz: 0.2 صغير، 0.5 متوسط، 0.8 كبير. "
               "نسبة التحسن موجبة = تحسن (مع مراعاة أن الزمن الأقل أفضل).")
    if between:
        st.subheader("بين المجموعتين")
        rows = []
        for ph, v in between.items():
            rows.append({"القياس": ph, "المقارنة": f"{v['groups'][0]} مقابل {v['groups'][1]}",
                         "قيمة t (Welch)": round(v["t"], 3), "الدلالة p": round(v["p"], 4), "حجم الأثر d": round(v["d"], 3),
                         "التفسير": ("تكافؤ المجموعتين" if v["p"] >= 0.05 else "فرق دال") if ph == "قبلي"
                         else ("فرق دال" if v["p"] < 0.05 else "فرق غير دال")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("القياس القبلي يُستخدم للتحقق من تكافؤ المجموعتين قبل التجربة.")
    means = L[L["var"] == var].groupby(["group", "phase"])["value"].mean().unstack("phase")
    means = means[[p for p in PHASES if p in means]]
    st.bar_chart(means)


# ============================================================================
# 7) الأنموذج التعليمي
# ============================================================================
def page_bank():
    st.title("📚 الأنموذج التعليمي")
    st.markdown("هذا البنك **مسودة** للعرض على الخبراء. عدّله في Excel ثم ارفعه هنا للتجربة، "
                "أو استبدل ملف `model_bank.xlsx` في GitHub ليصبح دائماً.")
    b = get_bank()
    default_bytes = b""
    if os.path.exists(BANK_PATH):
        with open(BANK_PATH, "rb") as f:
            default_bytes = f.read()
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ تحميل ملف الأنموذج الحالي", st.session_state.get("bank_bytes") or default_bytes,
                       "model_bank.xlsx")
    up = c2.file_uploader("رفع ملف أنموذج معدّل (للجلسة الحالية)", type=["xlsx"])
    if up is not None:
        try:
            model.load_bank(up.getvalue())
            st.session_state["bank_bytes"] = up.getvalue()
            st.success("تم تحميل الأنموذج المعدّل لهذه الجلسة.")
        except Exception as e:
            st.error(f"الملف غير صالح: {e}")
    if st.session_state.get("bank_bytes") and st.button("العودة إلى الأنموذج الأصلي"):
        st.session_state.pop("bank_bytes")
        st.rerun()
    t1, t2, t3, t4 = st.tabs(["الأخطاء", "التمارين", "الاختبارات", "الإعدادات"])
    t1.dataframe(b["errors"], hide_index=True, width="stretch")
    t2.dataframe(b["exercises"], hide_index=True, width="stretch")
    t3.dataframe(b["tests"], hide_index=True, width="stretch")
    t4.json(b["settings"])


# ============================================================================
# 8) الإعدادات والمساعدة
# ============================================================================
def page_settings():
    st.title("⚙️ الإعدادات والمساعدة")
    st.subheader("ربط Google Sheets")
    if store.remote:
        if st.button("اختبار الاتصال"):
            try:
                store.ping()
                st.success("الاتصال يعمل ✅")
            except storage.StoreError as e:
                st.error(str(e))
    else:
        st.markdown("""
1. أنشئ ملف **Google Sheets** جديداً (فارغاً).
2. من القائمة: **Extensions ← Apps Script** (الإضافات ← برمجة التطبيقات).
3. احذف الكود الموجود والصق محتوى ملف `apps_script.gs`، وغيّر كلمة السر في السطر `const TOKEN`.
4. اضغط **Deploy ← New deployment**، واختر النوع **Web app**:
   - Execute as: **Me**
   - Who has access: **Anyone**
5. وافق على الصلاحيات، ثم انسخ **Web app URL** (ينتهي بـ `/exec`).
6. في Streamlit: **Manage app ← ⋮ ← Settings ← Secrets** وأضف:
```
SHEETS_URL = "https://script.google.com/macros/s/.../exec"
SHEETS_TOKEN = "كلمة السر نفسها"
```
7. احفظ، ثم أعد تشغيل التطبيق (Reboot).
""")
    st.subheader("مفتاح Claude")
    st.markdown("أضف في Secrets السطر: `ANTHROPIC_API_KEY = \"sk-ant-...\"` (من console.anthropic.com).")
    st.subheader("نصائح التصوير")
    st.markdown("""
- هاتف **ثابت على حامل**، **بدون تقريب (Zoom)**، من الجانب على بعد 10–15 م.
- يظهر في الإطار آخر 3 خطوات والعارضة والمرتبة.
- التصوير بـ 120 أو 240 إطاراً/ث، وحدد عامل الحركة البطيئة في صفحة التحليل.
- إضاءة جيدة وملابس بلون مختلف عن الخلفية.
""")


PAGES = {
    "🏃 اللاعبون": page_players, "🎥 التحليل الحركي": page_analysis, "📏 الاختبارات والإنجاز": page_tests,
    "📈 المقارنة القبلية/البعدية": page_compare, "🗓️ البرنامج التدريبي": page_program,
    "📊 الإحصاء والتصدير": page_stats, "📚 الأنموذج التعليمي": page_bank, "⚙️ الإعدادات والمساعدة": page_settings,
}
try:
    PAGES[page]()
except storage.StoreError as e:
    st.error(f"مشكلة في حفظ/قراءة البيانات: {e}")
