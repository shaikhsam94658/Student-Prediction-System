import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 760px; padding-top: 1.4rem;}
.hero {background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); color: #fff;
       padding: 22px 26px; border-radius: 18px; margin-bottom: 18px;}
.hero h1 {color: #fff; font-size: 1.7rem; margin: 0 0 4px 0; padding: 0;}
.hero p {color: #E0E7FF; margin: 0; font-size: 0.95rem;}
.stepper {display: flex; margin: 8px 0 22px 0;}
.step {flex: 1; text-align: center; position: relative;}
.step:not(:last-child)::after {content: ""; position: absolute; top: 15px; left: 50%;
       width: 100%; height: 3px; background: #CBD5E1; z-index: 0;}
.step.done:not(:last-child)::after {background: #4F46E5;}
.dot {width: 32px; height: 32px; border-radius: 50%; background: #CBD5E1; color: #fff;
      display: inline-flex; align-items: center; justify-content: center; font-weight: 700;
      position: relative; z-index: 1; font-size: 0.9rem;}
.step.done .dot {background: #4F46E5;}
.step.active .dot {background: #4F46E5; box-shadow: 0 0 0 5px rgba(79,70,229,0.22);}
.lbl {font-size: 0.78rem; margin-top: 6px; opacity: 0.7;}
.step.active .lbl {opacity: 1; font-weight: 700;}
.sect-title {font-size: 1.25rem; font-weight: 700; margin: 0 0 2px 0;}
.sect-sub {opacity: 0.7; margin-bottom: 12px; font-size: 0.92rem;}
.result {color: #fff; border-radius: 20px; padding: 26px; text-align: center; margin: 14px 0;}
.result .rname {font-size: 1rem; opacity: 0.92;}
.result .rscore {font-size: 3.6rem; font-weight: 800; line-height: 1.1;}
.result .rscore span {font-size: 1.3rem; font-weight: 500; opacity: 0.85;}
.result .rbadge {display: inline-block; background: rgba(255,255,255,0.22); padding: 4px 16px;
                 border-radius: 999px; font-weight: 600; margin-top: 8px;}
.stButton > button {border-radius: 12px; font-weight: 600; padding: 0.55rem 1.2rem;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ load model
try:
    model = joblib.load(ROOT / "models" / "student_model.joblib")
    with open(ROOT / "models" / "metadata.json") as f:
        meta = json.load(f)
except FileNotFoundError:
    st.error("Model files not found. Run the notebook or `python student_performance.py` first "
             "so that the `models` folder contains `student_model.joblib` and `metadata.json`.")
    st.stop()

# ------------------------------------------------------------------ config
STEP_NAMES = ["Profile", "Study", "Family", "School", "Review"]
LEVELS = ["School", "College / Degree"]
STAGES = {0: ["Class 6-8", "Class 9-10", "Class 11-12"],
          1: ["Diploma", "Bachelor's degree", "Master's degree"]}

GROUPS = [
    ("Study Habits", "📚", "How the student studies and spends the week",
     ["Hours_Studied", "Attendance", "Previous_Scores", "Tutoring_Sessions",
      "Sleep_Hours", "Physical_Activity", "Extracurricular_Activities"]),
    ("Home & Family", "🏠", "Support and situation at home",
     ["Parental_Involvement", "Parental_Education_Level", "Family_Income",
      "Distance_from_Home", "Peer_Influence", "Learning_Disabilities"]),
    ("School & Resources", "🏫", "Learning environment and resources",
     ["School_Type", "Teacher_Quality", "Access_to_Resources", "Internet_Access", "Motivation_Level"]),
]
# any model column not placed above goes to the last group automatically
placed = {c for g in GROUPS for c in g[3]} | {"Gender"}
extra = [c for c in meta["columns"] if c not in placed]
if extra:
    GROUPS[-1] = (*GROUPS[-1][:3], GROUPS[-1][3] + extra)

LABELS = {
    "Hours_Studied": "Study hours", "Attendance": "Attendance (%)", "Previous_Scores": "Previous exam score",
    "Tutoring_Sessions": "Tutoring sessions", "Sleep_Hours": "Sleep hours", "Physical_Activity": "Physical activity",
    "Extracurricular_Activities": "Extracurricular activities", "Parental_Involvement": "Parental involvement",
    "Parental_Education_Level": "Parents' education", "Family_Income": "Family income",
    "Distance_from_Home": "Distance from home to school", "Peer_Influence": "Peer influence",
    "Learning_Disabilities": "Learning disability", "School_Type": "School type", "Teacher_Quality": "Teacher quality",
    "Access_to_Resources": "Access to resources", "Internet_Access": "Internet access",
    "Motivation_Level": "Motivation level", "Gender": "Gender",
}
HELP = {
    "Hours_Studied": "How many hours the student studies (as defined in the dataset).",
    "Attendance": "Percentage of classes attended.",
    "Previous_Scores": "Marks in the previous exam.",
    "Tutoring_Sessions": "Number of extra tutoring sessions.",
    "Access_to_Resources": "Books, library, study material and similar resources.",
}
ORDERS = [["Low", "Medium", "High"], ["Near", "Moderate", "Far"], ["Negative", "Neutral", "Positive"],
          ["High School", "College", "Postgraduate"], ["No", "Yes"], ["Public", "Private"], ["Male", "Female"]]
DEFAULTS = {"Learning_Disabilities": "No", "Internet_Access": "Yes", "Extracurricular_Activities": "Yes",
            "School_Type": "Public", "Peer_Influence": "Neutral", "Parental_Education_Level": "College",
            "Gender": "Male"}


def fkey(col):
    return f"f_{col}"


def order_opts(opts):
    for o in ORDERS:
        if set(opts) == set(o):
            return o
    return sorted(opts)


def default_for(col):
    if col in meta["numeric"]:
        v = meta["numeric"][col]["median"]
        return int(v) if float(v).is_integer() else v
    opts = order_opts(meta["categorical"][col])
    return DEFAULTS.get(col) if DEFAULTS.get(col) in opts else opts[len(opts) // 2]


def init_state():
    ss = st.session_state
    ss.setdefault("step", 0)
    ss.setdefault("result", None)
    ss.setdefault("p_name", "")
    ss.setdefault("p_level", LEVELS[0])
    ss.setdefault("p_stage_0", STAGES[0][1])
    ss.setdefault("p_stage_1", STAGES[1][1])
    for c in meta["columns"]:
        ss.setdefault(fkey(c), default_for(c))


PERSIST = ["p_name", "p_level", "p_stage_0", "p_stage_1"] + [fkey(c) for c in meta["columns"]]
init_state()
for k in PERSIST:                      # keep values alive while their widgets are on another step
    st.session_state[k] = st.session_state[k]


# ------------------------------------------------------------------ helpers
def go(delta):
    st.session_state.step += delta
    if delta < 0:
        st.session_state.result = None


def jump(step):
    st.session_state.step = step
    st.session_state.result = None


def restart():
    for k in PERSIST + ["result"]:
        st.session_state.pop(k, None)
    st.session_state.step = 0
    init_state()


def predict():
    row = pd.DataFrame([{c: st.session_state[fkey(c)] for c in meta["columns"]}])[meta["columns"]]
    st.session_state.result = max(0.0, min(100.0, float(model.predict(row)[0])))


def label_of(col):
    return LABELS.get(col, col.replace("_", " "))


def field(col):
    label, help_ = label_of(col), HELP.get(col)
    if col in meta["numeric"]:
        m = meta["numeric"][col]
        lo, hi = m["min"], m["max"]
        if float(lo).is_integer() and float(hi).is_integer():
            lo, hi = int(lo), int(hi)
        st.slider(label, lo, hi, key=fkey(col), help=help_)
    else:
        opts = order_opts(meta["categorical"][col])
        if len(opts) <= 3 and max(len(o) for o in opts) <= 8:
            st.radio(label, opts, key=fkey(col), horizontal=True, help=help_)
        else:
            st.selectbox(label, opts, key=fkey(col), help=help_)


def section_header(icon, title, sub):
    st.markdown(f'<div class="sect-title">{icon} {title}</div><div class="sect-sub">{sub}</div>',
                unsafe_allow_html=True)


def stepper(current):
    items = []
    for i, name in enumerate(STEP_NAMES):
        cls = "done" if i < current else ("active" if i == current else "")
        dot = "✓" if i < current else str(i + 1)
        items.append(f'<div class="step {cls}"><div class="dot">{dot}</div><div class="lbl">{name}</div></div>')
    st.markdown(f'<div class="stepper">{"".join(items)}</div>', unsafe_allow_html=True)


def tips():
    ss, out = st.session_state, []
    med = lambda c: meta["numeric"][c]["median"]
    if "Attendance" in meta["numeric"] and ss[fkey("Attendance")] < 80:
        out.append("Raise attendance towards 80% or more. It is the strongest factor in the training data.")
    if "Hours_Studied" in meta["numeric"] and ss[fkey("Hours_Studied")] < med("Hours_Studied"):
        out.append("Add a few more study hours per week. Study time is the second strongest factor.")
    if "Tutoring_Sessions" in meta["numeric"] and ss[fkey("Tutoring_Sessions")] == 0:
        out.append("Consider some tutoring or doubt-clearing sessions.")
    if ss.get(fkey("Parental_Involvement")) == "Low":
        out.append("Share progress and plans with parents or guardians regularly.")
    if ss.get(fkey("Access_to_Resources")) == "Low":
        out.append("Use library books, free online lessons and teacher notes to make up for fewer resources.")
    return out[:4]


# ------------------------------------------------------------------ page
st.markdown('<div class="hero"><h1>🎓 Student Performance Predictor</h1>'
            '<p>Answer a few quick steps and get an estimated exam score with simple suggestions.</p></div>',
            unsafe_allow_html=True)
step = st.session_state.step
stepper(step)

with st.container(border=True):
    if step == 0:
        section_header("👤", "About the student", "Basic details. Name and level are only used on this screen.")
        st.text_input("Student name", key="p_name", placeholder="e.g. Aarav Patil")
        st.radio("Level", LEVELS, key="p_level", horizontal=True)
        lv = LEVELS.index(st.session_state.p_level)
        st.selectbox("Class / Degree", STAGES[lv], key=f"p_stage_{lv}")
        if "Gender" in meta["categorical"]:
            st.radio("Gender", order_opts(meta["categorical"]["Gender"]), key=fkey("Gender"), horizontal=True)
        if lv == 1:
            st.info("The model learned from school-level exam data, so for college students the result is only a rough guide.")
    elif 1 <= step <= len(GROUPS):
        title, icon, sub, cols_g = GROUPS[step - 1]
        section_header(icon, title, sub)
        cols_in_step = [c for c in cols_g if c in meta["columns"]]
        left, right = st.columns(2)
        for i, c in enumerate(cols_in_step):
            with (left if i % 2 == 0 else right):
                field(c)
    else:
        section_header("✅", "Review and predict", "Check the details. Use Edit to change anything.")
        ss = st.session_state
        name = ss.p_name.strip() or "Student"
        lv = LEVELS.index(ss.p_level)
        st.markdown(f"**{name}** · {ss.p_level} · {ss[f'p_stage_{lv}']}" +
                    (f" · {ss[fkey('Gender')]}" if "Gender" in meta["categorical"] else ""))
        for gi, (title, icon, _, cols_g) in enumerate(GROUPS):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"**{icon} {title}**")
                st.caption(" · ".join(f"{label_of(c)}: {ss[fkey(c)]}" for c in cols_g if c in meta["columns"]))
            with c2:
                st.button("Edit", key=f"edit_{gi}", on_click=jump, args=(gi + 1,))
        st.button("🔮 Predict score", key="btn_predict", type="primary", on_click=predict,
                  use_container_width=True)

# ------------------------------------------------------------------ result
if step == len(STEP_NAMES) - 1 and st.session_state.result is not None:
    score = st.session_state.result
    thr = meta["at_risk_threshold"]
    if score < thr:
        badge, c1, c2 = "Needs support", "#F97316", "#DC2626"
    elif score < 70:
        badge, c1, c2 = "On track", "#0EA5E9", "#2563EB"
    else:
        badge, c1, c2 = "Strong", "#10B981", "#047857"
    nm = st.session_state.p_name.strip() or "Student"
    st.markdown(f'<div class="result" style="background: linear-gradient(135deg, {c1}, {c2});">'
                f'<div class="rname">{nm}\'s predicted exam score</div>'
                f'<div class="rscore">{score:.1f}<span> / 100</span></div>'
                f'<div class="rbadge">{badge}</div></div>', unsafe_allow_html=True)
    st.progress(min(max(score / 100, 0.0), 1.0))
    st.caption(f"Support threshold used by this app: {thr}. In the training data most students score between 65 and 69.")
    t = tips()
    if t:
        st.markdown("**Suggestions**")
        for x in t:
            st.markdown(f"- {x}")
        st.caption("Suggestions come from patterns in the dataset. They are not guarantees.")
    elif score >= thr:
        st.success("Current habits look good. Keep them up.")
    st.caption("This is a statistical estimate and should support, not replace, a teacher's judgement.")

# ------------------------------------------------------------------ navigation
st.write("")
nav_l, nav_r = st.columns(2)
with nav_l:
    if step > 0:
        st.button("← Back", key="btn_back", on_click=go, args=(-1,), use_container_width=True)
with nav_r:
    if step < len(STEP_NAMES) - 1:
        need_name = step == 0 and not st.session_state.p_name.strip()
        st.button("Next →", key="btn_next", type="primary", on_click=go, args=(1,),
                  disabled=need_name, use_container_width=True)
    else:
        st.button("↺ Start over", key="btn_restart", on_click=restart, use_container_width=True)
if step == 0 and not st.session_state.p_name.strip():
    st.caption("Enter the student's name to continue.")