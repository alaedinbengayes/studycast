"""StudyCast – Exam Score Estimator (CSCI104 Option B).

Loads the model trained in notebook/Exam_Score_Model.ipynb and predicts an exam
score from five study-profile inputs.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("model.joblib")
TEST_MAE = 1.11  # from the held-out test set in the notebook
DATASET_URL = "https://www.kaggle.com/datasets/lainguyn123/student-performance-factors"
FAIL_MSG = "We couldn't calculate an estimate. Please try again."

# field key -> label, help text, accepted range, training range, whole-number?
FIELDS = {
    "Hours_Studied": dict(label="Study hours per week", unit="hours/week", lo=0, hi=60,
                          train=(1, 44), whole=False, placeholder="e.g. 20"),
    "Attendance": dict(label="Class attendance (%)", unit="%", lo=0, hi=100,
                       train=(60, 100), whole=False, placeholder="e.g. 85"),
    "Previous_Scores": dict(label="Previous exam score", unit="points", lo=0, hi=100,
                            train=(50, 100), whole=False, placeholder="e.g. 75"),
    "Tutoring_Sessions": dict(label="Tutoring per month", unit="sessions", lo=0, hi=20,
                              train=(0, 8), whole=True, placeholder="e.g. 2"),
}
RESOURCE_OPTIONS = ["Low", "Medium", "High"]

st.set_page_config(page_title="StudyCast · Exam Score Estimator", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    return joblib.load(MODEL_PATH)


def inject_css(invalid_keys):
    highlight = "".join(
        f""".st-key-field_{k} input, .st-key-field_{k} div[data-baseweb="select"] > div {{
                border: 2px solid var(--sc-error) !important; background: #fff5f5 !important; }}
            .st-key-field_{k} label p {{ color: var(--sc-error) !important; font-weight: 600; }}"""
        for k in invalid_keys)
    st.markdown(f"""
<style>
  :root {{ --sc-navy:#12355b; --sc-teal:#0f8b8d; --sc-error:#c62828; --sc-muted:#5b6b7c; }}
  .block-container {{ padding-top: 4.5rem; max-width: 1100px; }}
  .sc-header {{ display:flex; align-items:center; gap:14px; padding:14px 18px; border-radius:14px;
               background:linear-gradient(120deg,var(--sc-navy),var(--sc-teal)); color:#fff; margin-bottom:1rem; }}
  .sc-header h1 {{ font-size:1.6rem; margin:0; color:#fff; padding:0; line-height:1.2; }}
  .sc-header p {{ margin:2px 0 0; opacity:.9; font-size:.95rem; }}
  .sc-result {{ border:2px solid var(--sc-teal); border-radius:14px; padding:20px; background:#f2fbfb; }}
  .sc-result h3 {{ margin:0 0 6px; color:var(--sc-navy); font-size:1.15rem; }}
  .sc-score {{ font-size:3.4rem; font-weight:800; color:var(--sc-navy); line-height:1.05; }}
  .sc-score span {{ font-size:1.2rem; font-weight:500; color:var(--sc-muted); }}
  .sc-empty {{ border:2px dashed #c9d4df; border-radius:14px; padding:20px; color:var(--sc-muted); }}
  .sc-empty h3 {{ margin:0 0 6px; color:var(--sc-navy); font-size:1.15rem; }}
  .sc-fielderr {{ color:var(--sc-error); font-size:.85rem; margin:-6px 0 8px; }}
  div[data-testid="stFormSubmitButton"] button {{ background:var(--sc-teal); color:#fff; border:0;
               font-weight:700; padding:.6rem 1.2rem; width:100%; }}
  div[data-testid="stFormSubmitButton"] button:hover {{ background:var(--sc-navy); color:#fff; }}
  {highlight}
</style>""", unsafe_allow_html=True)


def validate(values):
    """Return {field: message} for every missing or invalid input."""
    errors = {}
    for key, f in FIELDS.items():
        v = values[key]
        if v is None:
            errors[key] = f"This field is required. Enter a number from {f['lo']} to {f['hi']}."
        elif not f["lo"] <= v <= f["hi"]:
            errors[key] = f"{v:g} is out of range. Enter a number from {f['lo']} to {f['hi']}."
        elif f["whole"] and float(v) != int(v):
            errors[key] = "Enter a whole number of sessions (e.g. 0, 1, 2)."
    if values["Access_to_Resources"] not in RESOURCE_OPTIONS:
        errors["Access_to_Resources"] = "Choose Low, Medium or High."
    return errors


def predict(values):
    if st.query_params.get("test") == "model-failure":  # used only for the failure test in the report
        raise RuntimeError("Simulated model failure")
    row = pd.DataFrame([{k: values[k] for k in list(FIELDS) + ["Access_to_Resources"]}])
    score = float(load_model().predict(row)[0])
    return max(0.0, min(100.0, score))


# ---------- page ----------
st.session_state.setdefault("result", None)
st.session_state.setdefault("errors", {})
st.session_state.setdefault("failed", False)

st.markdown("""
<div class="sc-header">
  <svg width="46" height="46" viewBox="0 0 48 48" aria-hidden="true">
    <circle cx="24" cy="24" r="23" fill="#ffffff22" stroke="#fff" stroke-width="2"/>
    <path d="M8 20 24 12l16 8-16 8z" fill="#fff"/><path d="M15 24v7c0 2 4.5 4 9 4s9-2 9-4v-7l-9 4.5z" fill="#fff"/>
    <path d="M38 21v9" stroke="#fff" stroke-width="2"/><circle cx="38" cy="31" r="2" fill="#fff"/>
  </svg>
  <div><h1>StudyCast · Exam Score Estimator</h1>
  <p>University of Khorfakkan · CSCI104 Intro to AI project</p></div>
</div>""", unsafe_allow_html=True)

tab_predict, tab_how, tab_about = st.tabs(["🎯 Predict", "📊 How it works", "ℹ️ About"])

with tab_predict:
    st.write("Enter your study profile to estimate your next exam score. "
             "Use realistic or fictional values; nothing is saved.")
    col_form, col_result = st.columns([3, 2], gap="large")

    with col_form:
        with st.form("prediction", clear_on_submit=False, border=True):
            st.markdown("**Your study profile**")
            values, err_slots = {}, {}
            c1, c2 = st.columns(2)
            for i, (key, f) in enumerate(FIELDS.items()):
                with (c1 if i % 2 == 0 else c2):
                    with st.container(key=f"field_{key}"):
                        values[key] = st.number_input(
                            f["label"], value=None,
                            step=1 if f["whole"] else 1.0, format="%d" if f["whole"] else "%.1f",
                            min_value=None, max_value=None,
                            placeholder=f["placeholder"],
                            help=f"Accepted range: {f['lo']}–{f['hi']} {f['unit']}.")
                        st.caption(f"Range {f['lo']}–{f['hi']} {f['unit']}")
                    err_slots[key] = st.empty()
            with st.container(key="field_Access_to_Resources"):
                values["Access_to_Resources"] = st.selectbox(
                    "Access to study resources (books, internet, quiet space)",
                    RESOURCE_OPTIONS, index=None, placeholder="Choose Low, Medium or High")
            err_slots["Access_to_Resources"] = st.empty()
            submitted = st.form_submit_button("Predict Exam Score")

    if submitted:
        errors = validate(values)
        st.session_state.errors = errors
        st.session_state.result = None  # never show an old result for a new submission
        st.session_state.failed = False
        if not errors:
            try:
                score = predict(values)
                warn = [FIELDS[k]["label"] for k, f in FIELDS.items()
                        if not f["train"][0] <= values[k] <= f["train"][1]]
                st.session_state.result = dict(score=score, warn=warn)
            except Exception:
                st.session_state.failed = True

    errors = st.session_state.errors
    inject_css(errors.keys())
    for key, msg in errors.items():
        err_slots[key].markdown(f'<div class="sc-fielderr">⚠ {msg}</div>', unsafe_allow_html=True)

    with col_result:
        result = st.session_state.result
        if errors:
            st.error(f"Please fix the {len(errors)} highlighted field{'s' if len(errors) > 1 else ''}. "
                     "Your other values have been kept.")
        if st.session_state.failed:
            st.error(FAIL_MSG)
        if result:
            s = result["score"]
            st.markdown(f"""
<div class="sc-result" role="status" aria-live="polite">
  <h3>Estimated Exam Score</h3>
  <div class="sc-score">{s:.1f} <span>/ 100 points</span></div>
  <p style="margin:.6rem 0 0">Likely range: <b>{max(0, s - 2):.0f}–{min(100, s + 2):.0f}</b> points.
  The model's typical error on unseen students is ±{TEST_MAE:.1f} points.</p>
</div>""", unsafe_allow_html=True)
            if result["warn"]:
                st.warning("Some inputs are outside the range the model learned from ("
                           + ", ".join(result["warn"]) + "), so treat this estimate with extra caution.")
        elif not errors and not st.session_state.failed:
            st.markdown("""
<div class="sc-empty"><h3>Estimated Exam Score</h3>
Fill in all five fields and press <b>Predict Exam Score</b>. Your estimate will appear here.</div>""",
                        unsafe_allow_html=True)

with tab_how:
    st.subheader("How the estimate is made")
    st.markdown(f"""
1. **Your inputs** go into a scikit-learn pipeline saved as `model.joblib`.
2. The pipeline **fills any blank** with the training median, **one-hot encodes** *Access to resources*,
   and applies a **linear regression** model.
3. The model was trained on 80% of {6606:,} student records and tested on the other 20%.

| Metric (held-out test set) | Value |
|---|---|
| Mean absolute error (MAE) | **{TEST_MAE} points** |
| MAE if we always guessed the average | 2.76 points |
| R² | 0.71 |
| Predictions within ±2 points | 87% |

**Biggest effects:** each extra percentage point of attendance and each extra weekly study hour raise the estimate the most.

**Limitation:** a few students score far above what these five inputs suggest, and the data only covers
60–100% attendance and 1–44 study hours, so estimates outside those ranges are less reliable.
""")

with tab_about:
    st.subheader("About this project")
    st.markdown(f"""
* **Purpose:** help students see how study habits relate to exam results. It's a planning aid, not an official grade.
* **Dataset:** [Student Performance Factors – Kaggle]({DATASET_URL})
* **Tools:** Python, pandas, scikit-learn, Google Colab, Streamlit.
* **Course:** CSCI104 Introduction to AI, University of Khorfakkan.
""")
    st.caption("Class project. Use fictional inputs.")
