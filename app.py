"""
Credit Card Fraud Detector - Streamlit app

Run from this folder:   streamlit run app.py
Needs the files created by train_model.py inside ./models
"""

import html
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Fraud Detector", page_icon="🛡️", layout="wide")


# ============================================================
# STYLE
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

.stApp { background: #F3F5F9; }
.block-container { padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1180px; }
[data-testid="stSidebar"] [data-testid="stAlert"] { background: #1A2B4A; border: 1px solid #2E4168; border-radius: 10px; }
[data-testid="stSidebar"] [data-testid="stAlert"] * { color: #A9B6CA !important; }
footer { visibility: hidden; }

.stMarkdown, .stMarkdown *, [data-testid="stWidgetLabel"] p, button p, h1, h2, h3 {
    font-family: 'IBM Plex Sans', sans-serif;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E1E6EE !important;
    border-radius: 16px;
    box-shadow: 0 1px 2px rgba(18,32,58,0.05), 0 10px 28px rgba(18,32,58,0.06);
}

[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] .stDownloadButton button {
    width: 100%; min-height: 48px; border-radius: 10px; font-weight: 700;
}
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background: #F2B84B; border: 0; color: #12203A;
}
[data-testid="stSidebar"] button[kind="primary"] p,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p { color: #12203A !important; }
.stDownloadButton button { border: 1px solid #12203A; border-radius: 10px; font-weight: 600; }

.mono { font-family: 'IBM Plex Mono', monospace; }
.eyebrow { font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #4A5568; }
.muted { color: #4A5568; font-size: 14px; }
.title { font-size: 34px !important; font-weight: 700 !important; letter-spacing: -0.015em; line-height: 1.15 !important; color: #12203A; margin: 0 !important; }
.cardtitle { font-size: 18px !important; font-weight: 600 !important; color: #12203A; margin: 0 0 6px !important; }
.tile { display: flex; flex-direction: column; gap: 2px; padding: 16px 18px; border-radius: 14px; background: #F3F5F9; }
.tile .big { font-size: 28px; line-height: 1.2; }
.chip { display: inline-block; padding: 2px 10px; border-radius: 999px; background: #EEF1F5; color: #4A5568; font-size: 13px; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px; font-size: 14px; font-weight: 600; }
.banner { display: flex; align-items: center; gap: 14px; padding: 14px 16px; border-radius: 12px; }
.banner .head { font-size: 22px; font-weight: 700; line-height: 1.2; }
.v-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.v-chip { flex: 1 1 120px; display: flex; justify-content: space-between; gap: 8px; padding: 6px 10px; border-radius: 10px; background: #EEF1F5; font-size: 13px; }
.barrow { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.barlabel { width: 72px; font-size: 14px; color: #12203A; }
.bartrack { position: relative; flex: 1 1 0; min-width: 0; height: 18px; border-radius: 6px; background: #EEF1F5; }
.bar { position: absolute; top: 0; height: 18px; }
.barmid { position: absolute; left: 50%; top: -2px; width: 1px; height: 22px; background: #8793A3; }
.barval { width: 56px; text-align: right; font-size: 14px; color: #4A5568; }
.legend { display: flex; flex-wrap: wrap; gap: 20px; font-size: 14px; margin-bottom: 12px; color: #12203A; }
.legend span.sw { display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin-right: 8px; vertical-align: -2px; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def render(markup: str) -> None:
    """Render HTML. Lines are stripped and joined so Markdown never treats indentation as code."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


# ============================================================
# LOAD MODEL AND SUPPORT FILES
# ============================================================

REQUIRED_FILES = [
    "fraud_model.pkl", "fraud_threshold.pkl", "feature_columns.pkl",
    "model_metadata.json", "sample_transactions.csv",
    "threshold_comparison.csv", "model_comparison.csv",
]


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    missing = [name for name in REQUIRED_FILES if not (MODEL_DIR / name).exists()]
    if missing:
        return None, missing
    with open(MODEL_DIR / "model_metadata.json") as f:
        meta = json.load(f)
    artifacts = {
        "model": joblib.load(MODEL_DIR / "fraud_model.pkl"),
        "threshold": float(joblib.load(MODEL_DIR / "fraud_threshold.pkl")),
        "features": list(joblib.load(MODEL_DIR / "feature_columns.pkl")),
        "meta": meta,
        "sample": pd.read_csv(MODEL_DIR / "sample_transactions.csv"),
        "threshold_table": pd.read_csv(MODEL_DIR / "threshold_comparison.csv"),
        "model_table": pd.read_csv(MODEL_DIR / "model_comparison.csv"),
    }
    return artifacts, []


artifacts, missing_files = load_artifacts()
if artifacts is None:
    st.error(
        "Model files are missing from the `models` folder: "
        + ", ".join(f"`{name}`" for name in missing_files)
        + ". Run `python train_model.py` and copy the `models` folder next to this app."
    )
    st.stop()

model = artifacts["model"]
SAVED_THRESHOLD = artifacts["threshold"]
FEATURES = artifacts["features"]
meta = artifacts["meta"]
sample = artifacts["sample"]
MODEL_NAME = meta["model_name"]


def version_warnings() -> list:
    """Compare installed package versions with the ones used for training."""
    trained = meta.get("versions", {})
    found = []
    try:
        import sklearn
        if trained.get("scikit-learn") and trained["scikit-learn"] != sklearn.__version__:
            found.append(f"scikit-learn trained with {trained['scikit-learn']}, running {sklearn.__version__}")
    except ImportError:
        pass
    try:
        import xgboost
        if trained.get("xgboost") and trained["xgboost"] != xgboost.__version__:
            found.append(f"xgboost trained with {trained['xgboost']}, running {xgboost.__version__}")
    except ImportError:
        pass
    return found


# ============================================================
# SCORING AND EXPLANATIONS
# ============================================================

def score(frame: pd.DataFrame) -> np.ndarray:
    """Fraud probability for each row. Column order matches training exactly."""
    return model.predict_proba(frame[FEATURES].astype(float))[:, 1]


def explain_row(row_frame: pd.DataFrame, top_n: int = 7):
    """Top feature contributions (log-odds) for one transaction, or None if unavailable."""
    try:
        data = row_frame[FEATURES].astype(float)
        import xgboost as xgb
        if isinstance(model, xgb.XGBClassifier):
            contributions = model.get_booster().predict(xgb.DMatrix(data), pred_contribs=True)[0][:-1]
        elif hasattr(model, "named_steps"):
            return None
        else:
            import shap
            values = shap.TreeExplainer(model).shap_values(data)
            if isinstance(values, list):
                values = values[1]
            values = np.asarray(values)
            contributions = values[0, :, 1] if values.ndim == 3 else values[0]
        pairs = list(zip(FEATURES, [float(c) for c in contributions]))
        return sorted(pairs, key=lambda p: abs(p[1]), reverse=True)[:top_n]
    except Exception:
        return None


# ============================================================
# HTML BUILDERS
# ============================================================

def stat_tile(label: str, value: str, color: str = "#12203A") -> str:
    return (f'<div class="tile"><span class="eyebrow">{html.escape(label)}</span>'
            f'<span class="mono big" style="color:{color}">{html.escape(value)}</span></div>')


def gauge_svg(prob: float, threshold: float) -> str:
    """Half-circle gauge: filled to the probability, with a tick at the threshold."""
    def point(fraction: float, radius: float):
        angle = math.pi * (1 - fraction)
        return 120 + radius * math.cos(angle), 120 - radius * math.sin(angle)

    fill_color = "#B3261E" if prob >= threshold else "#0B6E63"
    track = '<path d="M20 120 A100 100 0 0 1 220 120" fill="none" stroke="#E3E8EE" stroke-width="18" stroke-linecap="round"></path>'
    fill = ""
    if prob > 0.004:
        end_x, end_y = point(min(prob, 1.0), 100)
        fill = (f'<path d="M20 120 A100 100 0 0 1 {end_x:.1f} {end_y:.1f}" fill="none" '
                f'stroke="{fill_color}" stroke-width="18" stroke-linecap="round"></path>')
    x1, y1 = point(threshold, 114)
    x2, y2 = point(threshold, 86)
    tick = f'<path d="M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}" stroke="#12203A" stroke-width="3" stroke-linecap="round"></path>'
    return (
        f'<svg viewBox="0 0 240 150" style="width:100%;max-width:320px;height:auto" role="img" '
        f'aria-label="Fraud probability {prob*100:.1f} percent, decision threshold {threshold:.2f}">'
        f'{track}{fill}{tick}'
        f'<text x="120" y="110" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="38" fill="{fill_color}">{prob*100:.1f}%</text>'
        f'<text x="20" y="144" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="12" fill="#4A5568">0.00</text>'
        f'<text x="220" y="144" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="12" fill="#4A5568">1.00</text>'
        f'</svg>'
    )


def contribution_bars(items) -> str:
    peak = max(abs(value) for _, value in items) or 1.0
    rows = []
    for name, value in items:
        width = abs(value) / peak * 48
        if value >= 0:
            bar = f'<div class="bar" style="left:50%;width:{width:.1f}%;background:#B3261E;border-radius:0 6px 6px 0"></div>'
        else:
            bar = f'<div class="bar" style="right:50%;width:{width:.1f}%;background:#0B6E63;border-radius:6px 0 0 6px"></div>'
        rows.append(
            f'<div class="barrow"><span class="barlabel mono">{html.escape(name)}</span>'
            f'<div class="bartrack">{bar}<div class="barmid"></div></div>'
            f'<span class="barval mono">{value:+.2f}</span></div>'
        )
    return "".join(rows)


def verdict_banner(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return ('<div class="banner" style="background:#FBE9E7;color:#8C1D18">'
                '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8C1D18" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
                '<path d="M12 3l10 18H2L12 3z"></path><path d="M12 10v5"></path><path d="M12 18v.01"></path></svg>'
                '<div><div class="head">Likely fraud</div>'
                '<div style="font-size:14px">Suggested action: hold for manual review</div></div></div>')
    return ('<div class="banner" style="background:#E3F3F0;color:#0B6E63">'
            '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#0B6E63" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="M5 12l5 5 9-10"></path></svg>'
            '<div><div class="head">Looks legitimate</div>'
            '<div style="font-size:14px">Suggested action: approve</div></div></div>')


def outcome_pill(true_label: int, flagged: bool) -> str:
    if true_label == 1 and flagged:
        text, bg, fg = "Model is correct: fraud caught", "#E3F3F0", "#0B6E63"
    elif true_label == 0 and not flagged:
        text, bg, fg = "Model is correct: legit approved", "#E3F3F0", "#0B6E63"
    elif true_label == 1 and not flagged:
        text, bg, fg = "Model missed this fraud", "#FBE9E7", "#8C1D18"
    else:
        text, bg, fg = "False alarm: this was legit", "#FFF2D6", "#7A4B00"
    return f'<span class="pill" style="background:{bg};color:{fg}">{text}</span>'


def feature_chips(row: pd.Series) -> str:
    chips = "".join(
        f'<div class="v-chip mono"><span style="color:#4A5568">{html.escape(col)}</span><span>{row[col]:.3f}</span></div>'
        for col in FEATURES if col not in ("Time", "Amount")
    )
    return f'<div class="v-grid">{chips}</div>'


# ============================================================
# REUSABLE CARDS
# ============================================================

def prediction_card(prob: float, threshold: float, true_label=None) -> None:
    with st.container(border=True):
        render('<p class="cardtitle">Prediction</p>')
        render(verdict_banner(prob, threshold))
        render(f'<div style="display:flex;flex-direction:column;align-items:center;gap:4px;padding-bottom:6px">{gauge_svg(prob, threshold)}'
               f'<span class="muted">Fraud probability. The dark tick marks the {threshold:.2f} threshold.</span></div>')
        if true_label is not None:
            label_text = "Fraud" if true_label == 1 else "Legit"
            render(f'<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:10px;margin:8px 0 10px">'
                   f'<span class="chip" style="font-size:14px">True label: <strong>{label_text}</strong></span>'
                   f'{outcome_pill(true_label, prob >= threshold)}</div>')


def explanation_card(row_frame: pd.DataFrame) -> None:
    items = explain_row(row_frame)
    if not items:
        return
    with st.container(border=True):
        render('<p class="cardtitle">Why the model scored it this way</p>'
               '<p class="muted" style="margin-top:2px">Top contributions to this prediction (TreeSHAP, in log-odds).</p>')
        render('<div class="legend">'
               '<span><span class="sw" style="background:#B3261E"></span>Pushes toward fraud (right)</span>'
               '<span><span class="sw" style="background:#0B6E63"></span>Pushes toward legit (left)</span></div>')
        render(contribution_bars(items))
        render('<p class="muted" style="font-size:13px">The V-names are anonymized, so only the ranking carries meaning. '
               'Read it as "which hidden signals mattered most".</p>')


def about_model_card(threshold: float) -> None:
    test = meta["test_metrics"]
    split = meta["split"]
    fraud_share = split["test_fraud"] / split["test_rows"] * 100
    with st.container(border=True):
        render('<p class="cardtitle">About the model</p>')
        columns = st.columns(4)
        for col, (label, key) in zip(columns, [("Precision", "Precision"), ("Recall", "Recall"),
                                               ("F1 score", "F1 Score"), ("PR-AUC", "PR-AUC")]):
            with col:
                render(stat_tile(label, f"{test[key]:.2f}"))
        render(
            f'<p class="muted" style="margin-top:14px">{html.escape(MODEL_NAME)} scored on a held-out test set of '
            f'{split["test_rows"]:,} transactions ({split["test_fraud"]} fraud). It caught {test["True Positives"]} of '
            f'{split["test_fraud"]} frauds, missed {test["False Negatives"]}, and raised {test["False Positives"]} false alarms. '
            f'Fraud is only about {fraud_share:.2f}% of transactions, so accuracy is not shown as a headline number.</p>'
            f'<p class="muted" style="font-size:13px">The model and the {SAVED_THRESHOLD:.2f} threshold were chosen on a separate '
            f'validation set; the test set was used once for the numbers above. '
            f'V1 to V28 are anonymized PCA components from the public credit card dataset.</p>'
        )
        with st.expander("How the four models compared (validation data, threshold 0.50)"):
            table = artifacts["model_table"][["Model", "Precision", "Recall", "F1 Score", "PR-AUC"]]
            st.dataframe(table, hide_index=True)


# ============================================================
# SIDEBAR
# ============================================================

FILTERS = ["Any transaction", "Fraud only", "Legit only"]

with st.sidebar:
    render('<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
           '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#F2B84B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
           '<path d="M12 3l8 3v6c0 4.5-3.2 8.2-8 9-4.8-.8-8-4.5-8-9V6l8-3z"></path><path d="M9 12l2 2 4-4"></path></svg>'
           '<span style="font-size:19px;font-weight:700">Fraud Detector</span></div>')

    mode = st.radio("Input mode", ["Sample transaction", "Upload CSV", "Manual entry"])

    sample_filter = FILTERS[1]
    load_clicked = False
    if mode == "Sample transaction":
        sample_filter = st.selectbox("Sample filter", FILTERS, index=1)
        st.caption("Fraud is rare in real data. This sample includes extra fraud rows so you can see both cases.")
        load_clicked = st.button("Load random transaction", type="primary")

    threshold = st.slider("Decision threshold", 0.05, 0.95, float(SAVED_THRESHOLD), 0.01)
    table = artifacts["threshold_table"]
    table = table[table["Model"] == MODEL_NAME]
    if not table.empty:
        near = table.iloc[(table["Threshold"] - threshold).abs().argsort().iloc[0]]
        st.caption(
            f"Closest tested threshold ({near['Threshold']:.2f}): validation precision "
            f"{near['Precision']:.2f}, recall {near['Recall']:.2f}. "
            "Lower catches more fraud but flags more legit payments."
        )
    st.caption(f"Training default: {SAVED_THRESHOLD:.2f}")

    st.info("V1 to V28 are anonymized PCA components, so they come from real dataset rows instead of being typed by hand.")

    warnings_found = version_warnings()
    if warnings_found:
        st.warning("Package versions differ from training: " + "; ".join(warnings_found))


# ============================================================
# HEADER
# ============================================================

head_left, head_right = st.columns([4, 1])
with head_left:
    render('<p class="title">Credit Card Fraud Detection</p>'
           '<p class="muted" style="max-width:640px;margin-top:8px">Score a real transaction, a batch of transactions, '
           'or your own values, and see how the trained model decides.</p>')
with head_right:
    render('<div style="text-align:right"><span class="pill" style="background:#E3F3F0;color:#0B6E63">'
           '<span style="width:8px;height:8px;border-radius:50%;background:#0B6E63;display:inline-block"></span>Model loaded</span></div>')

tile_cols = st.columns(3)
with tile_cols[0]:
    render(stat_tile("Model", MODEL_NAME))
with tile_cols[1]:
    render(stat_tile("Decision threshold", f"{threshold:.2f}"))
with tile_cols[2]:
    render(stat_tile("Test PR-AUC", f"{meta['test_metrics']['PR-AUC']:.2f}"))

st.write("")


# ============================================================
# MODE 1: SAMPLE TRANSACTION
# ============================================================

def sample_mode() -> None:
    if sample_filter == "Fraud only":
        pool = sample.index[sample["Class"] == 1]
    elif sample_filter == "Legit only":
        pool = sample.index[sample["Class"] == 0]
    else:
        pool = sample.index

    needs_new = (
        "sample_id" not in st.session_state
        or st.session_state.get("sample_filter_used") != sample_filter
        or load_clicked
    )
    if needs_new:
        st.session_state["sample_id"] = int(np.random.default_rng().choice(pool))
        st.session_state["sample_filter_used"] = sample_filter

    sample_id = st.session_state["sample_id"]
    row = sample.loc[sample_id]
    row_frame = row[FEATURES].to_frame().T.astype(float)
    prob = float(score(row_frame)[0])
    true_label = int(row["Class"])

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render('<div style="display:flex;justify-content:space-between;align-items:center;gap:8px">'
                   '<p class="cardtitle">Transaction</p>'
                   f'<span class="chip">Test set, sample row {sample_id + 1:,}</span></div>')
            render('<div style="display:flex;flex-wrap:wrap;gap:32px;margin:16px 0">'
                   f'<div style="display:flex;flex-direction:column"><span class="eyebrow">Amount</span>'
                   f'<span class="mono" style="font-size:34px;line-height:1.2">${row["Amount"]:,.2f}</span></div>'
                   f'<div style="display:flex;flex-direction:column"><span class="eyebrow">Time</span>'
                   f'<span class="mono" style="font-size:34px;line-height:1.2">{row["Time"]:,.0f} s</span></div></div>')
            with st.expander("Advanced: PCA features V1 to V28", expanded=True):
                render(feature_chips(row))
    with right:
        prediction_card(prob, threshold, true_label)

    explanation_card(row_frame)


# ============================================================
# MODE 2: UPLOAD CSV
# ============================================================

@st.cache_data
def example_csv() -> bytes:
    fraud_rows = sample[sample["Class"] == 1].head(5)
    legit_rows = sample[sample["Class"] == 0].head(20)
    return pd.concat([fraud_rows, legit_rows]).to_csv(index=False).encode("utf-8")


def upload_mode() -> None:
    with st.container(border=True):
        render('<p class="cardtitle">Batch scoring</p>'
               f'<p class="muted" style="margin-top:2px">Upload a CSV with these {len(FEATURES)} columns: '
               f'<span class="mono">{html.escape(", ".join(FEATURES[:3]))}, ... {html.escape(FEATURES[-1])}</span>. '
               'A Class column (0 or 1) is optional and, if present, is used to check the predictions.</p>')
        uploaded = st.file_uploader("CSV file", type=["csv"])
        st.download_button("Download an example CSV", example_csv(), "example_transactions.csv", "text/csv")

    if uploaded is None:
        return

    try:
        data = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Could not read that file as a CSV: {error}")
        return

    absent = [c for c in FEATURES if c not in data.columns]
    if absent:
        st.error("These required columns are missing: " + ", ".join(absent))
        return

    numeric = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    bad_rows = int(numeric.isnull().any(axis=1).sum())
    if bad_rows:
        st.error(f"{bad_rows} rows contain empty or non-numeric values. Fix them and upload again.")
        return
    if numeric.empty:
        st.error("The file has no rows.")
        return

    data[FEATURES] = numeric
    probabilities = score(data)
    flagged = probabilities >= threshold

    results = pd.DataFrame({
        "Row": np.arange(1, len(data) + 1),
        "Amount": data["Amount"].round(2),
        "Fraud probability": probabilities.round(4),
        "Verdict": np.where(flagged, "Fraud", "Legit"),
    })
    has_labels = "Class" in data.columns
    if has_labels:
        results["True label"] = np.where(data["Class"].astype(int) == 1, "Fraud", "Legit")

    with st.container(border=True):
        render('<p class="cardtitle">Results</p>')
        c1, c2, c3 = st.columns(3)
        with c1:
            render(stat_tile("Rows scored", f"{len(results):,}"))
        with c2:
            render(stat_tile("Flagged as fraud", f"{int(flagged.sum()):,}", "#B3261E"))
        with c3:
            render(stat_tile("Flag rate", f"{flagged.mean()*100:.1f}%"))

        if has_labels:
            truth = data["Class"].astype(int).to_numpy() == 1
            tp = int((flagged & truth).sum())
            fp = int((flagged & ~truth).sum())
            fn = int((~flagged & truth).sum())
            precision_text = f"{tp / (tp + fp):.2f}" if tp + fp else "n/a"
            recall_text = f"{tp / (tp + fn):.2f}" if tp + fn else "n/a"
            st.caption(f"Against the Class column: {tp} frauds caught, {fn} missed, {fp} false alarms "
                       f"(precision {precision_text}, recall {recall_text}).")

        shown = results.sort_values("Fraud probability", ascending=False).head(500)
        st.caption("Highest risk first. Showing up to 500 rows; the download contains every row.")
        st.dataframe(
            shown,
            hide_index=True,
            column_config={
                "Fraud probability": st.column_config.ProgressColumn(
                    "Fraud probability", min_value=0.0, max_value=1.0, format="%.3f"),
            },
        )
        st.download_button(
            "Download results CSV",
            results.to_csv(index=False).encode("utf-8"),
            "fraud_predictions.csv",
            "text/csv",
        )

    default_row = int(results["Fraud probability"].idxmax()) + 1
    row_number = st.number_input("Explain row number", min_value=1, max_value=len(data),
                                 value=default_row, step=1,
                                 help="Defaults to the riskiest row in the file.")
    explanation_card(data.iloc[[int(row_number) - 1]])


# ============================================================
# MODE 3: MANUAL ENTRY
# ============================================================

PROFILES = {
    "Typical legit transaction": sample[sample["Class"] == 0][FEATURES].median(),
    "Typical fraud transaction": sample[sample["Class"] == 1][FEATURES].median(),
}


def apply_profile() -> None:
    values = PROFILES[st.session_state["man_profile"]]
    for column in FEATURES:
        st.session_state[f"man_{column}"] = float(values[column])


def manual_mode() -> None:
    if "man_profile" not in st.session_state:
        st.session_state["man_profile"] = list(PROFILES)[0]
    for column in FEATURES:
        if f"man_{column}" not in st.session_state:
            st.session_state[f"man_{column}"] = float(PROFILES[st.session_state["man_profile"]][column])

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render('<p class="cardtitle">Your transaction</p>'
                   '<p class="muted" style="margin-top:2px">Start from a typical profile, then change Amount and Time. '
                   'The PCA features are optional.</p>')
            st.selectbox("Start from", list(PROFILES), key="man_profile", on_change=apply_profile)
            st.number_input("Amount ($)", min_value=0.0, step=1.0, format="%.2f", key="man_Amount")
            st.number_input("Time (seconds since the first transaction in the dataset)",
                            min_value=0.0, step=1.0, format="%.0f", key="man_Time")
            with st.expander("Advanced: PCA features V1 to V28"):
                pca_columns = [c for c in FEATURES if c not in ("Time", "Amount")]
                grid = st.columns(4)
                for index, column in enumerate(pca_columns):
                    with grid[index % 4]:
                        st.number_input(column, step=0.1, format="%.3f", key=f"man_{column}")

    values = {column: float(st.session_state[f"man_{column}"]) for column in FEATURES}
    row_frame = pd.DataFrame([values])[FEATURES]
    prob = float(score(row_frame)[0])

    with right:
        prediction_card(prob, threshold)

    explanation_card(row_frame)


# ============================================================
# RUN THE SELECTED MODE
# ============================================================

if mode == "Sample transaction":
    sample_mode()
elif mode == "Upload CSV":
    upload_mode()
else:
    manual_mode()

st.write("")
about_model_card(threshold)