import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import base64
import streamlit.components.v1 as components

from src.ai_crisis.preprocessing import simple_clean
from src.ai_crisis.model_io import load_model_and_vectorizer
from src.ai_crisis.predict import predict_text 

# -------------------------
# Config / paths
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "models")
MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, "linear_svm_calibrated.pkl"),
    os.path.join(MODELS_DIR, "linear_svm_model.pkl"),
]
VECT_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")
META_PATH = os.path.join(MODELS_DIR, "metadata.json")

# -------------------------
# Page config and theme
# -------------------------
page_icon = "🌍"
logo_path = os.path.join(PROJECT_ROOT, "logo.png")
if os.path.exists(logo_path):
    page_icon = logo_path

st.set_page_config(page_title="AI-Crisis Tweet Classifier", page_icon=page_icon, layout="wide")

st.markdown(
    """
<style>
.main {
    background-color: #0E1117;
    color: #FAFAFA;
}
[data-testid="stSidebar"] {
    background-color: #161A22;
    padding: 1.2rem 1rem;
    border-right: 1px solid #262730;
}
h1, h2, h3 {
    color: #FAFAFA !important;
}
.stButton > button {
    background-color: #1F6FEB;
    color: white;
    border-radius: 8px;
    border: none;
    transition: all 0.12s ease-in-out;
    padding: 0.45rem 0.9rem;
}
.stButton > button:hover {
    transform: translateY(-2px);
}
.stSuccess {
    background-color: rgba(35, 134, 54, 0.14);
    border-left: 4px solid #238636;
}
div[data-testid="stProgress"] > div > div > div {
    background-color: #1F6FEB;
    border-radius: 10px;
}
textarea, input {
    border-radius: 8px !important;
    border: 1px solid #30363D !important;
}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Header Banner (render with components.html)
# -------------------------
banner_logo_html = ""
if os.path.exists(logo_path):
    banner_logo_html = f'<img src="{logo_path}" width="56" style="vertical-align: middle; margin-right: 12px;">'

banner_html = f"""
<div style="background-color:#0F1720; padding:1rem; border-radius:8px; margin-bottom:1rem; display:flex; align-items:center;">
    {banner_logo_html}
    <div>
        <h1 style="color:#FAFAFA; text-align:left; margin-bottom:0.1rem;">⚡ AI-Powered Crisis Tweet Classifier</h1>
        <p style="color:#9AA0A6; text-align:left; font-size:14px; margin-top:0.2rem; margin-bottom:0;">
            Instantly classify crisis-related tweets as <b>informative</b> or <b>not_informative</b>
            using a calibrated Linear SVM (TF-IDF).
        </p>
    </div>
</div>
"""

components.html(banner_html, height=120)

# -------------------------
# Model Loading (cached)
# -------------------------
@st.cache_resource
def load_resources():
    model, vect = None, None
    for mp in MODEL_CANDIDATES:
        if os.path.exists(mp):
            try:
                model, vect = load_model_and_vectorizer(mp, VECT_PATH)
                if model is not None:
                    break
            except Exception as e:
                st.warning(f"Failed to load model from {mp}: {e}")
    if model is None and os.path.exists(VECT_PATH):
        try:
            _, vect = load_model_and_vectorizer(MODEL_CANDIDATES[0], VECT_PATH)
        except Exception:
            pass
    return model, vect

model, vect = load_resources()

# -------------------------
# Load label_map if present
# -------------------------
label_map = None
if os.path.exists(LABEL_MAP_PATH):
    try:
        label_map = json.load(open(LABEL_MAP_PATH, encoding="utf-8"))
    except Exception:
        label_map = None

def map_pred_to_label(p):
    """
    Convert a raw prediction (int, numpy scalar, or string) to a human-readable label.
    Uses label_map if available; otherwise falls back to 0/1 mapping.
    """
    if label_map is not None:
        key = str(p)
        return label_map.get(key, key)
    # fallback numeric mapping
    try:
        if str(p).isdigit():
            return "informative" if int(p) == 1 else "not_informative"
    except Exception:
        pass
    return str(p)

# -------------------------
# Prediction Helper
# -------------------------
def batch_predict(model, vect, texts):
    cleaned = [simple_clean(t) for t in texts]
    X = vect.transform(cleaned)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        preds = model.predict(X)
        confs = []
        for i, _ in enumerate(preds):
            idx = int(np.argmax(probs[i]))
            confs.append(float(probs[i, idx]))
        return preds, np.array(confs), cleaned

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            confs = 1 / (1 + np.exp(-scores))
        else:
            confs = 1 / (1 + np.exp(-np.max(scores, axis=1)))
        preds = model.predict(X)
        return preds, np.array(confs), cleaned

    preds = model.predict(X)
    confs = np.repeat(0.5, len(preds))
    return preds, confs, cleaned

# -------------------------
# Sidebar Info
# -------------------------
with st.sidebar:
    st.header("Model status")
    if model is None or vect is None:
        st.error("Model or vectorizer not found in `data/processed/models/`.")
        st.write("Expected files:")
        for p in MODEL_CANDIDATES:
            st.write(f"- {os.path.relpath(p, PROJECT_ROOT)}")
        st.write(f"- {os.path.relpath(VECT_PATH, PROJECT_ROOT)}")
    else:
        st.success("Model and vectorizer loaded ✅")
        st.write(f"Model type: `{model.__class__.__name__}`")
        # Classes (readable)
        try:
            classes_readable = []
            for c in getattr(model, "classes_", []):
                try:
                    if hasattr(c, "item"):
                        classes_readable.append(str(int(c)))
                    else:
                        classes_readable.append(str(c))
                except Exception:
                    classes_readable.append(str(c))
            if label_map is not None:
                classes_readable = [label_map.get(str(x), str(x)) for x in classes_readable]
            st.json({i: label for i, label in enumerate(classes_readable)})
        except Exception:
            pass
        # Vectorizer
        try:
            st.write("Vectorizer vocab size:", len(vect.vocabulary_))
        except Exception:
            pass
        # Metadata
        if os.path.exists(META_PATH):
            try:
                meta = json.load(open(META_PATH, encoding="utf-8"))
                st.markdown("---")
                st.subheader("Model info")
                st.write(f"**Name:** {meta.get('model_name', 'N/A')}")
                st.write(f"**Version:** {meta.get('version', 'N/A')}")
                st.write(f"**Saved:** {meta.get('date_saved', 'N/A')}")
            except Exception:
                pass

        st.markdown("---")
        demo_data = pd.DataFrame({
            "tweet_text": [
                "Massive earthquake in city center, people trapped under debris.",
                "Attending the concert tonight, can't wait!",
                "Fire near gas station, emergency crews responding.",
                "Sunny day by the beach 🌊"
            ]
        })
        csv_data = demo_data.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Demo CSV",
            data=csv_data,
            file_name="demo_tweets.csv",
            mime="text/csv",
        )

# -------------------------
# Main Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["💬 Single Tweet", "📂 Batch CSV", "ℹ️ About"])

# -------------------------
# Tab 1: Single Tweet
# -------------------------
with tab1:
    st.subheader("Classify a single tweet")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    if col_ex1.button("Example — Flood"):
        st.session_state["single_text"] = "Flood in district X, bridges washed away, need rescue teams!"
    if col_ex2.button("Example — Fire"):
        st.session_state["single_text"] = "Huge fire near central market, people trapped, fire service needed."
    if col_ex3.button("Example — Not urgent"):
        st.session_state["single_text"] = "Watching the game at home, big crowd here."

    single_text = st.text_area(
        "Enter tweet text",
        value=st.session_state.get("single_text", ""),
        height=140,
        placeholder="Type or paste a tweet..."
    )

    if st.button("Predict (single)"):
        if not single_text.strip():
            st.warning("Please enter tweet text.")
        elif model is None or vect is None:
            st.error("Model or vectorizer missing.")
        else:
            preds, confs, cleaned = batch_predict(model, vect, [single_text])
            pred_raw, conf = preds[0], float(confs[0])
            label = map_pred_to_label(pred_raw)
            st.success(f"Predicted label: **{label}**")
            st.write(f"Confidence: **{conf*100:.2f}%**")
            prog = min(max(int(conf * 100), 0), 100)
            st.progress(prog)
            st.caption("Cleaned input used for vectorizer:")
            st.code(cleaned[0])

# -------------------------
# Tab 2: Batch CSV
# -------------------------
with tab2:
    st.subheader("Batch classify tweets from a CSV file")
    st.write("Upload a CSV with a text column (`clean_text`, `tweet_text`, `text`, or `tweet`).")
    uploaded = st.file_uploader("Upload CSV with tweets", type=["csv"])

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df_in = None

        if df_in is not None:
            text_col = next((c for c in ["clean_text", "tweet_text", "text", "tweet"] if c in df_in.columns), None)
            if not text_col:
                st.error("CSV must contain one of: clean_text, tweet_text, text, tweet.")
            elif model is None or vect is None:
                st.error("Model/vectorizer missing.")
            else:
                with st.spinner("Classifying..."):
                    preds, confs, cleaned = batch_predict(model, vect, df_in[text_col].astype(str).tolist())

                # Map predictions to readable labels 
                mapped_labels = [map_pred_to_label(p) for p in preds]

                df_out = df_in.copy()
                df_out["pred_label"] = mapped_labels
                df_out["pred_conf"] = np.round(confs, 6)
                df_out["clean_text_used"] = cleaned

                st.success(f"Classified {len(df_out)} rows.")

                # Top 5 preview
                try:
                    preview = (
                        df_out[[text_col, "pred_label", "pred_conf"]]
                        .rename(columns={text_col: "Tweet Text", "pred_label": "Predicted Label", "pred_conf": "Confidence"})
                        .sort_values(by="Confidence", ascending=False)
                        .head(5)
                    )
                    preview["Confidence"] = (preview["Confidence"] * 100).map(lambda v: f"{v:.2f}%")
                    st.markdown("### 🔎 Top 5 predictions (by confidence)")
                    st.dataframe(preview.reset_index(drop=True), use_container_width=True)
                except Exception:
                    pass

                # Metrics (consistent because labels are mapped)
                try:
                    cnt_info = int((df_out["pred_label"] == "informative").sum())
                    cnt_not = int((df_out["pred_label"] == "not_informative").sum())
                    colA, colB = st.columns(2)
                    colA.metric("Informative", cnt_info)
                    colB.metric("Not Informative", cnt_not)
                except Exception:
                    pass

                # Download full CSV
                buf = io.StringIO()
                df_out.to_csv(buf, index=False)
                st.download_button("📥 Download full predictions CSV", buf.getvalue(), "predictions.csv", "text/csv")

# -------------------------
# Tab 3: About
# -------------------------
with tab3:
    st.subheader("About this project")
    st.markdown(
        """
**AI-Powered Crisis Tweet Classifier** uses a Linear SVM trained on crisis-related datasets (e.g., CrisisLexT26)
to classify tweets as *informative* or *not informative* during emergency events.

### 💡 Features
- Real-time single tweet classification  
- Batch CSV upload + top-5 preview  
- Calibrated SVM for confidence estimation  
- Modern Streamlit UI (Dark Mode)

### ⚙️ Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
"""
)



demo_gif = os.path.join(PROJECT_ROOT, "demo.gif")
if os.path.exists(demo_gif):
    with open(demo_gif, "rb") as f:
        gif_bytes = f.read()
    gif_b64 = base64.b64encode(gif_bytes).decode("utf-8")  
    gif_html = f"""
    <div style="text-align:center;">
        <img
            src="data:image/gif;base64,{gif_b64}"
            alt="App demo"
            style="max-width:100%; height:auto; border-radius:8px;"
        />
    </div>
    """
    components.html(gif_html, height=800)
else:
    st.info("Add a `demo.gif` to show a short animation here.")



    
#-------------------------
#Footer / Notes
#-------------------------

st.markdown("---")
st.caption("Notes:")
st.markdown(
"""

The app prefers a calibrated model (linear_svm_calibrated.pkl) for accurate probabilities.

Falls back to decision_function if calibration unavailable.

Ensure tfidf_vectorizer.pkl matches the training vectorizer.

@MD SHAOWN RAHMAN
"""
)