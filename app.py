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
/* Main app background */
.main {
    background: linear-gradient(135deg, #0E1117 0%, #1a1f2e 100%);
    color: #FAFAFA;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161A22 0%, #1f2632 100%);
    padding: 1rem 0.8rem;
    border-right: 1px solid #262730;
}

/* Typography */
h1, h2, h3, h4 {
    color: #FAFAFA !important;
    font-weight: 600;
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}

h1 {
    background: linear-gradient(135deg, #1F6FEB 0%, #8B5CF6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1F6FEB 0%, #2563EB 100%);
    color: white;
    border-radius: 10px;
    border: none;
    transition: all 0.3s ease;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    box-shadow: 0 4px 6px rgba(31, 111, 235, 0.3);
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(31, 111, 235, 0.4);
    background: linear-gradient(135deg, #2563EB 0%, #1F6FEB 100%);
}

/* Success/Info boxes */
.stSuccess {
    background: linear-gradient(135deg, rgba(35, 134, 54, 0.15) 0%, rgba(35, 134, 54, 0.1) 100%);
    border-left: 4px solid #238636;
    border-radius: 8px;
    padding: 1rem;
}

.stInfo {
    background: linear-gradient(135deg, rgba(31, 111, 235, 0.15) 0%, rgba(31, 111, 235, 0.1) 100%);
    border-left: 4px solid #1F6FEB;
    border-radius: 8px;
    padding: 1rem;
}

.stWarning {
    background: linear-gradient(135deg, rgba(251, 188, 5, 0.15) 0%, rgba(251, 188, 5, 0.1) 100%);
    border-left: 4px solid #FBB805;
    border-radius: 8px;
    padding: 1rem;
}

.stError {
    background: linear-gradient(135deg, rgba(248, 81, 73, 0.15) 0%, rgba(248, 81, 73, 0.1) 100%);
    border-left: 4px solid #F85149;
    border-radius: 8px;
    padding: 1rem;
}

/* Progress bars */
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #1F6FEB 0%, #8B5CF6 100%);
    border-radius: 10px;
}

/* Input fields */
textarea, input {
    border-radius: 10px !important;
    border: 2px solid #30363D !important;
    background-color: #161B22 !important;
    color: #FAFAFA !important;
    transition: all 0.3s ease;
}

textarea:focus, input:focus {
    border-color: #1F6FEB !important;
    box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.1) !important;
}

/* Cards/Containers */
.custom-card {
    background: linear-gradient(135deg, rgba(22, 27, 34, 0.8) 0%, rgba(30, 36, 46, 0.8) 100%);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 700;
}

[data-testid="stMetricLabel"] {
    font-size: 0.9rem;
    opacity: 0.8;
}

/* Badges */
.prediction-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.95rem;
    margin: 0.5rem 0;
}

.badge-informative {
    background: linear-gradient(135deg, #238636 0%, #2EA043 100%);
    color: white;
    box-shadow: 0 2px 8px rgba(35, 134, 54, 0.3);
}

.badge-not-informative {
    background: linear-gradient(135deg, #F85149 0%, #FF6B6B 100%);
    color: white;
    box-shadow: 0 2px 8px rgba(248, 81, 73, 0.3);
}

/* Confidence bar container */
.confidence-container {
    background: rgba(22, 27, 34, 0.6);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid #30363D;
}

/* Stats cards */
.stats-card {
    background: linear-gradient(135deg, rgba(31, 111, 235, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
    border: 1px solid rgba(31, 111, 235, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

/* Tabs */
[data-baseweb="tab-list"] {
    gap: 0.3rem;
    background-color: transparent;
    margin-bottom: 0.5rem;
}

[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

[data-baseweb="tab"]:hover {
    background-color: rgba(31, 111, 235, 0.1);
}

[aria-selected="true"] {
    background-color: rgba(31, 111, 235, 0.2) !important;
    border-bottom: 2px solid #1F6FEB !important;
}

/* Dataframe styling */
.dataframe {
    border-radius: 8px;
    overflow: hidden;
}

/* Footer */
footer {visibility: hidden;}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #161B22;
}

::-webkit-scrollbar-thumb {
    background: #30363D;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #484F58;
}

/* Character counter */
.char-counter {
    text-align: right;
    font-size: 0.85rem;
    color: #9AA0A6;
    margin-top: 0.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Hero Text
# -------------------------
st.markdown("""
<div style="text-align: center; padding: 1rem 0 1.5rem 0;">
    <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #1F6FEB 0%, #8B5CF6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        ⚡ AI-Crisis Tweet Classifier
    </h1>
    <p style="color: #9AA0A6; font-size: 1rem; margin-top: 0;">
        Classify crisis-related tweets as informative or not_informative using ML
    </p>
</div>
""", unsafe_allow_html=True)

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
    st.markdown("### 📊 Model Status")
    st.markdown("---")
    
    if model is None or vect is None:
        st.error("⚠️ Model or vectorizer not found")
        with st.expander("Expected files", expanded=False):
            for p in MODEL_CANDIDATES:
                st.text(f"• {os.path.basename(p)}")
            st.text(f"• {os.path.basename(VECT_PATH)}")
    else:
        # Status badge
        st.success("✅ Model and vectorizer loaded successfully")
        
        # Model info card
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown(f"**Model Type:** `{model.__class__.__name__}`")
        
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
            
            st.markdown("**Classes:**")
            for i, label in enumerate(classes_readable):
                badge_color = "#238636" if "informative" in label.lower() else "#F85149"
                st.markdown(f'<span style="color: {badge_color}; font-weight: 600;">{i}: {label}</span>', unsafe_allow_html=True)
        except Exception:
            pass
        
        # Vectorizer
        try:
            vocab_size = len(vect.vocabulary_)
            st.markdown(f"**Vocabulary Size:** `{vocab_size:,}`")
        except Exception:
            pass
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Metadata
        if os.path.exists(META_PATH):
            try:
                meta = json.load(open(META_PATH, encoding="utf-8"))
                st.markdown("---")
                st.markdown("### 📋 Model Metadata")
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.write(f"**Name:** {meta.get('model_name', 'N/A')}")
                st.write(f"**Version:** {meta.get('version', 'N/A')}")
                st.write(f"**Saved:** {meta.get('date_saved', 'N/A')}")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception:
                pass

        st.markdown("---")
        st.markdown("### 🎯 Quick Start")
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
            label="📥 Download Demo CSV",
            data=csv_data,
            file_name="demo_tweets.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("💬 Use example buttons for quick testing\n\n📊 CSV should contain: `tweet_text`, `text`, or `clean_text`\n\n🎯 Higher confidence = more reliable prediction")

# -------------------------
# Main Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["💬 Single Tweet", "📂 Batch CSV", "ℹ️ About"])

# -------------------------
# Tab 1: Single Tweet
# -------------------------
with tab1:
    # Initialize session state if not present
    if "single_text" not in st.session_state:
        st.session_state["single_text"] = ""
    
    # Example texts
    example_texts = {
        "flood": "Flood in district X, bridges washed away, need rescue teams!",
        "fire": "Huge fire near central market, people trapped, fire service needed.",
        "not_urgent": "Watching the game at home, big crowd here."
    }
    
    # Compact layout - text area and example buttons side by side
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Text area with character counter
        # Don't use key to avoid session state modification conflicts
        single_text = st.text_area(
            "**Enter Tweet Text**",
            value=st.session_state.get("single_text", ""),
            height=100,
            placeholder="Type or paste a tweet here...",
            help="The text will be automatically cleaned and preprocessed before classification"
        )
        
        # Update session state when text changes
        st.session_state["single_text"] = single_text
        
        # Character counter and predict button in same row
        col_char, col_btn = st.columns([3, 1])
        with col_char:
            char_count = len(single_text)
            char_limit = 280
            char_color = "#9AA0A6" if char_count <= char_limit else "#F85149"
            st.markdown(f'<div class="char-counter" style="color: {char_color}; margin-top: -10px;">{char_count}/{char_limit} characters</div>', unsafe_allow_html=True)
        with col_btn:
            predict_btn = st.button("🚀 Predict", use_container_width=True, type="primary")
    
    with col_right:
        st.markdown("**Quick Examples:**")
        if st.button("🌊 Flood", use_container_width=True, help="Example of an informative crisis tweet", key="btn_flood"):
            st.session_state["single_text"] = example_texts["flood"]
            st.rerun()
        if st.button("🔥 Fire", use_container_width=True, help="Another informative crisis example", key="btn_fire"):
            st.session_state["single_text"] = example_texts["fire"]
            st.rerun()
        if st.button("📺 Not Urgent", use_container_width=True, help="Example of a non-informative tweet", key="btn_not_urgent"):
            st.session_state["single_text"] = example_texts["not_urgent"]
            st.rerun()
    
    # Prediction logic
    if predict_btn:
        if not single_text.strip():
            st.warning("⚠️ Please enter tweet text before predicting.")
        elif model is None or vect is None:
            st.error("❌ Model or vectorizer is missing. Please check the sidebar for model status.")
        else:
            with st.spinner("🔄 Processing tweet..."):
                preds, confs, cleaned = batch_predict(model, vect, [single_text])
                pred_raw, conf = preds[0], float(confs[0])
                label = map_pred_to_label(pred_raw)
            
            # Prediction results - compact layout
            st.markdown("---")
            
            # Compact results in columns
            col_result1, col_result2 = st.columns([2, 1])
            
            with col_result1:
                # Determine badge class and color
                is_informative = "informative" in label.lower()
                badge_class = "badge-informative" if is_informative else "badge-not-informative"
                conf_color = "#238636" if is_informative else "#F85149"
                
                # Label badge
                st.markdown(f'<div class="prediction-badge {badge_class}" style="margin-bottom: 0.5rem;">📌 {label.replace("_", " ").title()}</div>', unsafe_allow_html=True)
                
                # Progress bar
                conf_percent = conf * 100
                prog = min(max(int(conf * 100), 0), 100)
                st.progress(prog)
                st.caption(f"Confidence: **{conf_percent:.1f}%**")
            
            with col_result2:
                if conf_percent >= 80:
                    conf_text = "Very High"
                elif conf_percent >= 65:
                    conf_text = "High"
                elif conf_percent >= 50:
                    conf_text = "Moderate"
                else:
                    conf_text = "Low"
                st.markdown(f"<div style='padding: 1rem; background: rgba(31, 111, 235, 0.1); border-radius: 8px; text-align: center;'><strong style='color: {conf_color};'>{conf_text} Confidence</strong></div>", unsafe_allow_html=True)
            
            # Cleaned text preview
            with st.expander("🔍 View Preprocessed Text", expanded=False):
                st.code(cleaned[0], language="text")

# -------------------------
# Tab 2: Batch CSV
# -------------------------
with tab2:
    # Compact header with file uploader and format info
    col_batch1, col_batch2 = st.columns([3, 1])
    with col_batch1:
        uploaded = st.file_uploader(
            "📤 Upload CSV File", 
            type=["csv"],
            help="Select a CSV file containing tweet text in one of the supported columns"
        )
    with col_batch2:
        with st.expander("📋 Format", expanded=False):
            st.markdown("""
            **Required columns:**
            - `clean_text`
            - `tweet_text`
            - `text`
            - `tweet`
            """)

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            st.success(f"✅ Successfully loaded CSV with **{len(df_in)}** rows")
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
            st.info("💡 Make sure your file is a valid CSV format and try again.")
            df_in = None

        if df_in is not None:
            # Show preview of uploaded data
            with st.expander("👀 Preview Uploaded Data", expanded=False):
                st.dataframe(df_in.head(10), use_container_width=True)
                st.caption(f"Showing first 10 rows of {len(df_in)} total rows.")
            
            text_col = next((c for c in ["clean_text", "tweet_text", "text", "tweet"] if c in df_in.columns), None)
            if not text_col:
                st.error("❌ CSV must contain one of: `clean_text`, `tweet_text`, `text`, or `tweet`.")
                st.info("💡 Please rename your text column to match one of the supported names.")
            elif model is None or vect is None:
                st.error("❌ Model or vectorizer is missing. Please check the sidebar for model status.")
            else:
                # Classification button
                if st.button("🚀 Start Batch Classification", type="primary", use_container_width=True):
                    with st.spinner(f"🔄 Classifying {len(df_in)} tweets... This may take a moment."):
                        preds, confs, cleaned = batch_predict(model, vect, df_in[text_col].astype(str).tolist())

                    # Map predictions to readable labels 
                    mapped_labels = [map_pred_to_label(p) for p in preds]

                    df_out = df_in.copy()
                    df_out["pred_label"] = mapped_labels
                    df_out["pred_conf"] = np.round(confs, 6)
                    df_out["clean_text_used"] = cleaned

                    st.success(f"✅ Successfully classified **{len(df_out)}** tweets!")
                    try:
                        st.balloons()
                    except Exception:
                        pass  # Balloons not available in all Streamlit versions

                    # Statistics cards - compact
                    try:
                        cnt_info = int((df_out["pred_label"] == "informative").sum())
                        cnt_not = int((df_out["pred_label"] == "not_informative").sum())
                        total = len(df_out)
                        pct_info = (cnt_info / total * 100) if total > 0 else 0
                        pct_not = (cnt_not / total * 100) if total > 0 else 0
                        avg_conf = float(np.mean(confs) * 100)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "✅ Informative",
                                cnt_info,
                                delta=f"{pct_info:.1f}%",
                                delta_color="normal"
                            )
                        
                        with col2:
                            st.metric(
                                "❌ Not Informative",
                                cnt_not,
                                delta=f"{pct_not:.1f}%",
                                delta_color="inverse"
                            )
                        
                        with col3:
                            st.metric(
                                "📈 Average Confidence",
                                f"{avg_conf:.1f}%",
                                help="Average confidence score across all predictions"
                            )
                        
                        with col4:
                            st.metric(
                                "📊 Total Tweets",
                                total
                            )
                    except Exception as e:
                        st.warning(f"Could not compute statistics: {e}")

                    # Top 5 preview - compact
                    try:
                        preview = (
                            df_out[[text_col, "pred_label", "pred_conf"]]
                            .rename(columns={text_col: "Tweet Text", "pred_label": "Predicted Label", "pred_conf": "Confidence"})
                            .sort_values(by="Confidence", ascending=False)
                            .head(5)
                        )
                        preview["Confidence"] = (preview["Confidence"] * 100).map(lambda v: f"{v:.2f}%")
                        
                        with st.expander("🔝 Top 5 Predictions (Highest Confidence)", expanded=True):
                            st.dataframe(preview.reset_index(drop=True), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.warning(f"Could not display preview: {e}")

                    # Download section - compact
                    buf = io.StringIO()
                    df_out.to_csv(buf, index=False)
                    csv_str = buf.getvalue()
                    
                    st.download_button(
                        "📥 Download Full Results CSV",
                        csv_str,
                        "predictions.csv",
                        "text/csv",
                        use_container_width=True,
                        help="Download includes original columns plus pred_label, pred_conf, and clean_text_used"
                    )
                    
                    # Store in session state for potential future use
                    st.session_state["last_batch_results"] = df_out

# -------------------------
# Tab 3: About
# -------------------------
with tab3:
    st.markdown("### ℹ️ About This Project")
    
    st.markdown("""
    <div class="custom-card">
    <h4>🎯 Project Overview</h4>
    <p>
    The <strong>AI-Powered Crisis Tweet Classifier</strong> is a machine learning application designed to 
    automatically classify crisis-related tweets as <span style="color: #238636;"><strong>informative</strong></span> 
    or <span style="color: #F85149;"><strong>not_informative</strong></span> during emergency events.
    </p>
    <p>
    This tool leverages a <strong>Linear SVM</strong> model trained on the <strong>CrisisLexT26</strong> and 
    <strong>CrisisLexT6</strong> datasets, achieving <strong>93% accuracy</strong> in classification tasks.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.markdown("""
        <div class="custom-card">
        <h4>✨ Key Features</h4>
        <ul style="line-height: 2;">
        <li>⚡ <strong>Real-time</strong> single tweet classification</li>
        <li>📊 <strong>Batch processing</strong> via CSV upload</li>
        <li>🎯 <strong>Calibrated confidence</strong> scores</li>
        <li>🎨 <strong>Modern dark-mode</strong> UI</li>
        <li>📈 <strong>Detailed statistics</strong> and analytics</li>
        <li>💾 <strong>Export results</strong> to CSV</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_feat2:
        st.markdown("""
        <div class="custom-card">
        <h4>🧠 Technical Details</h4>
        <ul style="line-height: 2;">
        <li><strong>Model:</strong> Linear SVM (Calibrated)</li>
        <li><strong>Features:</strong> TF-IDF Vectorization</li>
        <li><strong>Accuracy:</strong> 93%</li>
        <li><strong>Precision:</strong> 94%</li>
        <li><strong>Recall:</strong> 93%</li>
        <li><strong>F1-Score:</strong> 93%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    
    with st.expander("📦 Installation & Setup", expanded=False):
        st.markdown("""
        ```bash
        # Clone the repository
        git clone https://github.com/shaownXjony/AI-Crisis-Tweet-Classifier.git
        cd AI-Crisis-Tweet-Classifier

        # Install dependencies
        pip install -r requirements.txt

        # Download NLTK resources (if needed)
        python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

        # Run the Streamlit app
        streamlit run app.py
        ```
        """)
    
    with st.expander("📚 Dataset Information", expanded=False):
        st.markdown("""
        **CrisisLexT26 & CrisisLexT6** are annotated tweet datasets containing:
        - Multiple crisis events (earthquakes, floods, fires, etc.)
        - Binary classification labels (informative / not_informative)
        - Timestamps and event metadata
        - High-quality annotations for training ML models
        """)
    
    with st.expander("🔧 Model Architecture", expanded=False):
        st.markdown("""
        1. **Text Preprocessing**: Cleaning, tokenization, lemmatization, stopword removal
        2. **Feature Extraction**: TF-IDF vectorization with optimized parameters
        3. **Model Training**: Linear SVM with calibrated probabilities
        4. **Evaluation**: Cross-validation and comprehensive metrics
        5. **Deployment**: Streamlit web interface for real-time predictions
        """)
    
    # Demo GIF
    st.markdown("---")
    st.markdown("### 🎬 App Demo")
    
    demo_gif = os.path.join(PROJECT_ROOT, "demo.gif")
    if os.path.exists(demo_gif):
        with open(demo_gif, "rb") as f:
            gif_bytes = f.read()
        gif_b64 = base64.b64encode(gif_bytes).decode("utf-8")  
        gif_html = f"""
        <div style="text-align:center; background: rgba(22, 27, 34, 0.6); padding: 1.5rem; border-radius: 12px; border: 1px solid #30363D;">
            <img
                src="data:image/gif;base64,{gif_b64}"
                alt="App demo"
                style="max-width:100%; height:auto; border-radius:8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);"
            />
        </div>
        """
        components.html(gif_html, height=600)
    else:
        st.info("💡 Add a `demo.gif` file to the project root to display an app demonstration here.")



    
#-------------------------
#Footer / Notes
#-------------------------

st.markdown("---")
st.markdown("""
<div style="background: rgba(22, 27, 34, 0.6); padding: 1.5rem; border-radius: 12px; border: 1px solid #30363D; margin-top: 2rem;">
    <h4 style="color: #FAFAFA; margin-bottom: 1rem;">📝 Technical Notes</h4>
    <ul style="line-height: 2; color: #9AA0A6;">
        <li>The app prefers a calibrated model (<code>linear_svm_calibrated.pkl</code>) for accurate probabilities.</li>
        <li>Falls back to <code>decision_function</code> if calibration unavailable.</li>
        <li>Ensure <code>tfidf_vectorizer.pkl</code> matches the training vectorizer.</li>
    </ul>
    <hr style="border-color: #30363D; margin: 1.5rem 0;">
    <p style="text-align: center; color: #9AA0A6; margin: 0;">
        Developed by <strong style="color: #FAFAFA;">Md. Shaown Rahman</strong> | 
        <a href="https://github.com/shaownXjony" target="_blank" style="color: #1F6FEB; text-decoration: none;">GitHub</a> | 
        <a href="mailto:shaownrahman30@gmail.com" style="color: #1F6FEB; text-decoration: none;">Email</a>
    </p>
</div>
""", unsafe_allow_html=True)
