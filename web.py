import streamlit as st
import tensorflow as tf
import numpy as np
import time

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="KrishiGuard AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Model Prediction ───────────────────────────────────────────
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    return result_index, confidence

CLASS_NAMES = [
    'Apple - Apple Scab','Apple - Black Rot','Apple - Cedar Apple Rust','Apple - Healthy',
    'Blueberry - Healthy','Cherry - Powdery Mildew','Cherry - Healthy',
    'Corn - Cercospora Leaf Spot','Corn - Common Rust','Corn - Northern Leaf Blight','Corn - Healthy',
    'Grape - Black Rot','Grape - Esca (Black Measles)','Grape - Leaf Blight','Grape - Healthy',
    'Orange - Huanglongbing (Citrus Greening)','Peach - Bacterial Spot','Peach - Healthy',
    'Bell Pepper - Bacterial Spot','Bell Pepper - Healthy',
    'Potato - Early Blight','Potato - Late Blight','Potato - Healthy',
    'Raspberry - Healthy','Soybean - Healthy','Squash - Powdery Mildew',
    'Strawberry - Leaf Scorch','Strawberry - Healthy',
    'Tomato - Bacterial Spot','Tomato - Early Blight','Tomato - Late Blight',
    'Tomato - Leaf Mold','Tomato - Septoria Leaf Spot','Tomato - Spider Mites',
    'Tomato - Target Spot','Tomato - Yellow Leaf Curl Virus','Tomato - Mosaic Virus','Tomato - Healthy'
]

DISEASE_INFO = {
    "Apple Scab": {"severity": "Medium", "tip": "Apply fungicide spray and remove infected leaves early."},
    "Black Rot": {"severity": "High", "tip": "Prune infected parts and use copper-based fungicides."},
    "Cedar Apple Rust": {"severity": "Medium", "tip": "Remove nearby cedar trees and apply protective fungicide."},
    "Powdery Mildew": {"severity": "Medium", "tip": "Improve air circulation and use sulfur-based sprays."},
    "Cercospora Leaf Spot": {"severity": "Medium", "tip": "Rotate crops and apply chlorothalonil fungicide."},
    "Common Rust": {"severity": "High", "tip": "Plant resistant varieties and apply fungicide at first sign."},
    "Northern Leaf Blight": {"severity": "High", "tip": "Use disease-resistant hybrids and timely fungicide."},
    "Esca (Black Measles)": {"severity": "High", "tip": "Remove infected wood; no effective chemical cure."},
    "Leaf Blight": {"severity": "Medium", "tip": "Avoid overhead irrigation and apply copper fungicide."},
    "Huanglongbing (Citrus Greening)": {"severity": "Critical", "tip": "No cure; remove infected trees to prevent spread."},
    "Bacterial Spot": {"severity": "High", "tip": "Use copper sprays and avoid working in wet conditions."},
    "Early Blight": {"severity": "Medium", "tip": "Remove lower leaves and apply fungicide preventively."},
    "Late Blight": {"severity": "Critical", "tip": "Apply fungicide immediately and remove infected plants."},
    "Leaf Mold": {"severity": "Medium", "tip": "Improve ventilation and reduce humidity in greenhouse."},
    "Septoria Leaf Spot": {"severity": "Medium", "tip": "Apply fungicide and mulch to reduce soil splash."},
    "Spider Mites": {"severity": "Medium", "tip": "Use miticide sprays or introduce predatory mites."},
    "Target Spot": {"severity": "Medium", "tip": "Apply fungicide and remove heavily infected leaves."},
    "Yellow Leaf Curl Virus": {"severity": "High", "tip": "Control whitefly vectors and remove infected plants."},
    "Mosaic Virus": {"severity": "High", "tip": "Control aphids and remove infected plants immediately."},
    "Leaf Scorch": {"severity": "Medium", "tip": "Improve irrigation management and apply fungicide."},
}

SEVERITY_COLOR = {
    "Critical": "#FF1744",
    "High": "#FF6D00",
    "Medium": "#FFD600",
}

# ─── Global CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --green-deep:   #1B4332;
    --green-mid:    #2D6A4F;
    --green-light:  #52B788;
    --gold:         #D4A017;
    --gold-light:   #F4C842;
    --cream:        #FFF8EE;
    --text-dark:    #1A1A2E;
    --text-muted:   #5a6472;
    --glass:        rgba(255,255,255,0.08);
    --shadow:       0 8px 32px rgba(27,67,50,0.18);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--text-dark);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--green-deep) 0%, #0D2818 100%);
    border-right: none;
    box-shadow: 4px 0 24px rgba(0,0,0,0.2);
}
[data-testid="stSidebar"] * { color: #e8f5e9 !important; }
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin: 0.25rem 0;
    transition: all 0.2s ease;
    cursor: pointer;
    display: block;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(82,183,136,0.25);
    border-color: var(--green-light);
}
[data-testid="stSidebar"] .stInfo {
    background: rgba(212,160,23,0.15) !important;
    border-left: 3px solid var(--gold) !important;
    border-radius: 8px;
    color: #ffe082 !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1); }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, var(--green-deep) 0%, var(--green-mid) 60%, #40916C 100%);
    border-radius: 24px;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
}
.hero-banner::before {
    content: "🌾";
    font-size: 220px;
    position: absolute;
    right: -20px;
    top: -40px;
    opacity: 0.07;
    line-height: 1;
}
.hero-banner::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--gold), var(--gold-light), var(--gold));
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    color: #ffffff;
    line-height: 1.1;
    margin: 0 0 0.5rem 0;
    letter-spacing: -1px;
}
.hero-sub {
    font-size: 1.15rem;
    color: rgba(255,255,255,0.75);
    font-weight: 300;
    margin: 0 0 2rem 0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(212,160,23,0.2);
    border: 1px solid var(--gold);
    color: var(--gold-light);
    padding: 0.35rem 0.9rem;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

/* ── Stat Cards ── */
.stats-row { display: flex; gap: 1rem; margin: 1.5rem 0 2.5rem; flex-wrap: wrap; }
.stat-card {
    flex: 1;
    min-width: 140px;
    background: #ffffff;
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 16px rgba(27,67,50,0.08);
    border-top: 4px solid var(--green-light);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-card:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(27,67,50,0.15); }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--green-deep);
}
.stat-label { font-size: 0.78rem; color: var(--text-muted); font-weight: 500; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Section Heading ── */
.section-head {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 2rem 0 1rem;
}
.section-head h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--green-deep);
    margin: 0;
}
.section-line { flex: 1; height: 2px; background: linear-gradient(90deg, var(--green-light), transparent); border-radius: 2px; }

/* ── How it works ── */
.steps-grid { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
.step-card {
    flex: 1;
    min-width: 160px;
    background: #fff;
    border-radius: 16px;
    padding: 1.5rem 1.2rem;
    text-align: center;
    border: 1px solid rgba(82,183,136,0.2);
    position: relative;
    overflow: hidden;
}
.step-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--green-light), var(--gold));
}
.step-num {
    width: 38px; height: 38px;
    border-radius: 50%;
    background: var(--green-deep);
    color: #fff;
    font-weight: 700;
    font-size: 1rem;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 0.8rem;
}
.step-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.step-title { font-weight: 600; color: var(--green-deep); font-size: 0.9rem; }
.step-desc { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.3rem; line-height: 1.5; }

/* ── Upload Zone ── */
.upload-zone {
    background: linear-gradient(135deg, #f0faf4, #e8f5e9);
    border: 2px dashed var(--green-light);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0 1.5rem;
    text-align: center;
}
.upload-zone-title { font-weight: 600; color: var(--green-mid); font-size: 1rem; margin-bottom: 0.3rem; }
.upload-zone-sub { font-size: 0.8rem; color: var(--text-muted); }

/* ── Image Preview Card ── */
.preview-card {
    background: #fff;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(27,67,50,0.1);
    border: 1px solid rgba(82,183,136,0.15);
}
.preview-header {
    background: var(--green-deep);
    padding: 0.8rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.preview-dot { width: 10px; height: 10px; border-radius: 50%; }

/* ── Result Cards ── */
.result-healthy {
    background: linear-gradient(135deg, #f0faf4, #e8f5e9);
    border-radius: 20px;
    padding: 2rem;
    border-left: 6px solid #2D6A4F;
    box-shadow: 0 4px 20px rgba(27,67,50,0.1);
    text-align: center;
    animation: fadeUp 0.5s ease;
}
.result-disease {
    background: linear-gradient(135deg, #fff8f0, #fff3e0);
    border-radius: 20px;
    padding: 2rem;
    border-left: 6px solid #FF6D00;
    box-shadow: 0 4px 20px rgba(255,109,0,0.12);
    animation: fadeUp 0.5s ease;
}
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; display: block; }

/* ── Detail Row ── */
.detail-row { display: flex; gap: 1rem; margin-top: 1.5rem; flex-wrap: wrap; }
.detail-chip {
    flex: 1;
    min-width: 120px;
    background: #fff;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.detail-chip-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); font-weight: 600; }
.detail-chip-value { font-size: 1.05rem; font-weight: 700; color: var(--green-deep); margin-top: 0.3rem; }

/* ── Severity Badge ── */
.severity-critical { color: #FF1744; font-weight: 700; }
.severity-high { color: #FF6D00; font-weight: 700; }
.severity-medium { color: #F9A825; font-weight: 700; }

/* ── Tip Box ── */
.tip-box {
    background: linear-gradient(135deg, #fffde7, #fff9c4);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    border-left: 4px solid var(--gold);
    margin-top: 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}
.tip-icon { font-size: 1.4rem; flex-shrink: 0; }
.tip-text { font-size: 0.9rem; color: #5D4037; line-height: 1.6; }
.tip-label { font-weight: 700; color: #E65100; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.2rem; }

/* ── Confidence Bar ── */
.conf-wrap { margin: 1.2rem 0 0.3rem; }
.conf-label { display: flex; justify-content: space-between; font-size: 0.82rem; color: var(--text-muted); margin-bottom: 0.4rem; }
.conf-bar-bg { background: #e0e0e0; border-radius: 50px; height: 10px; overflow: hidden; }
.conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, var(--green-mid), var(--green-light));
    transition: width 1s ease;
}

/* ── About Cards ── */
.about-card {
    background: #fff;
    border-radius: 18px;
    padding: 1.8rem;
    box-shadow: 0 2px 16px rgba(27,67,50,0.08);
    border-top: 4px solid var(--green-light);
    margin-bottom: 1rem;
}
.about-card h4 { font-family: 'Playfair Display', serif; color: var(--green-deep); margin: 0 0 0.8rem; font-size: 1.1rem; }
.about-card p, .about-card li { color: var(--text-muted); font-size: 0.9rem; line-height: 1.7; }
.about-card ul { padding-left: 1.2rem; margin: 0; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--green-deep), var(--green-mid)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.75rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 16px rgba(27,67,50,0.25) !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(27,67,50,0.35) !important;
    background: linear-gradient(135deg, #0D2818, var(--green-deep)) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    border: none !important;
}
[data-testid="stFileUploader"] > div {
    background: transparent !important;
    border: none !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--green-mid) !important; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}
.fade-in { animation: fadeUp 0.6s ease; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.8rem 0 0.5rem;">
        <div style="font-size:4rem; line-height:1;">🌾</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:900;
                    color:#ffffff; letter-spacing:-0.5px; margin-top:0.4rem;">KrishiGuard</div>
        <div style="font-size:0.72rem; color:rgba(255,255,255,0.45); letter-spacing:3px;
                    text-transform:uppercase; margin-top:4px;">AI  CROP  GUARD  ·  v2.0</div>
        <div style="margin-top:0.8rem; display:inline-block; background:rgba(212,160,23,0.2);
                    border:1px solid rgba(212,160,23,0.5); border-radius:50px;
                    padding:0.25rem 0.9rem; font-size:0.72rem; color:#ffe082; font-weight:600;">
            ● LIVE
        </div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.1); margin: 1rem 0;">
    """, unsafe_allow_html=True)

    app_mode = st.radio(
        "Navigation",
        ["🏠  Home", "🔍  Disease Finder", "📚  About"],
        label_visibility="hidden"
    )

    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.1); margin: 1.2rem 0 0.8rem;">
    <div style="background:rgba(212,160,23,0.12); border:1px solid rgba(212,160,23,0.4);
                border-radius:10px; padding:0.9rem 1rem; font-size:0.8rem; color:#ffe082;">
        💡 <strong>Pro Tip:</strong> Use high-resolution, well-lit leaf photos for the most accurate diagnosis.
    </div>
    <div style="text-align:center; margin-top:2rem; font-size:0.72rem; color:rgba(255,255,255,0.25);">
        © 2025 KrishiGuard AI<br>Made with ❤️ by Rohit · Pune
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════
if app_mode == "🏠  Home":
    # Hero Banner
    st.markdown("""
    <div class="hero-banner fade-in">
        <div class="hero-title">KrishiGuard AI 🌾</div>
        <div class="hero-sub">AI-powered Crop Disease Detection System — Protect your harvest before it's too late.</div>
        <span class="hero-badge">⚡ Real-time Analysis</span>
        <span class="hero-badge">🎯 98.7% Accuracy</span>
        <span class="hero-badge">🌍 38 Plant Classes</span>
        <span class="hero-badge">📱 Mobile Friendly</span>
    </div>
    """, unsafe_allow_html=True)

    # Visual flow banner (no external image needed)
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1B4332,#2D6A4F,#40916C);
                border-radius:20px; padding:2rem 2.5rem; margin-bottom:1.5rem;
                display:flex; align-items:center; justify-content:center;
                gap:1.2rem; flex-wrap:wrap; box-shadow:0 8px 32px rgba(27,67,50,0.2);">
        <div style="text-align:center; flex:1; min-width:80px;">
            <div style="font-size:3rem; line-height:1;">🌱</div>
            <div style="font-size:0.72rem; color:rgba(255,255,255,0.65); margin-top:0.4rem; font-weight:500;">Seedling</div>
        </div>
        <div style="font-size:1.4rem; color:rgba(255,255,255,0.25); flex-shrink:0;">›</div>
        <div style="text-align:center; flex:1; min-width:80px;">
            <div style="font-size:3rem; line-height:1;">🍃</div>
            <div style="font-size:0.72rem; color:rgba(255,255,255,0.65); margin-top:0.4rem; font-weight:500;">Leaf Scan</div>
        </div>
        <div style="font-size:1.4rem; color:rgba(255,255,255,0.25); flex-shrink:0;">›</div>
        <div style="text-align:center; flex:1; min-width:80px;">
            <div style="font-size:3rem; line-height:1;">🧠</div>
            <div style="font-size:0.72rem; color:rgba(255,255,255,0.65); margin-top:0.4rem; font-weight:500;">AI Analysis</div>
        </div>
        <div style="font-size:1.4rem; color:rgba(255,255,255,0.25); flex-shrink:0;">›</div>
        <div style="text-align:center; flex:1; min-width:80px;">
            <div style="font-size:3rem; line-height:1;">📋</div>
            <div style="font-size:0.72rem; color:rgba(255,255,255,0.65); margin-top:0.4rem; font-weight:500;">Diagnosis</div>
        </div>
        <div style="font-size:1.4rem; color:rgba(255,255,255,0.25); flex-shrink:0;">›</div>
        <div style="text-align:center; flex:1; min-width:80px;">
            <div style="font-size:3rem; line-height:1;">🌾</div>
            <div style="font-size:0.72rem; color:rgba(255,255,255,0.65); margin-top:0.4rem; font-weight:500;">Healthy Crop</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown("""
    <div class="stats-row fade-in">
        <div class="stat-card">
            <div class="stat-num">87K+</div>
            <div class="stat-label">Training Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">38</div>
            <div class="stat-label">Disease Classes</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">98.7%</div>
            <div class="stat-label">Validation Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">&lt; 5s</div>
            <div class="stat-label">Detection Speed</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">16L</div>
            <div class="stat-label">CNN Architecture</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="section-head">
        <h3>🚀 How It Works</h3>
        <div class="section-line"></div>
    </div>
    <div class="steps-grid fade-in">
        <div class="step-card">
            <div class="step-icon">📸</div>
            <div class="step-num">1</div>
            <div class="step-title">Capture</div>
            <div class="step-desc">Take a clear, well-lit photo of the suspect plant leaf</div>
        </div>
        <div class="step-card">
            <div class="step-icon">⬆️</div>
            <div class="step-num">2</div>
            <div class="step-title">Upload</div>
            <div class="step-desc">Visit Disease Finder and submit your leaf image</div>
        </div>
        <div class="step-card">
            <div class="step-icon">🧠</div>
            <div class="step-num">3</div>
            <div class="step-title">AI Analyzes</div>
            <div class="step-desc">Deep CNN model scans 128x128 feature patterns instantly</div>
        </div>
        <div class="step-card">
            <div class="step-icon">📋</div>
            <div class="step-num">4</div>
            <div class="step-title">Get Report</div>
            <div class="step-desc">Receive diagnosis, severity rating and treatment tips</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Supported Crops
    st.markdown("""
    <div class="section-head">
        <h3>🌱 Supported Crops</h3>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    crops = ["🍎 Apple","🫐 Blueberry","🍒 Cherry","🌽 Corn","🍇 Grape",
             "🍊 Orange","🍑 Peach","🫑 Bell Pepper","🥔 Potato",
             "🫒 Raspberry","🫘 Soybean","🥒 Squash","🍓 Strawberry","🍅 Tomato"]
    cols = st.columns(7)
    for i, crop in enumerate(crops):
        with cols[i % 7]:
            st.markdown(f"""
            <div style="background:#fff; border-radius:12px; padding:0.7rem 0.5rem;
                        text-align:center; font-size:0.78rem; font-weight:600;
                        color:var(--green-deep); box-shadow:0 2px 8px rgba(27,67,50,0.08);
                        margin-bottom:0.5rem; border:1px solid rgba(82,183,136,0.15);">
                {crop}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cta = st.columns([1, 2, 1])[1]
    with col_cta:
        if st.button("🔍 Start Disease Detection →"):
            st.info("👈 Select **Disease Finder** from the sidebar to begin!")


# ═══════════════════════════════════════════════════════════════
# DISEASE FINDER PAGE
# ═══════════════════════════════════════════════════════════════
elif app_mode == "🔍  Disease Finder":
    st.markdown("""
    <div class="fade-in">
        <h1 style="font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:900;
                   color:var(--green-deep); margin-bottom:0.3rem;">
            🔍 Crop Disease Finder
        </h1>
        <p style="color:var(--text-muted); font-size:1rem; margin-bottom:2rem;">
            Upload a plant leaf photo and let AI diagnose diseases in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("""
        <div class="section-head">
            <h3>📤 Upload Leaf Image</h3>
            <div class="section-line"></div>
        </div>
        <div class="upload-zone">
            <div class="upload-zone-title">📁 Drag & Drop or Click to Browse</div>
            <div class="upload-zone-sub">Supported: JPG, PNG, JPEG · Max size: 10MB</div>
        </div>
        """, unsafe_allow_html=True)

        test_image = st.file_uploader(
            "Choose plant leaf image",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed",
            help="Upload a clear, single-leaf photo for best accuracy"
        )

        if test_image:
            st.markdown("""
            <div class="section-head" style="margin-top:1.5rem;">
                <h3>📷 Image Preview</h3>
                <div class="section-line"></div>
            </div>
            <div class="preview-card">
                <div class="preview-header">
                    <div class="preview-dot" style="background:#ff5f56;"></div>
                    <div class="preview-dot" style="background:#ffbd2e; margin-left:4px;"></div>
                    <div class="preview-dot" style="background:#27c93f; margin-left:4px;"></div>
                    <span style="color:rgba(255,255,255,0.6); font-size:0.75rem; margin-left:0.7rem;">
                        leaf_sample.jpg
                    </span>
                </div>
            """, unsafe_allow_html=True)
            st.image(test_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("🚀 Analyze Disease Now", type="primary")
        else:
            analyze_btn = False
            # placeholder illustration
            st.markdown("""
            <div style="text-align:center; padding: 3rem 1rem; color:var(--text-muted);">
                <div style="font-size:4rem; margin-bottom:1rem;">🍃</div>
                <div style="font-weight:600; font-size:0.95rem; color:var(--green-mid);">No image uploaded yet</div>
                <div style="font-size:0.82rem; margin-top:0.3rem;">Upload a leaf photo above to begin analysis</div>
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        st.markdown("""
        <div class="section-head">
            <h3>📋 Diagnosis Report</h3>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        if test_image and analyze_btn:
            with st.spinner("🔬 Scanning leaf patterns with AI..."):
                time.sleep(0.8)  # Brief pause for UX polish
                result_index, confidence = model_prediction(test_image)

            diagnosis = CLASS_NAMES[result_index]
            plant, disease = diagnosis.split(" - ")

            if "Healthy" in disease:
                st.markdown(f"""
                <div class="result-healthy fade-in">
                    <span class="result-emoji">🎉</span>
                    <div class="result-title" style="color:#1B4332;">Plant is Healthy!</div>
                    <p style="color:#2D6A4F; font-size:0.95rem; margin:0.5rem 0 0;">
                        Great news! Your <strong>{plant}</strong> appears to be disease-free and thriving.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                severity = DISEASE_INFO.get(disease, {}).get("severity", "Medium")
                tip = DISEASE_INFO.get(disease, {}).get("tip", "Consult your local agronomist.")
                sev_class = f"severity-{severity.lower()}"

                st.markdown(f"""
                <div class="result-disease fade-in">
                    <span class="result-emoji">⚠️</span>
                    <div class="result-title" style="color:#BF360C;">{disease} Detected</div>
                    <p style="color:#5D4037; font-size:0.93rem; margin:0.4rem 0 0;">
                        Possible <strong>{disease}</strong> found in your <strong>{plant}</strong> plant.
                        Act quickly to prevent further spread.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Detail chips
            severity_val = "N/A" if "Healthy" in disease else DISEASE_INFO.get(disease, {}).get("severity", "—")
            sev_color = SEVERITY_COLOR.get(severity_val, "#2D6A4F") if "Healthy" not in disease else "#2D6A4F"

            st.markdown(f"""
            <div class="detail-row">
                <div class="detail-chip">
                    <div class="detail-chip-label">🌿 Plant</div>
                    <div class="detail-chip-value">{plant}</div>
                </div>
                <div class="detail-chip">
                    <div class="detail-chip-label">🦠 Condition</div>
                    <div class="detail-chip-value" style="font-size:0.88rem;">{disease}</div>
                </div>
                <div class="detail-chip">
                    <div class="detail-chip-label">⚡ Severity</div>
                    <div class="detail-chip-value" style="color:{sev_color};">{severity_val}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence Bar
            st.markdown(f"""
            <div class="conf-wrap">
                <div class="conf-label">
                    <span>Model Confidence</span>
                    <span style="font-weight:700; color:var(--green-mid);">{confidence:.1f}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{confidence:.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Treatment Tip
            if "Healthy" not in disease:
                tip_text = DISEASE_INFO.get(disease, {}).get("tip", "Consult your local agronomist for advice.")
                st.markdown(f"""
                <div class="tip-box">
                    <span class="tip-icon">💊</span>
                    <div>
                        <div class="tip-label">Recommended Action</div>
                        <div class="tip-text">{tip_text}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="tip-box">
                    <span class="tip-icon">✅</span>
                    <div>
                        <div class="tip-label">Maintenance Tip</div>
                        <div class="tip-text">Keep up with regular watering, fertilization, and periodic disease monitoring to maintain plant health.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        elif not test_image:
            st.markdown("""
            <div style="text-align:center; padding:3.5rem 1rem; background:#fff;
                        border-radius:20px; border:1px dashed rgba(82,183,136,0.3);">
                <div style="font-size:3.5rem; margin-bottom:1rem; animation: pulse 2s infinite;">🔍</div>
                <div style="font-weight:600; color:var(--green-mid); font-size:1rem;">
                    Awaiting Analysis
                </div>
                <div style="color:var(--text-muted); font-size:0.82rem; margin-top:0.4rem;">
                    Upload a leaf image on the left and click Analyze
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:3.5rem 1rem; background:#fff;
                        border-radius:20px; border:1px dashed rgba(82,183,136,0.3);">
                <div style="font-size:3rem; margin-bottom:1rem;">👆</div>
                <div style="font-weight:600; color:var(--green-mid);">Image Ready!</div>
                <div style="color:var(--text-muted); font-size:0.82rem; margin-top:0.4rem;">
                    Click <strong>Analyze Disease Now</strong> to get your diagnosis
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ABOUT PAGE
# ═══════════════════════════════════════════════════════════════
elif app_mode == "📚  About":
    st.markdown("""
    <div class="fade-in">
        <h1 style="font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:900;
                   color:var(--green-deep); margin-bottom:0.3rem;">
            📚 About KrishiGuard AI
        </h1>
        <p style="color:var(--text-muted); font-size:1rem; margin-bottom:2rem;">
            Built to empower every farmer with the power of artificial intelligence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="about-card fade-in">
            <h4>🎯 Project Mission</h4>
            <p>KrishiGuard AI is an intelligent crop protection system that helps farmers quickly 
            identify plant diseases through leaf image analysis. Early detection means faster 
            intervention and significantly reduced crop losses.</p>
        </div>

        <div class="about-card fade-in">
            <h4>📊 Dataset Information</h4>
            <ul>
                <li>Source: Kaggle Plant Diseases Dataset</li>
                <li>Total Images: 87,000+ RGB images</li>
                <li>Disease Classes: 38 categories</li>
                <li>Train Split: 70,295 images (80%)</li>
                <li>Validation Split: 17,572 images (20%)</li>
                <li>Real-world Test Set: 33 curated images</li>
                <li>Augmentation: Rotation, flip, zoom</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card fade-in">
            <h4>🛠️ Technical Architecture</h4>
            <ul>
                <li><strong>Framework:</strong> TensorFlow 2.0</li>
                <li><strong>Model:</strong> Custom 16-layer CNN</li>
                <li><strong>Input Size:</strong> 128 × 128 pixels</li>
                <li><strong>Training:</strong> 50 epochs, Adam optimizer</li>
                <li><strong>Val Accuracy:</strong> 98.7%</li>
                <li><strong>Inference:</strong> GPU-accelerated &lt; 5 seconds</li>
            </ul>
        </div>

        <div class="about-card fade-in">
            <h4>🌾 Supported Plants (14 Varieties)</h4>
            <p>Apple · Blueberry · Cherry · Corn · Grape · Orange · Peach · 
            Bell Pepper · Potato · Raspberry · Soybean · Squash · Strawberry · Tomato</p>
            <p style="margin-top:0.6rem;">
                Covering <strong>38 unique disease conditions</strong> including bacterial, fungal, 
                viral infections and healthy states.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:2rem; padding:1.5rem;
                background:linear-gradient(135deg, var(--green-deep), var(--green-mid));
                border-radius:20px; color:rgba(255,255,255,0.8); font-size:0.88rem;">
        © 2025 KrishiGuard AI &nbsp;·&nbsp; Developed with ❤️‍🔥 by <strong style="color:#ffe082;">Rohit</strong> in Pune
        &nbsp;·&nbsp; Powered by TensorFlow & Streamlit
    </div>
    """, unsafe_allow_html=True)