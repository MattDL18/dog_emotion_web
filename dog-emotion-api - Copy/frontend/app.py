"""
Dog Emotion Detector - Streamlit Frontend
Group Robaitics - MAPUA University
"""

import os
import io
import requests
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# -- Config ------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

EMOTION_COLORS = {
    "angry":   "#E74C3C",
    "happy":   "#2ECC71",
    "relaxed": "#3498DB",
    "sad":     "#9B59B6",
}
EMOTION_EMOJI = {
    "angry":   "😡",
    "happy":   "😊",
    "relaxed": "😌",
    "sad":     "😢",
}
DEFAULT_COLOR = "#F39C12"

# -- Page setup --------------------------------------------------------------
st.set_page_config(
    page_title="Dog Emotion Detector",
    page_icon="🐕",
    layout="wide",
)

# -- Global CSS --------------------------------------------------------------
st.markdown("""
<style>
/* ── Typography ── */
h1 { font-size: 2.4rem !important; font-weight: 800 !important; }

/* ── Metric chips ── */
div[data-testid="metric-container"] {
    background: #1e1e2e;
    border: 1px solid #2e2e3e;
    border-radius: 10px;
    padding: 10px 16px;
}

/* ── Upload area ── */
section[data-testid="stFileUploader"] > div:first-child {
    border: 2px dashed #444466 !important;
    border-radius: 14px !important;
    background: #0f0f1a !important;
}

/* ── Result card ── */
.result-card {
    background: #1a1a2e;
    border: 1px solid #2e2e4e;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
}
.emotion-title {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 12px 0;
}
.badge {
    display: inline-block;
    background: #2e2e4e;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.85rem;
    margin-right: 8px;
    margin-bottom: 6px;
    color: #ccc;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #12121f !important;
}
.status-badge {
    display: inline-block;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 4px;
}
.status-ok  { background: #1a3a2a; color: #2ECC71; border: 1px solid #2ECC71; }
.status-err { background: #3a1a1a; color: #E74C3C; border: 1px solid #E74C3C; }

/* ── Divider ── */
hr { border-color: #2e2e4e !important; }
</style>
""", unsafe_allow_html=True)

# -- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🐕 Dog Emotion Detector")
    st.markdown("*Group Robaitics · MAPUA*")
    st.divider()

    # Backend health
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        hybrid_available = health.get("hybrid", False)
        single_ok = health.get("single", False)

        st.markdown(
            '<span class="status-badge status-ok">&#x25CF; Backend Connected</span>',
            unsafe_allow_html=True,
        )
        s_icon = "✅" if single_ok else "❌"
        h_icon = "✅" if hybrid_available else "❌"
        st.caption(f"{s_icon} Single model &nbsp;&nbsp; {h_icon} Hybrid model")
    except Exception:
        hybrid_available = False
        st.markdown(
            '<span class="status-badge status-err">&#x25CF; Backend Offline</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Start the server: `uvicorn main:app --reload`")

    st.divider()

    # Model selector
    st.markdown("**Model Pipeline**")
    model_choice = st.selectbox(
        "model_select",
        options=["hybrid", "single"],
        format_func=lambda x: (
            "🔬 Hybrid (two-stage)" if x == "hybrid" else "⚡ Single-stage"
        ),
        label_visibility="collapsed",
    )
    if model_choice == "hybrid":
        st.caption("Localizer → crop → classifier. Returns full emotion probabilities.")
    else:
        st.caption("One-shot YOLO-seg. Faster, returns top label only.")

    if model_choice == "hybrid" and not hybrid_available:
        st.warning("Hybrid weights missing on backend.")

    st.divider()

    # Confidence
    st.markdown("**Confidence Threshold**")
    conf_thresh = st.slider(
        "conf_slider",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        label_visibility="collapsed",
    )
    st.caption(f"Detections below **{conf_thresh:.0%}** are ignored.")

# -- Header ------------------------------------------------------------------
st.markdown("# 🐕 Dog Emotion Detector")
st.markdown(
    "<p style='color:#888; margin-top:-12px; margin-bottom:24px;'>"
    "Group Robaitics &nbsp;·&nbsp; MAPUA University</p>",
    unsafe_allow_html=True,
)

# -- Upload ------------------------------------------------------------------
uploaded = st.file_uploader(
    "🐾  Drag and drop a dog photo here, or click to browse  (JPG / PNG, max 10 MB)",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is None:
    st.markdown(
        "<div style='text-align:center; color:#555; margin-top:40px; font-size:1.1rem;'>"
        "Upload a photo above to get started.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

image_bytes = uploaded.read()
pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")

# -- Predict -----------------------------------------------------------------
with st.spinner("Running inference..."):
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            params={"model": model_choice, "conf": conf_thresh},   # query params, not form body
            files={"file": (uploaded.name, image_bytes, uploaded.type)},
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Lost connection to the backend. Is uvicorn still running?")
        st.stop()
    except requests.exceptions.HTTPError:
        st.error(f"Backend error {resp.status_code}: {resp.text}")
        st.stop()

detections = result.get("detections", [])

# -- Draw bounding boxes -----------------------------------------------------
draw = ImageDraw.Draw(pil_image)
W, H = pil_image.size
box_w = max(3, int(min(W, H) / 150))

try:
    font_size = max(16, int(min(W, H) / 35))
    font = ImageFont.truetype("arial.ttf", size=font_size)
except Exception:
    font = ImageFont.load_default()

for det in detections:
    emotion = det["emotion"]
    conf    = det["confidence"]
    x1, y1, x2, y2 = det["box"]
    color = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)

    draw.rectangle([x1, y1, x2, y2], outline=color, width=box_w)
    label = f"{EMOTION_EMOJI.get(emotion, '')} {emotion}  {conf:.0%}"
    bbox  = draw.textbbox((x1, y1 - 4), label, font=font, anchor="lb")
    pad = 4
    draw.rectangle(
        [bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
        fill=color,
    )
    draw.text((x1, y1 - 4), label, fill="white", font=font, anchor="lb")

# -- Results layout ----------------------------------------------------------
st.divider()
col_img, col_res = st.columns([3, 2], gap="large")

with col_img:
    caption = (
        f"**{result['model'].capitalize()}** pipeline &nbsp;·&nbsp; "
        f"{result['inference_ms']} ms &nbsp;·&nbsp; "
        f"{len(detections)} detection{'s' if len(detections) != 1 else ''}"
    )
    st.markdown(caption)
    st.image(pil_image, use_container_width=True)

with col_res:
    if not detections:
        st.markdown(
            "<div style='text-align:center; padding:40px 0; color:#888;'>"
            "<div style='font-size:3rem'>🔍</div>"
            "<div style='font-size:1.1rem; margin-top:8px;'>No dogs detected</div>"
            "<div style='font-size:0.85rem; margin-top:6px;'>Try lowering the confidence threshold<br>or use a clearer photo.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for i, det in enumerate(detections, 1):
            emotion = det["emotion"]
            color   = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)
            emoji   = EMOTION_EMOJI.get(emotion, "")

            st.markdown(
                f"<div class='result-card'>"
                f"<div class='emotion-title' style='color:{color}'>{emoji} {emotion.capitalize()}</div>"
                f"<span class='badge'>Confidence: {det['confidence']:.1%}</span>"
                + (f"<span class='badge'>Detection: {det['detection_score']:.1%}</span>" if det.get('detection_score') else "")
                + f"</div>",
                unsafe_allow_html=True,
            )

            # Probability bar chart (hybrid only)
            if det.get("probabilities"):
                probs = det["probabilities"]
                emotions = list(probs.keys())
                values   = list(probs.values())
                colors   = [EMOTION_COLORS.get(e, DEFAULT_COLOR) for e in emotions]

                # Sort by value descending
                paired = sorted(zip(values, emotions, colors), reverse=True)
                values, emotions, colors = zip(*paired)

                fig = go.Figure(go.Bar(
                    x=list(emotions),
                    y=list(values),
                    marker_color=list(colors),
                    marker_line_width=0,
                    text=[f"{v:.0%}" for v in values],
                    textposition="outside",
                    textfont=dict(color="white", size=12),
                ))
                fig.update_layout(
                    height=220,
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(
                        range=[0, 1.15],
                        tickformat=".0%",
                        gridcolor="#2e2e4e",
                        color="#888",
                    ),
                    xaxis=dict(color="#ccc"),
                    showlegend=False,
                    font=dict(family="sans-serif"),
                )
                st.plotly_chart(fig, use_container_width=True)

# -- Raw JSON (collapsed) ----------------------------------------------------
with st.expander("Raw API response"):
    st.json(result)
