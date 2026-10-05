"""
Dog Emotion Detector - Streamlit App (Self-contained, no separate backend)
Group Robaitics - MAPUA University
"""

import os
import io
import urllib.request
import threading
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# MODEL DOWNLOAD CONFIG
# Replace these URLs with your actual GitHub Release asset URLs
# after uploading the .pt files to a GitHub Release.
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
MODEL_FILES = {
    "single_best.pt": "https://github.com/MattDL18/dog_emotion_web/releases/download/v1.0/single_best.pt",
    "localizer.pt":   "https://github.com/MattDL18/dog_emotion_web/releases/download/v1.0/localizer.pt",
    "cls_m.pt":       "https://github.com/MattDL18/dog_emotion_web/releases/download/v1.0/cls_m.pt",
    # "cls_s.pt":     "https://...",  # add if you have this one too
}

def download_models():
    """Download all model files if they don't exist locally."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename, url in MODEL_FILES.items():
        dest = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest):
            st.toast(f"⬇️ Downloading {filename}...", icon="⏳")
            urllib.request.urlretrieve(url, dest)

# ---------------------------------------------------------------------------
# MODEL LOADING (cached — runs only once per server session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    from ultralytics import YOLO
    download_models()

    models_path = MODEL_DIR
    single = YOLO(os.path.join(models_path, "single_best.pt"))
    NAMES  = [single.names[i] for i in sorted(single.names)]

    loc = YOLO(os.path.join(models_path, "localizer.pt"))
    CLS_SPECS = [("cls_m.pt", 288), ("cls_s.pt", 384)]
    clfs = [
        (YOLO(os.path.join(models_path, f)), s)
        for f, s in CLS_SPECS
        if os.path.exists(os.path.join(models_path, f))
    ]

    return single, NAMES, loc, clfs

# ---------------------------------------------------------------------------
# INFERENCE HELPERS (same logic as your original main.py)
# ---------------------------------------------------------------------------
LOCK = threading.Lock()

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

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1e-9)

def dedupe(dets, thr=0.7):
    kept = []
    for d in sorted(dets, key=lambda d: -d["confidence"]):
        if all(iou(d["box"], k["box"]) < thr for k in kept):
            kept.append(d)
    return kept

def square_crop(arr, box, margin=0.10):
    H, W = arr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = int(round(max(max(x2 - x1, y2 - y1) * (1 + 2 * margin), 16)))
    a, b = int(round(cx - s / 2)), int(round(cy - s / 2))
    canvas = np.full((s, s, 3), 114, np.uint8)
    xa, ya = max(a, 0), max(b, 0)
    xb, yb = min(a + s, W), min(b + s, H)
    if xb > xa and yb > ya:
        canvas[ya - b:yb - b, xa - a:xb - a] = arr[ya:yb, xa:xb]
    return canvas

def classify(crops, clfs, NAMES):
    outs = []
    for m, size in clfs:
        batch = crops + [np.ascontiguousarray(c[:, ::-1]) for c in crops]
        res = m.predict(batch, imgsz=size, verbose=False)
        p = np.stack([r.probs.data.cpu().numpy() for r in res])
        p = (p[:len(crops)] + p[len(crops):]) / 2
        full = np.zeros((len(crops), len(NAMES)), np.float32)
        for k, n in m.names.items():
            full[:, NAMES.index(n)] = p[:, k]
        outs.append(full)
    return np.mean(outs, axis=0)

def run_single(img, single, NAMES, conf):
    r = single.predict(img, imgsz=640, conf=conf, agnostic_nms=True, verbose=False)[0]
    dets = []
    if r.boxes is not None:
        for j in range(len(r.boxes)):
            dets.append({
                "box":             r.boxes.xyxy[j].cpu().numpy().round(1).tolist(),
                "emotion":         single.names[int(r.boxes.cls[j])],
                "confidence":      round(float(r.boxes.conf[j]), 3),
                "detection_score": None,
                "probabilities":   None,
            })
    return dedupe(dets)

def run_hybrid(img, loc, clfs, NAMES, conf):
    r = loc.predict(img, imgsz=640, conf=conf, verbose=False)[0]
    dets = []
    if r.boxes is not None and len(r.boxes):
        xyxy  = r.boxes.xyxy.cpu().numpy()
        dconf = r.boxes.conf.cpu().numpy()
        probs = classify([square_crop(r.orig_img, b) for b in xyxy], clfs, NAMES)
        for j in range(len(xyxy)):
            c = int(probs[j].argmax())
            dets.append({
                "box":             xyxy[j].round(1).tolist(),
                "emotion":         NAMES[c],
                "confidence":      round(float(probs[j, c]), 3),
                "detection_score": round(float(dconf[j]), 3),
                "probabilities":   {n: round(float(probs[j, i]), 3) for i, n in enumerate(NAMES)},
            })
    return dets

# ---------------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dog Emotion Detector",
    page_icon="🐕",
    layout="wide",
)

st.markdown("""
<style>
h1 { font-size: 2.4rem !important; font-weight: 800 !important; }
div[data-testid="metric-container"] {
    background: #1e1e2e; border: 1px solid #2e2e3e;
    border-radius: 10px; padding: 10px 16px;
}
section[data-testid="stFileUploader"] > div:first-child {
    border: 2px dashed #444466 !important;
    border-radius: 14px !important;
    background: #0f0f1a !important;
}
.result-card {
    background: #1a1a2e; border: 1px solid #2e2e4e;
    border-radius: 14px; padding: 20px 22px; margin-bottom: 14px;
}
.emotion-title { font-size: 1.8rem; font-weight: 700; margin: 0 0 12px 0; }
.badge {
    display: inline-block; background: #2e2e4e; border-radius: 20px;
    padding: 4px 14px; font-size: 0.85rem; margin-right: 8px;
    margin-bottom: 6px; color: #ccc;
}
section[data-testid="stSidebar"] { background: #12121f !important; }
hr { border-color: #2e2e4e !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LOAD MODELS (triggers download on first boot)
# ---------------------------------------------------------------------------
with st.spinner("🔄 Loading models (first launch may take a minute)..."):
    single, NAMES, loc, clfs = load_models()

hybrid_available = bool(clfs)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🐕 Dog Emotion Detector")
    st.markdown("*Group Robaitics · MAPUA*")
    st.divider()

    st.markdown('<span style="color:#2ECC71; font-weight:600;">&#x25CF; Models Loaded</span>', unsafe_allow_html=True)
    st.caption(f"✅ Single model &nbsp;&nbsp; {'✅' if hybrid_available else '❌'} Hybrid model")
    st.divider()

    st.markdown("**Model Pipeline**")
    model_choice = st.selectbox(
        "model_select",
        options=["hybrid", "single"],
        format_func=lambda x: "🔬 Hybrid (two-stage)" if x == "hybrid" else "⚡ Single-stage",
        label_visibility="collapsed",
    )
    if model_choice == "hybrid":
        st.caption("Localizer → crop → classifier. Returns full emotion probabilities.")
    else:
        st.caption("One-shot YOLO. Faster, returns top label only.")
    if model_choice == "hybrid" and not hybrid_available:
        st.warning("Hybrid weights missing.")

    st.divider()
    st.markdown("**Confidence Threshold**")
    conf_thresh = st.slider(
        "conf_slider", min_value=0.05, max_value=0.95,
        value=0.25, step=0.05, label_visibility="collapsed",
    )
    st.caption(f"Detections below **{conf_thresh:.0%}** are ignored.")

# ---------------------------------------------------------------------------
# HEADER + UPLOAD
# ---------------------------------------------------------------------------
st.markdown("# 🐕 Dog Emotion Detector")
st.markdown(
    "<p style='color:#888; margin-top:-12px; margin-bottom:24px;'>"
    "Group Robaitics &nbsp;·&nbsp; MAPUA University</p>",
    unsafe_allow_html=True,
)

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
img_array   = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------
with st.spinner("Running inference..."):
    import time
    t0 = time.perf_counter()
    with LOCK:
        if model_choice == "single":
            detections = run_single(img_array, single, NAMES, conf_thresh)
        else:
            detections = run_hybrid(img_array, loc, clfs, NAMES, conf_thresh)
    inference_ms = round((time.perf_counter() - t0) * 1000)

# ---------------------------------------------------------------------------
# DRAW BOUNDING BOXES
# ---------------------------------------------------------------------------
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
    draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad], fill=color)
    draw.text((x1, y1 - 4), label, fill="white", font=font, anchor="lb")

# ---------------------------------------------------------------------------
# RESULTS LAYOUT
# ---------------------------------------------------------------------------
st.divider()
col_img, col_res = st.columns([3, 2], gap="large")

with col_img:
    caption = (
        f"**{model_choice.capitalize()}** pipeline &nbsp;·&nbsp; "
        f"{inference_ms} ms &nbsp;·&nbsp; "
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
        for det in detections:
            emotion = det["emotion"]
            color   = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)
            emoji   = EMOTION_EMOJI.get(emotion, "")
            st.markdown(
                f"<div class='result-card'>"
                f"<div class='emotion-title' style='color:{color}'>{emoji} {emotion.capitalize()}</div>"
                f"<span class='badge'>Confidence: {det['confidence']:.1%}</span>"
                + (f"<span class='badge'>Detection: {det['detection_score']:.1%}</span>" if det.get("detection_score") else "")
                + "</div>",
                unsafe_allow_html=True,
            )
            if det.get("probabilities"):
                probs    = det["probabilities"]
                emotions = list(probs.keys())
                values   = list(probs.values())
                colors   = [EMOTION_COLORS.get(e, DEFAULT_COLOR) for e in emotions]
                paired   = sorted(zip(values, emotions, colors), reverse=True)
                values, emotions, colors = zip(*paired)
                fig = go.Figure(go.Bar(
                    x=list(emotions), y=list(values),
                    marker_color=list(colors), marker_line_width=0,
                    text=[f"{v:.0%}" for v in values],
                    textposition="outside",
                    textfont=dict(color="white", size=12),
                ))
                fig.update_layout(
                    height=220, margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[0, 1.15], tickformat=".0%", gridcolor="#2e2e4e", color="#888"),
                    xaxis=dict(color="#ccc"),
                    showlegend=False, font=dict(family="sans-serif"),
                )
                st.plotly_chart(fig, use_container_width=True)

with st.expander("Raw response"):
    st.json({
        "model": model_choice,
        "width": W, "height": H,
        "inference_ms": inference_ms,
        "detections": detections,
    })