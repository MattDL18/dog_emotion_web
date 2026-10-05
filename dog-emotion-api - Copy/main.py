import time, threading
from pathlib import Path
import numpy as np, cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

MODELS = Path("models")
LOCK = threading.Lock()          # one inference at a time (models aren't guaranteed thread-safe)

# ---- Version A: original single model ----
single = YOLO(str(MODELS / "single_best.pt"))
NAMES = [single.names[i] for i in sorted(single.names)]

# ---- Version B: two-stage pipeline ----
loc = YOLO(str(MODELS / "localizer.pt"))
CLS_SPECS = [("cls_m.pt", 288), ("cls_s.pt", 384)]
clfs = [(YOLO(str(MODELS / f)), s) for f, s in CLS_SPECS if (MODELS / f).is_file()]
for m, _ in clfs:
    assert set(m.names.values()) == set(NAMES), f"Class names differ: {m.names} vs {NAMES}"

app = FastAPI(title="Dog Emotion API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------- helpers ----------------
def read_image(file: UploadFile):
    data = file.file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 10 MB)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not a valid image")
    return img

def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1e-9)

def dedupe(dets, thr=0.7):
    """Keep one emotion per dog: highest confidence wins among heavily overlapping boxes."""
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
    xa, ya, xb, yb = max(a, 0), max(b, 0), min(a + s, W), min(b + s, H)
    if xb > xa and yb > ya:
        canvas[ya - b:yb - b, xa - a:xb - a] = arr[ya:yb, xa:xb]
    return canvas

def classify(crops):
    outs = []
    for m, size in clfs:
        batch = crops + [np.ascontiguousarray(c[:, ::-1]) for c in crops]   # + mirrored copies
        res = m.predict(batch, imgsz=size, verbose=False)
        p = np.stack([r.probs.data.cpu().numpy() for r in res])
        p = (p[:len(crops)] + p[len(crops):]) / 2
        full = np.zeros((len(crops), len(NAMES)), np.float32)
        for k, n in m.names.items():
            full[:, NAMES.index(n)] = p[:, k]
        outs.append(full)
    return np.mean(outs, axis=0)

# ---------------- the two versions ----------------
def run_single(img, conf):
    r = single.predict(img, imgsz=640, conf=conf, agnostic_nms=True, verbose=False)[0]
    dets = []
    if r.boxes is not None:
        for j in range(len(r.boxes)):
            dets.append({
                "box": r.boxes.xyxy[j].cpu().numpy().round(1).tolist(),
                "emotion": single.names[int(r.boxes.cls[j])],
                "confidence": round(float(r.boxes.conf[j]), 3),
                "detection_score": None,
                "probabilities": None,          # the single model gives only its top label
            })
    return dedupe(dets)

def run_hybrid(img, conf):
    r = loc.predict(img, imgsz=640, conf=conf, verbose=False)[0]
    dets = []
    if r.boxes is not None and len(r.boxes):
        xyxy = r.boxes.xyxy.cpu().numpy(); dconf = r.boxes.conf.cpu().numpy()
        probs = classify([square_crop(r.orig_img, b) for b in xyxy])
        for j in range(len(xyxy)):
            c = int(probs[j].argmax())
            dets.append({
                "box": xyxy[j].round(1).tolist(),
                "emotion": NAMES[c],
                "confidence": round(float(probs[j, c]), 3),
                "detection_score": round(float(dconf[j]), 3),
                "probabilities": {n: round(float(probs[j, i]), 3) for i, n in enumerate(NAMES)},
            })
    return dets

# ---------------- endpoints ----------------
@app.get("/health")
def health():
    return {"status": "ok", "single": True, "hybrid": bool(clfs), "classes": NAMES}

@app.post("/predict")
def predict(file: UploadFile = File(...), model: str = "hybrid", conf: float = 0.25):
    if model not in ("single", "hybrid"):
        raise HTTPException(400, "model must be 'single' or 'hybrid'")
    if model == "hybrid" and not clfs:
        raise HTTPException(503, "Hybrid classifier weights are not installed")
    img = read_image(file)
    t0 = time.perf_counter()
    with LOCK:
        dets = run_single(img, conf) if model == "single" else run_hybrid(img, conf)
    return {"model": model, "width": img.shape[1], "height": img.shape[0],
            "inference_ms": round((time.perf_counter() - t0) * 1000), "detections": dets}