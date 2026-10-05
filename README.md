# Dog Emotion API — Group Robaitics

FastAPI backend that serves two versions of a YOLO-based dog emotion detector (angry · happy · relaxed · sad).

| Version | Description |
|---------|-------------|
| **single** | One-stage YOLO-seg model that detects dogs *and* classifies emotions in one pass |
| **hybrid** | Two-stage pipeline: YOLO localizer → crop → classifier(s) |

---

## Project layout

```
dog-emotion-api/
├── main.py              ← FastAPI app (both pipeline versions)
├── requirements.txt     ← Python dependencies
├── models/              ← Put your .pt files here (NOT in git – download from Drive)
│   ├── single_best.pt   ← single-stage weights
│   ├── localizer.pt     ← hybrid stage 1
│   └── cls_m.pt         ← hybrid stage 2 classifier (288 px)
├── frontend/            ← Drop the front-end files here
│   └── ...
├── .env.example         ← environment variable template
└── README.md
```

---

## Quick start

### 1. Get the model weights

The `.pt` files are **not in this repo** (they are ~200 MB each).  
Download them from the shared Google Drive folder and place them in `models/`.

> **Required files**
> - `models/single_best.pt`
> - `models/localizer.pt`
> - `models/cls_m.pt`

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API endpoints

### `GET /health`

Returns which models are loaded.

```json
{
  "status": "ok",
  "single": true,
  "hybrid": true,
  "classes": ["angry", "happy", "relaxed", "sad"]
}
```

### `POST /predict`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | image | — | JPEG / PNG, max 10 MB |
| `model` | string | `"hybrid"` | `"single"` or `"hybrid"` |
| `conf` | float | `0.25` | Detection confidence threshold |

**Example with curl**

```bash
# health check
curl http://localhost:8000/health

# single-model prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@dog.jpg" \
  -F "model=single"

# hybrid prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@dog.jpg" \
  -F "model=hybrid"
```

**Response**

```json
{
  "model": "hybrid",
  "width": 1080,
  "height": 1080,
  "inference_ms": 142,
  "detections": [
    {
      "box": [120.5, 80.1, 900.3, 950.7],
      "emotion": "happy",
      "confidence": 0.823,
      "detection_score": 0.941,
      "probabilities": {
        "angry": 0.031,
        "happy": 0.823,
        "relaxed": 0.112,
        "sad": 0.034
      }
    }
  ]
}
```

> The `single` model returns `detection_score: null` and `probabilities: null` — it only outputs the top label.

---

## Front-end integration

The front end is in `frontend/`.  
The API allows requests from any origin (`Access-Control-Allow-Origin: *`), so any browser-based app can call it directly during development.

When submitting a photo, pass the `model` field in the `FormData` to let the user choose which pipeline to use:

```js
const form = new FormData();
form.append("file", imageBlob);
form.append("model", "hybrid"); // or "single"
const res = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: form,
});
const data = await res.json();
console.log(data.detections[0].emotion);
```

---

## Environment variables

Copy `.env.example` to `.env` and adjust as needed.

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Port to bind uvicorn |
| `HOST` | `0.0.0.0` | Host to bind uvicorn |
| `MODELS_DIR` | `models` | Path to the folder with .pt files |

---

## Model performance (validation set, repaired labels)

| Metric | single_best.pt |
|--------|---------------|
| Box mAP50 | 0.858 |
| Mask mAP50 | 0.841 |

Per-class mask AP50 (val):

| Class | val | test |
|-------|-----|------|
| angry | 0.891 | 0.943 |
| happy | 0.849 | 0.855 |
| relaxed | 0.798 | 0.736 |
| sad | 0.825 | 0.715 |

> ⚠️ The 0.90 mAP50 target was not reached. See the project report for the root-cause analysis (label filter bug) and its fix.

---

## Team

**Group Robaitics** — MAPUA University · 7th Semester AI2

---

## License

For academic use only.
