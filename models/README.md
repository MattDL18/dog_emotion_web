# models/

This folder holds the YOLO `.pt` weight files.  
These files are **excluded from git** (see `.gitignore`) because they are large binaries.

## Download from Google Drive

Get the files from the shared Drive link (ask the model training teammate):

| File | Description | Size (approx) |
|------|-------------|---------------|
| `single_best.pt` | One-stage YOLO-seg (detect + classify) | ~207 MB |
| `localizer.pt` | Hybrid stage 1 — dog localizer | ~207 MB |
| `cls_m.pt` | Hybrid stage 2 — emotion classifier 288 px | ~79 MB |

Place all three files directly in this `models/` folder before starting the server.

## Verify

After placing the files, run:

```bash
python -c "from pathlib import Path; [print(p.name, p.stat().st_size//1e6, 'MB') for p in Path('models').glob('*.pt')]"
```
