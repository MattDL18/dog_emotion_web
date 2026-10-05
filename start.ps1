# start.ps1 - launches the FastAPI backend and Streamlit frontend together
# Run with: powershell -ExecutionPolicy Bypass -File start.ps1

Write-Host ""
Write-Host "Dog Emotion Detector - Group Robaitics" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Check for required model weights
$models = @("models/single_best.pt", "models/localizer.pt", "models/cls_m.pt")
$missing = $models | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Host ""
    Write-Host "WARNING: Missing model weights:" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "   - $_" -ForegroundColor Yellow }
    Write-Host "Download them from the shared Drive and place them in models/" -ForegroundColor Yellow
    Write-Host "The server will still start but hybrid mode may be unavailable." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Write-Host "Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "uvicorn main:app --host 0.0.0.0 --port 8000"

Write-Host "Waiting for backend to load models (3 seconds)..." -ForegroundColor DarkGray
Start-Sleep -Seconds 3

Write-Host "Starting Streamlit frontend on http://localhost:8501 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "streamlit run frontend/app.py --server.port 8501"

Write-Host ""
Write-Host "Both servers are running." -ForegroundColor Cyan
Write-Host "   Backend  -> http://localhost:8000/docs" -ForegroundColor White
Write-Host "   Frontend -> http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "Close the two terminal windows to stop the servers." -ForegroundColor DarkGray
