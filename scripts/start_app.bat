@echo off
setlocal
start "ML-Backend :8000" cmd /k "cd /d E:\Opencode-Project\ml-platform\backend && python -m uvicorn app.main:app --port 8000"
start "ML-Frontend :5173" cmd /k "cd /d E:\Opencode-Project\ml-platform\frontend && npm run dev"
endlocal
