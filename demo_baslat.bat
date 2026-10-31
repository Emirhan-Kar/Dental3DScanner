@echo off
REM DentalScanner demo arayuzu - cift tiklayarak baslatin.
REM Tarayicida http://127.0.0.1:8050 otomatik acilir. Kapatmak icin bu pencereyi kapatin.
chcp 65001 >nul
cd /d "%~dp0"
if exist "venv_gpu\Scripts\python.exe" (
    "venv_gpu\Scripts\python.exe" demo\app.py
) else (
    python demo\app.py
)
pause
