@echo off
rem Drone Monitor - jalankan server. Otomatis setup (venv + requirements) kalau belum lengkap.
setlocal
cd /d "%~dp0"
set CHECK=import cv2, fastapi, uvicorn, pymavlink, serial, torch, ultralytics

rem 1) sudah ada .venv?  pakai itu
if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
    goto verify
)

rem 2) belum ada .venv: Python sistem sudah lengkap paketnya?  pakai itu
where py >nul 2>nul && (py -c "%CHECK%" >nul 2>nul && (set PY=py& goto run))
where python >nul 2>nul && (python -c "%CHECK%" >nul 2>nul && (set PY=python& goto run))

rem 3) belum lengkap: setup otomatis (buat .venv, install torch + requirements, driver CP210x)
echo Paket belum lengkap. Menjalankan setup (sekali saja, download torch bisa ~2.5 GB)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Setup gagal: .venv tidak terbentuk. Pastikan Python 3.10+ terpasang dan ada di PATH.
    pause
    exit /b 1
)
set PY=.venv\Scripts\python.exe

:verify
%PY% -c "%CHECK%" >nul 2>nul
if errorlevel 1 (
    echo Paket di .venv belum lengkap. Memasang requirements...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    %PY% -c "%CHECK%" >nul 2>nul || (echo Masih gagal. Lihat pesan error di atas.& pause& exit /b 1)
)

:run
if not exist "models\best.pt" echo PERHATIAN: models\best.pt belum ada, deteksi tidak aktif sampai model disalin.
echo Membuka http://localhost:8000 ...
start "" "http://localhost:8000"
%PY% server.py %*
pause
