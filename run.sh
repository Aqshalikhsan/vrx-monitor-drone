#!/usr/bin/env bash
# Drone Monitor - jalankan server di Linux. Otomatis setup (venv + requirements) kalau belum lengkap.
cd "$(dirname "$(readlink -f "$0")")"
CHECK='import cv2, fastapi, uvicorn, pymavlink, serial, torch, ultralytics'

# 1) sudah ada .venv?  pakai itu
if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
# 2) belum ada .venv: Python sistem sudah lengkap paketnya?  pakai itu
elif python3 -c "$CHECK" >/dev/null 2>&1; then
    PY=python3
# 3) belum lengkap: setup otomatis (buat .venv, install torch + requirements)
else
    echo "Paket belum lengkap. Menjalankan setup (sekali saja, download torch bisa ~2.5 GB)..."
    bash setup.sh || true
    if [ ! -x .venv/bin/python ]; then
        echo
        echo "Setup gagal: .venv tidak terbentuk. Pastikan Python 3.10+ dan python3-venv terpasang."
        exit 1
    fi
    PY=.venv/bin/python
fi

if ! "$PY" -c "$CHECK" >/dev/null 2>&1; then
    echo "Paket di .venv belum lengkap. Memasang requirements..."
    bash setup.sh
    "$PY" -c "$CHECK" >/dev/null 2>&1 || { echo "Masih gagal. Lihat pesan error di atas."; exit 1; }
fi

[ -f models/best.pt ] || echo "PERHATIAN: models/best.pt belum ada, deteksi tidak aktif sampai model disalin."
echo "Membuka http://localhost:8000 ..."
( sleep 2; xdg-open "http://localhost:8000" >/dev/null 2>&1 ) &
exec "$PY" server.py "$@"
