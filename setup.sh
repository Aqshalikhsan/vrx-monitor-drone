#!/usr/bin/env bash
# Setup Drone Monitor di Linux (Debian/Ubuntu, Fedora, Arch). Jalankan sekali di device baru:
#   bash setup.sh
# Opsi:  --cpu   paksa torch versi CPU (tanpa GPU NVIDIA)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

CPU=0
for a in "$@"; do
    case "$a" in
        --cpu) CPU=1 ;;
        -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
        *) echo "Opsi tidak dikenal: $a"; exit 1 ;;
    esac
done

step() { printf '\n\033[36m== %s\033[0m\n' "$1"; }
warn() { printf '\033[33m%s\033[0m\n' "$1"; }
err()  { printf '\033[31m%s\033[0m\n' "$1"; }

# ---- 1. Python ----
step "Cek Python"
PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
    err "Python tidak ditemukan. Install dulu:"
    err "  Debian/Ubuntu : sudo apt install python3 python3-venv python3-pip"
    err "  Fedora        : sudo dnf install python3"
    err "  Arch          : sudo pacman -S python"
    exit 1
fi
"$PY" --version
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || { err "Butuh Python 3.10+."; exit 1; }

# ---- 2. Paket sistem yang dibutuhkan OpenCV & venv ----
step "Paket sistem"
if command -v apt-get >/dev/null; then
    MISSING=""
    "$PY" -c 'import venv, ensurepip' 2>/dev/null || MISSING="$MISSING python3-venv"
    ldconfig -p | grep -q libGL.so.1     || MISSING="$MISSING libgl1"
    ldconfig -p | grep -q libglib-2.0.so || MISSING="$MISSING libglib2.0-0"
    if [ -n "$MISSING" ]; then
        echo "Memasang:$MISSING (butuh sudo)"
        sudo apt-get install -y $MISSING
    else
        echo "Sudah lengkap."
    fi
elif command -v dnf >/dev/null; then
    ldconfig -p | grep -q libGL.so.1 || sudo dnf install -y mesa-libGL
    echo "Sudah lengkap."
elif command -v pacman >/dev/null; then
    ldconfig -p | grep -q libGL.so.1 || sudo pacman -S --noconfirm --needed mesa
    echo "Sudah lengkap."
else
    warn "Distro tidak dikenal. Pastikan libGL (untuk OpenCV) dan modul venv Python terpasang."
fi

# ---- 3. Virtual environment ----
step "Virtual environment .venv"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
VPY="$PWD/.venv/bin/python"
"$VPY" -m pip install --upgrade pip --quiet

# ---- 4. Torch: CUDA kalau ada GPU NVIDIA, kalau tidak CPU ----
step "PyTorch"
if [ "$CPU" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU NVIDIA terdeteksi -> torch CUDA 12.8 (download ~2.5 GB)"
    "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
    echo "Tanpa GPU NVIDIA -> torch CPU (deteksi lebih lambat, ~5-10 fps untuk model nano)"
    "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# ---- 5. Paket lain ----
step "Paket aplikasi"
"$VPY" -m pip install -r requirements.txt

# ---- 6. Folder & cek model ----
step "Folder"
mkdir -p models logs
[ -f models/best.pt ] || warn "PERHATIAN: models/best.pt belum ada. Salin model YOLO kamu ke folder models/ (bisa juga dipilih dari web)."

# ---- 7. Akses serial (radio telemetry) & kamera ----
# Driver CP210x/FTDI/CH340 sudah bawaan kernel Linux; yang perlu hanya izin ke /dev/ttyUSB* & /dev/video*.
step "Izin serial & kamera"
NEED_RELOGIN=0
for grp in dialout uucp video; do
    getent group "$grp" >/dev/null || continue
    if id -nG "$USER" | tr ' ' '\n' | grep -qx "$grp"; then
        echo "User $USER sudah di grup $grp."
    else
        echo "Menambahkan $USER ke grup $grp (butuh sudo)..."
        sudo usermod -aG "$grp" "$USER" && NEED_RELOGIN=1
    fi
done
if command -v brltty >/dev/null 2>&1; then
    warn "brltty terpasang: di Ubuntu ini sering merebut CP210x sehingga /dev/ttyUSB tidak muncul."
    warn "Kalau radio tidak terdeteksi:  sudo apt remove brltty"
fi

# ---- 8. Verifikasi ----
step "Verifikasi"
"$VPY" -c "import cv2, fastapi, uvicorn, pymavlink, serial, torch; print('OpenCV', cv2.__version__, '| torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
"$VPY" -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

chmod +x run.sh 2>/dev/null || true
printf '\n\033[32mSelesai. Jalankan dengan:  ./run.sh   (atau: .venv/bin/python server.py)\033[0m\n'
echo "Lalu buka http://localhost:8000"
[ "$NEED_RELOGIN" -eq 1 ] && warn "Logout/login (atau reboot) dulu supaya izin grup serial/kamera aktif."
exit 0
