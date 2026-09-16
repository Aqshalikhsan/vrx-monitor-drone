# Setup Drone Monitor di Windows. Jalankan sekali di device baru:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
# Opsi:  -Cpu   paksa torch versi CPU (tanpa GPU NVIDIA)
param([switch]$Cpu)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

# ---- 1. Python ----
Step "Cek Python"
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python -ErrorAction SilentlyContinue }
if (-not $py) {
    Write-Host "Python tidak ditemukan. Install Python 3.11+ dari https://www.python.org/downloads/ (centang 'Add to PATH'), lalu jalankan lagi." -ForegroundColor Red
    exit 1
}
$PY = $py.Source
& $PY --version

# ---- 2. Virtual environment (opsional tapi disarankan) ----
Step "Virtual environment .venv"
if (-not (Test-Path ".venv")) { & $PY -m venv .venv }
$VPY = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $VPY -m pip install --upgrade pip --quiet

# ---- 3. Torch: CUDA kalau ada GPU NVIDIA, kalau tidak CPU ----
Step "PyTorch"
$hasNvidia = (-not $Cpu) -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($hasNvidia) {
    Write-Host "GPU NVIDIA terdeteksi -> torch CUDA 12.8 (download ~2.5 GB)"
    & $VPY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "Tanpa GPU NVIDIA -> torch CPU (deteksi lebih lambat, ~5-10 fps untuk model nano)"
    & $VPY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

# ---- 4. Paket lain ----
Step "Paket aplikasi"
& $VPY -m pip install -r requirements.txt

# ---- 5. Folder & cek model ----
Step "Folder"
foreach ($d in "models", "logs") { if (-not (Test-Path $d)) { New-Item -ItemType Directory $d | Out-Null } }
if (-not (Test-Path "models\best.pt")) {
    Write-Host "PERHATIAN: models\best.pt belum ada. Salin model YOLO kamu ke folder models\ (bisa juga dipilih dari web)." -ForegroundColor Yellow
}

# ---- 6. Driver radio telemetry (CP210x) ----
Step "Driver CP210x (radio telemetry SiK / 3DR / Holybro)"
$drv = pnputil /enum-drivers 2>$null | Select-String -Pattern "silabser" -Quiet
if ($drv) {
    Write-Host "Driver CP210x sudah terpasang."
} elseif (Test-Path "CP210x_Universal_Windows_Driver\silabser.inf") {
    Write-Host "Memasang driver CP210x dari folder proyek (butuh hak admin)..."
    try { pnputil /add-driver "CP210x_Universal_Windows_Driver\silabser.inf" /install }
    catch { Write-Host "Gagal (bukan admin?). Klik kanan silabser.inf -> Install, atau jalankan PowerShell sebagai Administrator." -ForegroundColor Yellow }
} else {
    Write-Host "Driver belum ada. Download 'CP210x Universal Windows Driver' dari https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers lalu klik kanan silabser.inf -> Install." -ForegroundColor Yellow
}

# ---- 7. Verifikasi ----
Step "Verifikasi"
& $VPY -c "import cv2, fastapi, uvicorn, pymavlink, serial, torch; print('OpenCV', cv2.__version__, '| torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
& $VPY -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

Write-Host "`nSelesai. Jalankan dengan:  run.bat   (atau: .venv\Scripts\python server.py)" -ForegroundColor Green
Write-Host "Lalu buka http://localhost:8000"
