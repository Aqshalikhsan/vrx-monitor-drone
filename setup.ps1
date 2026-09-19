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

# ---- 3. Torch: CUDA kalau ada GPU NVIDIA (build sesuai generasi GPU), kalau tidak CPU ----
# Laptop tanpa NVIDIA (Intel/AMD/iGPU) -> torch CPU: deteksi tetap jalan, hanya lebih lambat.
Step "PyTorch"
$hasNvidia = (-not $Cpu) -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($hasNvidia) {
    # Build torch dipilih dari compute capability GPU:
    #   >= 7.5 (GTX 16xx, RTX 20xx s/d 50xx)      -> cu128 (torch terbaru)
    #   <  7.5 (GTX 750/900/10xx, Quadro/MX lama) -> cu118 (torch 2.7.1, build terakhir yang masih bawa sm_50-sm_70
    #                                                 di Windows; cu126 Windows hanya sm_61+, cu128 hanya sm_75+)
    # Driver lama tanpa query compute_cap -> dianggap GPU lama -> cu118.
    $cc = 0.0
    try { $cc = [double](((nvidia-smi --query-gpu=compute_cap --format=csv,noheader) | Select-Object -First 1).Trim()) } catch {}
    $idx = if ($cc -ge 7.5) { "cu128" } else { "cu118" }
    Write-Host "GPU NVIDIA terdeteksi (compute capability $cc) -> torch CUDA build $idx (download ~2.5 GB)"
    # torch build lain yang sudah terpasang (mis. cu128 di GPU lama) tidak diganti pip biasa -> lepas dulu
    & $VPY -m pip uninstall -y torch torchvision 2>$null | Out-Null
    & $VPY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/$idx
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
