# Drone Monitor

Monitoring drone di browser (`http://localhost:8000`): video VRX 5.8 GHz, deteksi objek YOLO (`best.pt`),
telemetry ArduPilot (MAVLink), estimasi lokasi objek (lat/lon), pelacakan objek dengan ID terkunci,
dan export CSV.

```
vrx/
├─ server.py            backend FastAPI (video MJPEG, WebSocket telemetry/perintah, export CSV)
├─ sources/
│  ├─ video.py          kamera VRX + thread inferensi model + anti blue-screen
│  ├─ telemetry.py      MAVLink (auto-cari COM port radio/FC)
│  ├─ geolocate.py      piksel -> lat/lon (FOV, tilt kamera, pose drone)
│  ├─ tracker.py        ID objek terkunci (radius lokasi + kontinuitas kotak)
│  └─ logger.py         log CSV
├─ static/index.html    halaman monitoring
├─ models/best.pt       model YOLO (salin sendiri; TIDAK ikut kalau proyek dibagikan tanpa file ini)
├─ config.json          pengaturan terakhir (telemetry, model, kamera, radius) - dibuat otomatis
├─ tracks.json          objek terlacak tersimpan - dibuat otomatis
├─ logs/                CSV telemetry & estimasi per sesi - dibuat otomatis
├─ requirements.txt
├─ setup.ps1            setup otomatis Windows
└─ run.bat              jalankan server + buka browser
```

---

## Pindah ke device lain: apa yang ikut, apa yang harus disiapkan

| Ikut saat folder disalin | Harus disiapkan di device baru |
|---|---|
| Semua kode (`server.py`, `sources/`, `static/`) | **Python 3.10-3.13** |
| `models/best.pt` (kalau disalin) | **Paket pip** (`setup.ps1` atau manual) - termasuk **torch yang cocok dengan GPU device itu** |
| `config.json` (COM auto, hwid radio, conf, HFOV, tilt, radius) | **Driver CP210x** untuk radio telemetry (COM port tidak muncul tanpa ini) |
| `tracks.json` (objek terlacak) | Colok **VRX** (USB UVC, tidak perlu driver) dan **radio telemetry** |
| `logs/` (log lama) | Buka firewall kalau mau diakses dari HP (`--host 0.0.0.0`) |

Jadi: **salin folder, install Python, dobel-klik `run.bat`** (setup otomatis saat pertama kali). Localhost dan port sama (8000) di device mana pun, apa pun jaringannya.

`config.json` boleh ikut - isinya adaptif (`"mav": "auto"` mencari radio di COM manapun, `hwid` hanya
preferensi). Kalau device baru pakai radio berbeda, otomatis tetap ketemu lewat vendor ID.

---

## Instalasi (Windows)

### Cara cepat: cukup `run.bat`

1. Salin seluruh folder `vrx` ke device baru (flashdisk / zip / git). Pastikan `models\best.pt` ikut.
2. Install **Python 3.11+** dari https://www.python.org/downloads/ - centang **Add python.exe to PATH**.
3. Dobel-klik **`run.bat`**.

`run.bat` memeriksa sendiri: kalau Python sistem sudah punya semua paket -> langsung jalan. Kalau belum
(device baru) -> otomatis menjalankan `setup.ps1`: membuat `.venv`, memasang torch **CUDA** kalau ada GPU
NVIDIA (atau **CPU** kalau tidak), memasang `requirements.txt`, membuat folder `models/` dan `logs/`,
memasang driver CP210x dari folder proyek. Download torch CUDA ~2.5 GB, hanya sekali. Setelah itu server
jalan dan browser terbuka di http://localhost:8000.

Setup manual (opsional): `powershell -ExecutionPolicy Bypass -File setup.ps1` (tambah `-Cpu` untuk paksa
torch CPU). Driver CP210x butuh PowerShell **sebagai Administrator**; kalau tidak, klik kanan
`CP210x_Universal_Windows_Driver\silabser.inf` -> Install.

### Localhost di device / jaringan lain

`localhost` = `127.0.0.1` = komputer itu sendiri. **Tidak bergantung pada jaringan**: WiFi berbeda, LAN
kantor, atau tanpa internet sama sekali, `http://localhost:8000` tetap jalan di device yang menjalankan
`run.bat` (hanya tile peta OpenStreetMap yang butuh internet). Jaringan baru perlu diperhatikan hanya kalau
dashboard mau dibuka dari device lain (HP/laptop kedua): jalankan `run.bat --host 0.0.0.0`, cek IP laptop
dengan `ipconfig`, lalu buka `http://<IP-laptop>:8000` dari device yang ada di WiFi yang sama. Kalau
diblokir, izinkan Python di Windows Firewall (jaringan privat).

### Cara manual (kalau tidak mau pakai skrip)

```powershell
py -m venv .venv
.venv\Scripts\activate

# GPU NVIDIA (RTX 20xx ke atas, driver terbaru):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# atau tanpa GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
python server.py
```

Catatan GPU: RTX 50xx (Blackwell) **wajib** build `cu128` atau lebih baru. RTX 20xx-40xx bisa `cu128`
atau `cu124`. Cek dengan `python -c "import torch; print(torch.cuda.is_available())"`.

### Driver radio telemetry (CP210x)

Radio SiK/3DR/Holybro memakai chip Silicon Labs CP210x. Tanpa driver, Windows menampilkan
"CP2104 USB to UART Bridge Controller" dengan tanda seru dan **tidak ada COM port**.

- Folder `CP210x_Universal_Windows_Driver\` sudah ada di proyek: klik kanan `silabser.inf` -> **Install**
  (atau biarkan `setup.ps1` memasangnya, butuh PowerShell sebagai Administrator).
- Atau download dari https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers.
- Setelah itu di Device Manager -> Ports (COM & LPT) muncul "Silicon Labs CP210x ... (COMx)".

Radio FTDI / CH340 / USB langsung ke Pixhawk biasanya tidak butuh driver tambahan di Windows 10/11.

---

## Menjalankan

```
run.bat                                  # atau: .venv\Scripts\python server.py
py server.py --cam test --mav sim        # uji tanpa hardware
py server.py --host 0.0.0.0              # bisa dibuka dari HP di WiFi yang sama: http://<ip-laptop>:8000
py server.py --port 8080                 # kalau 8000 dipakai aplikasi lain
py server.py --help                      # semua opsi
```

Semua pengaturan (sumber telemetry, model, confidence, HFOV/tilt kamera, radius pelacakan) bisa diubah dari
halaman web tanpa restart dan tersimpan ke `config.json`.

### Pengaturan flight controller (ArduPilot) untuk radio di TELEM1

| Parameter | Nilai |
|---|---|
| `SERIAL1_PROTOCOL` | 2 (MAVLink2) |
| `SERIAL1_BAUD` | 57 (57600, sama dengan radio) |
| `BRD_SER1_RTSCTS` | 0 (radio SiK tidak pakai flow control) |

Radio darat dan udara harus punya `NETID` sama dan `SERIAL_SPEED=57`.

---

## Masalah umum

| Gejala | Penyebab / solusi |
|---|---|
| Browser: "localhost refused to connect" | Server belum jalan. Jalankan `run.bat`, biarkan terminalnya terbuka. |
| VRX "tidak ada" | Colok USB VRX, tutup aplikasi lain yang memakai kamera (OBS, Zoom). Cek `py server.py --cam 1`. |
| Telemetry `waiting` terus | Radio belum tercolok / driver CP210x belum terpasang (lihat di atas). |
| Telemetry `connecting` / tidak ada heartbeat, padahal COM ada | Drone mati, radio udara belum link (LED hijau solid = link), kabel TX/RX ke FC terbalik, atau baud FC tidak 57600. |
| Port COM "Access is denied" | Dipakai Mission Planner. Tutup MP, atau pakai MAVLink Mirror MP -> pilih UDP di web. |
| Model `error: ...` saat load | `ultralytics`/`torch` belum terpasang, atau torch tidak cocok GPU -> jalankan ulang `setup.ps1`. |
| Video blue screen | Sinyal 5.8 GHz hilang; server otomatis menahan frame terakhir dan memberi label merah. |
| Lat/lon objek "belum ada GPS" | Drone belum fix GPS (di dalam ruangan). Untuk uji pakai "pose manual" di kartu Estimasi. |

---

## Data yang dihasilkan

- `tracks.json` - daftar objek terlacak (permanen, hapus lewat tombol di web).
- `logs/telemetry_<sesi>.csv` - telemetry 2,5 Hz. `logs/estimates_<sesi>.csv` - estimasi lokasi tiap deteksi.
- Tombol **Download CSV** ada di kartu Objek terlacak, Telemetry, dan Estimasi lokasi.
