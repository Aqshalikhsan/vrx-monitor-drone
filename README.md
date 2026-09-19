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
├─ setup.ps1 / run.bat  Windows: setup otomatis / jalankan server + buka browser
└─ setup.sh  / run.sh   Linux  : setup otomatis / jalankan server + buka browser
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

# GPU NVIDIA baru (GTX 16xx, RTX 20xx s/d 50xx):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# GPU NVIDIA lama (GTX 750/900/10xx, MX/Quadro lama; compute capability < 7.5):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# tanpa GPU NVIDIA (Intel/AMD/iGPU - jalan di laptop apa pun):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
python server.py
```

Catatan GPU: `setup.ps1`/`setup.sh` memilih build ini otomatis dari `nvidia-smi --query-gpu=compute_cap`.
RTX 50xx (Blackwell) **wajib** `cu128`; GPU dengan compute capability < 7.5 tidak punya kernel di `cu128`
(error `no kernel image is available`) dan butuh `cu118`. Kalau build tidak cocok, server **tidak error**:
deteksi otomatis pindah ke CPU (lihat pesan `[analyzer] ... -> pakai CPU` di konsol) - lebih lambat saja.
Cek: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"`.

### Driver radio telemetry (CP210x)

Radio SiK/3DR/Holybro memakai chip Silicon Labs CP210x. Tanpa driver, Windows menampilkan
"CP2104 USB to UART Bridge Controller" dengan tanda seru dan **tidak ada COM port**.

- Folder `CP210x_Universal_Windows_Driver\` sudah ada di proyek: klik kanan `silabser.inf` -> **Install**
  (atau biarkan `setup.ps1` memasangnya, butuh PowerShell sebagai Administrator).
- Atau download dari https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers.
- Setelah itu di Device Manager -> Ports (COM & LPT) muncul "Silicon Labs CP210x ... (COMx)".

Radio FTDI / CH340 / USB langsung ke Pixhawk biasanya tidak butuh driver tambahan di Windows 10/11.

---

## Instalasi (Linux)

Alurnya sama dengan Windows, skripnya `setup.sh` dan `run.sh`:

```bash
sudo apt install python3 python3-venv python3-pip   # Debian/Ubuntu; Fedora: dnf install python3
chmod +x setup.sh run.sh
./run.sh                                             # setup otomatis saat pertama kali, lalu jalan
```

Setup manual (opsional): `bash setup.sh` (tambah `--cpu` untuk paksa torch CPU). Skrip ini juga:

- memasang `libgl1`/`libglib2.0-0` yang dibutuhkan OpenCV (Debian/Ubuntu);
- memasang torch **CUDA** kalau `nvidia-smi` ada (build sesuai generasi GPU), kalau tidak torch CPU;
- menambahkan user ke grup `dialout` (serial) dan `video` (kamera). **Logout/login sekali** setelahnya.

Driver CP210x / FTDI / CH340 **sudah bawaan kernel Linux**, tidak perlu install apa pun. Radio muncul sebagai
`/dev/ttyUSB0` (cek `ls /dev/ttyUSB*`), pilih dari halaman web atau `./run.sh --mav /dev/ttyUSB0`.
Mode `auto` mendeteksi VID:PID yang sama seperti di Windows.

Masalah khas Linux:

| Gejala | Solusi |
|---|---|
| `/dev/ttyUSB0` tidak muncul padahal radio tercolok | Ubuntu: paket `brltty` merebut CP210x -> `sudo apt remove brltty`, cabut-colok radio. |
| `Permission denied: /dev/ttyUSB0` | Belum di grup `dialout` (atau belum logout/login). Cek `groups`. Darurat: `sudo chmod a+rw /dev/ttyUSB0`. |
| VRX tidak terbuka | `ls /dev/video*`; VRX UVC biasanya bikin 2 device, coba `./run.sh --cam 0` atau `--cam 2`. Tutup aplikasi lain (Cheese, OBS). |
| `libGL.so.1: cannot open shared object` | `sudo apt install libgl1` (skrip setup seharusnya sudah memasangnya). |

---

## Menjalankan

```
run.bat                                  # Windows; atau: .venv\Scripts\python server.py
./run.sh                                 # Linux;   atau: .venv/bin/python server.py
py server.py --cam test --mav sim        # uji tanpa hardware
py server.py --host 0.0.0.0              # bisa dibuka dari HP di WiFi yang sama: http://<ip-laptop>:8000
py server.py --port 8080                 # kalau 8000 dipakai aplikasi lain
py server.py --help                      # semua opsi
```

Semua pengaturan (sumber telemetry, model, confidence, HFOV/tilt kamera, radius pelacakan) bisa diubah dari
halaman web tanpa restart dan tersimpan ke `config.json`.

### Mode tampilan video (kartu "Model deteksi" -> Tampilan)

| Mode | Yang digambar di video |
|---|---|
| **Deteksi** | kotak deteksi + label + estimasi jarak/koordinat (default) |
| **Trace** | heatmap jejak aktivitas objek ala [Ultralytics Heatmap](https://docs.ultralytics.com/guides/heatmaps/): tiap deteksi menambah "panas" di posisinya, makin lama objek berada di satu tempat makin terang warnanya |
| **Trace + Deteksi** | keduanya |

Opsi heatmap: colormap (parula/jet/turbo/...), opasitas, **Fade** = waktu paruh peluruhan jejak dalam detik
(0 = jejak menumpuk terus seperti di video Ultralytics; misal 30 = jejak lama memudar separuh tiap 30 s,
cocok untuk siaran drone yang terus berjalan), dan tombol **Reset** untuk mengosongkan heatmap.
Rekaman video (tombol Rekam) ikut merekam tampilan sesuai mode yang aktif.

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
| Browser: "localhost refused to connect" | Server belum jalan. Jalankan `run.bat` / `./run.sh`, biarkan terminalnya terbuka. |
| VRX "tidak ada" | Colok USB VRX, tutup aplikasi lain yang memakai kamera (OBS, Zoom). Cek `py server.py --cam 1`. |
| Telemetry `waiting` terus | Radio belum tercolok / driver CP210x belum terpasang (lihat di atas). |
| Telemetry `connecting` / tidak ada heartbeat, padahal COM ada | Drone mati, radio udara belum link (LED hijau solid = link), kabel TX/RX ke FC terbalik, atau baud FC tidak 57600. |
| Port COM "Access is denied" | Dipakai Mission Planner. Tutup MP, atau pakai MAVLink Mirror MP -> pilih UDP di web. |
| Model `error: ...` saat load | `ultralytics`/`torch` belum terpasang -> jalankan ulang `setup.ps1` / `setup.sh`. |
| Konsol: `GPU ... tidak didukung torch ... -> pakai CPU` | Deteksi tetap jalan (di CPU). Supaya pakai GPU: jalankan ulang `setup.ps1` / `setup.sh` (memilih build torch sesuai GPU, download ~2.5 GB). |
| Video blue screen | Sinyal 5.8 GHz hilang; server otomatis menahan frame terakhir dan memberi label merah. |
| Lat/lon objek "belum ada GPS" | Drone belum fix GPS (di dalam ruangan). Untuk uji pakai "pose manual" di kartu Estimasi. |

---

## Data yang dihasilkan

- `tracks.json` - daftar objek terlacak (permanen, hapus lewat tombol di web).
- `logs/telemetry_<sesi>.csv` - telemetry 2,5 Hz. `logs/estimates_<sesi>.csv` - estimasi lokasi tiap deteksi.
- Tombol **Download CSV** ada di kartu Objek terlacak, Telemetry, dan Estimasi lokasi.
