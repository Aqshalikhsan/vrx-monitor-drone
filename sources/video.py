"""
Sumber video VRX 5.8GHz (USB UVC) + analisis model (.pt) - dua thread terpisah.

  VideoSource  : baca kamera secepat mungkin, overlay deteksi terakhir, encode JPEG
  Analyzer     : ambil frame TERBARU, jalankan model, simpan deteksi (frame lama dibuang)

Dengan begitu FPS tampilan = FPS kamera, tidak tertahan oleh kecepatan model.
Logika pencarian kamera diambil dari vrx_viewer.py.
"""
import os
import sys
import threading
import time
import cv2

# urutan backend yang dicoba; Windows: MSMF terbukti jalan untuk VRX di vrx_viewer.py, sisanya fallback.
# Linux: V4L2 (device /dev/video*), lalu CAP_ANY.
if sys.platform == "win32":
    BACKENDS = (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)
else:
    BACKENDS = (cv2.CAP_V4L2, cv2.CAP_ANY)


def open_cam(index, width=None, height=None):
    for backend in BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # buffer kecil = latency kecil
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        cap.release()
    return None


class _TestCam:
    """Pengganti VideoCapture untuk tes tanpa VRX: pola bergerak ~30 fps."""

    def __init__(self, w=640, h=480):
        self.w, self.h, self.n = w, h, 0

    def read(self):
        import numpy as np
        time.sleep(1 / 30)
        self.n += 1
        f = np.zeros((self.h, self.w, 3), np.uint8)
        f[:] = (30, 30, 30)
        x = int((self.n * 4) % self.w)
        cv2.circle(f, (x, self.h // 2), 40, (0, 140, 255), -1)
        cv2.putText(f, f"TEST PATTERN  frame {self.n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return True, f

    def release(self):
        pass


class _FileCam:
    """Video rekaman, diputar ulang terus (loop) dengan kecepatan asli."""

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.dt = 1.0 / fps

    def read(self):
        time.sleep(self.dt)
        ok, f = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        return ok, f

    def release(self):
        self.cap.release()


def find_vrx(max_index=6):
    for i in range(max_index):
        cap = open_cam(i)
        if cap:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[video] kamera index {i} aktif ({w}x{h})")
            return i, cap
        print(f"[video] index {i} kosong")
    return None, None


# ----------------------------------------------------------------------------- analyzer
class Analyzer(threading.Thread):
    """
    Thread inferensi. Subclass / ganti `infer()` untuk model lain.
    Deteksi disimpan sebagai list dict: {"box": (x1,y1,x2,y2), "label": str, "conf": float}
    """

    def __init__(self, video, model_path=None, conf=0.4, imgsz=640, device=None, half=True,
                 geolocator=None, tracker=None):
        super().__init__(daemon=True, name="analyzer")
        self.video = video
        self.geolocator = geolocator  # sources.geolocate.Geolocator: beri lat/lon tiap deteksi
        self.tracker = tracker        # sources.tracker.ObjectTracker: ID objek berdasarkan lokasi
        self.model_path = model_path
        self.conf_th = float(conf)   # bisa diubah live lewat set_params()
        self.imgsz = int(imgsz)
        self.device = device         # None = auto (cuda kalau ada), "cpu", "0"
        self.half = half             # FP16 di GPU: ~2x lebih cepat, akurasi praktis sama
        self.model = None
        self.names = {}
        self.status = "off" if not model_path else "loading"  # off | loading | ready | error: ...
        self.device_name = "-"
        self.detections = []
        self.fps = 0.0
        self.latency_ms = 0.0
        self._stop = threading.Event()

    def set_params(self, conf=None, imgsz=None):
        if conf is not None:
            self.conf_th = min(max(float(conf), 0.01), 0.99)
        if imgsz is not None:
            self.imgsz = max(32, int(imgsz) // 32 * 32)  # YOLO butuh kelipatan 32

    def _prec(self):
        """Argumen presisi: ultralytics >= 8.4 pakai quantize=16, versi lama pakai half=True."""
        if not self.half:
            return {}
        if not hasattr(self, "_prec_kw"):
            try:
                from ultralytics.cfg import get_cfg
                self._prec_kw = {"quantize": 16} if hasattr(get_cfg(), "quantize") else {"half": True}
            except Exception:
                self._prec_kw = {"half": True}
        return self._prec_kw

    def load(self):
        if not self.model_path:
            print("[analyzer] tanpa model (pass-through)")
            return
        import numpy as np
        import torch
        from ultralytics import YOLO  # pip install ultralytics
        self.status = "loading"
        self.model = YOLO(self.model_path)
        self.names = self.model.names
        use_cuda = torch.cuda.is_available() and str(self.device) != "cpu"
        if not use_cuda:
            self.half = False
        # warm-up: inferensi pertama lambat (alokasi CUDA / compile), jangan sampai kena frame nyata
        dummy = np.zeros((self.imgsz, self.imgsz, 3), np.uint8)
        self.model.predict(dummy, imgsz=self.imgsz, device=self.device, verbose=False, **self._prec())
        self.device_name = (torch.cuda.get_device_name(0) if use_cuda else "CPU") + (" fp16" if self.half else "")
        self.status = "ready"
        print(f"[analyzer] model {self.model_path} siap ({len(self.names)} kelas) di {self.device_name}")

    def infer(self, frame):
        if self.model is None:
            return []
        res = self.model.predict(frame, conf=self.conf_th, imgsz=self.imgsz,
                                 device=self.device, verbose=False, **self._prec())[0]
        names = res.names
        out = []
        if res.boxes is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                out.append({"box": (x1, y1, x2, y2),
                            "label": names[int(b.cls[0])],
                            "conf": float(b.conf[0])})
        return out

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self.load()
        except Exception as e:
            self.status = f"error: {e}"
            print(f"[analyzer] gagal memuat model: {e}")
            return
        seq = 0
        n, t0 = 0, time.time()
        while not self._stop.is_set():
            seq, frame = self.video.latest_raw(seq, timeout=1.0)  # selalu frame paling baru
            if frame is None:
                continue
            t = time.perf_counter()
            try:
                dets = self.infer(frame)
                if self.geolocator is not None and dets:
                    h, w = frame.shape[:2]
                    self.geolocator.annotate(dets, w, h)
                if self.tracker is not None:
                    self.tracker.update(dets)   # tetap dipanggil saat kosong -> status objek jadi lost
            except Exception as e:
                print(f"[analyzer] error inferensi: {e}")
                dets = []
            self.latency_ms = (time.perf_counter() - t) * 1000
            self.detections = dets
            n += 1
            if time.time() - t0 >= 1.0:
                self.fps = n / (time.time() - t0)
                n, t0 = 0, time.time()


def draw_detections(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        fw = frame.shape[1]
        tid = d.get("track_id")
        head = (f"#{tid} " if tid else "") + f'{d["label"]} {d["conf"]:.2f}'
        (tw, _), _ = cv2.getTextSize(head, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(frame, head, (max(0, min(x1, fw - tw - 4)), max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        g = d.get("geo")
        if g and g.get("ok"):
            # titik acuan (kaki objek) + jarak & koordinat estimasi; teks dijepit agar tidak keluar frame
            cv2.circle(frame, ((x1 + x2) // 2, y2), 4, (0, 200, 255), -1)
            txt = f'{g["dist"]:.0f} m  {g["lat"]:.6f}, {g["lon"]:.6f}'
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(frame, txt, (max(0, min(x1, fw - tw - 4)), min(y2 + 18, frame.shape[0] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)


def blue_ratio(frame):
    """Rasio piksel biru-dominan (0..1). VRX analog menampilkan layar biru saat sinyal hilang."""
    small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA)
    b, g, r = (small[:, :, i].astype(int) for i in range(3))
    blue = (b > 90) & (b > r + 40) & (b > g + 30)
    return float(blue.mean())


# ----------------------------------------------------------------------------- recorder
class Recorder(threading.Thread):
    """
    Rekam video yang tampil di browser (overlay deteksi ikut) ke file dengan fps tetap.
    Frame diambil dari VideoSource.latest_frame() tiap 1/fps detik: kalau kamera lebih lambat
    frame diduplikasi, kalau lebih cepat frame dilewati -> durasi file = durasi nyata.
    Satu Recorder = satu file; buat baru untuk rekaman berikutnya.
    """
    # MPEG-4 (mp4v) selalu ada di FFmpeg bawaan OpenCV; H.264 (avc1) tidak, karena wheel opencv-python
    # tidak membawa OpenH264 -> mencobanya hanya mencetak error. File .mp4 mp4v diputar VLC / Windows Media Player.
    CODECS = (("mp4v", ".mp4"), ("XVID", ".avi"))

    def __init__(self, video, directory, fps=25.0, prefix="rekaman"):
        super().__init__(daemon=True, name="recorder")
        self.video = video
        self.directory = directory
        self.fps = float(fps) if fps and fps > 1 else 25.0
        self.session = time.strftime("%Y%m%d_%H%M%S")
        self.prefix = prefix
        self.path = None           # ditentukan saat frame pertama (butuh ukuran & codec yang berhasil)
        self.frames = 0
        self.started = time.time()
        self.error = None
        self._stop = threading.Event()
        self._writer = None

    @property
    def filename(self):
        return os.path.basename(self.path) if self.path else None

    def seconds(self):
        return self.frames / self.fps

    def size_bytes(self):
        try:
            return os.path.getsize(self.path) if self.path else 0
        except OSError:
            return 0

    def _open(self, w, h):
        os.makedirs(self.directory, exist_ok=True)
        for fourcc, ext in self.CODECS:
            wr = self._try_open(fourcc, ext, w, h)
            if wr is not None:
                return wr
        self.error = "tidak ada codec video yang tersedia"
        return None

    def _try_open(self, fourcc, ext, w, h):
        path = os.path.join(self.directory, f"{self.prefix}_{self.session}{ext}")
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), self.fps, (w, h))
        if wr.isOpened():
            self.path = path
            print(f"[rec] mulai {path} ({w}x{h} @ {self.fps:.0f} fps, codec {fourcc})")
            return wr
        wr.release()
        try:
            os.remove(path)  # VideoWriter kadang meninggalkan file kosong
        except OSError:
            pass
        return None

    def stop(self):
        self._stop.set()

    def run(self):
        dt = 1.0 / self.fps
        nxt = time.perf_counter()
        size = None
        while not self._stop.is_set():
            frame = self.video.latest_frame()
            if frame is not None:
                h, w = frame.shape[:2]
                if self._writer is None:
                    self._writer = self._open(w, h)
                    if self._writer is None:
                        break
                    size = (w, h)
                if (w, h) != size:  # resolusi kamera berubah di tengah rekaman: sesuaikan ke ukuran awal
                    frame = cv2.resize(frame, size)
                self._writer.write(frame)
                self.frames += 1
            nxt += dt
            wait = nxt - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                nxt = time.perf_counter()  # tertinggal (mis. laptop sleep): jangan mengejar
        if self._writer is not None:
            self._writer.release()
            print(f"[rec] selesai {self.path}: {self.frames} frame, {self.seconds():.1f} s, "
                  f"{self.size_bytes() / 1e6:.1f} MB")


# ----------------------------------------------------------------------------- video
class VideoSource(threading.Thread):
    def __init__(self, cam_index=None, width=None, height=None, jpeg_quality=70,
                 hold_on_loss=True, blue_threshold=0.85):
        super().__init__(daemon=True, name="video")
        self.cam_index = cam_index
        self.req_size = (width, height)
        self.jpeg_quality = jpeg_quality
        self.analyzer = None  # di-set dari server kalau ada model
        # blue screen: tahan frame bagus terakhir saat VRX kehilangan sinyal
        self.hold_on_loss = hold_on_loss
        self.blue_threshold = blue_threshold

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._raw = None       # ndarray BGR frame bagus terakhir (untuk analyzer)
        self._raw_seq = 0      # naik hanya saat ada frame bagus baru -> analyzer tidak mengulang frame tahan
        self._jpeg = None      # bytes JPEG frame terakhir + overlay (untuk browser)
        self._frame = None     # ndarray BGR frame terakhir + overlay (untuk recorder)
        self._seq = 0
        self._stop = threading.Event()
        self._last_good = None

        self.connected = False
        self.width = self.height = 0
        self.fps = 0.0
        self.signal_ok = True
        self.signal_lost_since = None
        self.blue = 0.0        # rasio biru frame terakhir (untuk debug/kalibrasi ambang)

    # ---- dipakai server / analyzer ----
    def latest(self, last_seq, timeout=1.0):
        """Tunggu JPEG dengan seq > last_seq. Return (seq, jpeg) atau (last_seq, None) jika timeout."""
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            if self._seq > last_seq:
                return self._seq, self._jpeg
            return last_seq, None

    def latest_raw(self, last_seq, timeout=1.0):
        """Frame bagus terbaru untuk analyzer; frame blue screen tidak pernah dikirim ke sini."""
        with self._cond:
            if self._raw_seq <= last_seq:
                self._cond.wait(timeout)
            if self._raw_seq > last_seq:
                return self._raw_seq, self._raw
            return last_seq, None

    def snapshot(self):
        with self._lock:
            return self._jpeg

    def latest_frame(self):
        """Frame BGR terakhir persis seperti yang tampil di browser (overlay + tag sinyal hilang)."""
        with self._lock:
            return self._frame

    def stop(self):
        self._stop.set()

    # ---- thread ----
    def run(self):
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self.connected = False
                time.sleep(2.0)
                continue
            self.connected = True
            self._loop(cap)
            cap.release()
            self.connected = False

    def _open(self):
        if self.cam_index == "test":
            return _TestCam()
        if isinstance(self.cam_index, str):
            try:
                return _FileCam(self.cam_index)
            except Exception as e:
                print(f"[video] tidak bisa membuka file {self.cam_index}: {e}")
                return None
        if self.cam_index is not None:
            cap = open_cam(self.cam_index, *self.req_size)
            if cap is None:
                print(f"[video] kamera index {self.cam_index} tidak bisa dibuka, coba lagi...")
            return cap
        idx, cap = find_vrx()
        if cap is None:
            print("[video] VRX tidak ditemukan. Pastikan USB tercolok dan tidak dipakai aplikasi lain.")
        return cap

    def _loop(self, cap):
        frames, t0 = 0, time.time()
        fails = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 30:  # ~1.5 s tanpa frame -> anggap putus, buka ulang
                    print("[video] frame gagal terus, buka ulang kamera")
                    return
                time.sleep(0.05)
                continue
            fails = 0
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                self.fps = frames / dt
                frames, t0 = 0, time.time()
            self.height, self.width = frame.shape[:2]

            # ---- deteksi blue screen (sinyal VRX hilang) ----
            self.blue = blue_ratio(frame) if self.hold_on_loss else 0.0
            lost = self.blue >= self.blue_threshold
            if lost:
                if self.signal_lost_since is None:
                    self.signal_lost_since = time.time()
                    print("[video] sinyal hilang (blue screen) -> tahan frame terakhir")
                self.signal_ok = False
                good = False
                if self._last_good is not None:
                    frame = self._last_good.copy()
                # belum pernah ada frame bagus: tampilkan apa adanya, tapi jangan disimpan / dianalisis
            else:
                if self.signal_lost_since is not None:
                    print(f"[video] sinyal kembali setelah {time.time() - self.signal_lost_since:.1f} s")
                self.signal_lost_since = None
                self.signal_ok = True
                self._last_good = frame
                good = True

            raw = frame  # analyzer dapat frame bersih (tanpa overlay)
            if self.analyzer is not None and self.analyzer.detections:
                frame = frame.copy()
                draw_detections(frame, self.analyzer.detections)
            if not good:
                frame = frame if frame is not raw else frame.copy()
                self._draw_hold_tag(frame)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                continue
            with self._cond:
                if good:
                    self._raw = raw
                    self._raw_seq += 1
                self._jpeg = buf.tobytes()
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def _draw_hold_tag(self, frame):
        h, w = frame.shape[:2]
        held = time.time() - (self.signal_lost_since or time.time())
        txt = f"SINYAL HILANG  frame terakhir {held:.1f} s"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x, y = w - tw - 14, 8
        cv2.rectangle(frame, (x - 6, y), (x + tw + 6, y + th + 12), (0, 0, 0), -1)
        cv2.putText(frame, txt, (x, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2)
        cv2.rectangle(frame, (1, 1), (w - 2, h - 2), (60, 60, 255), 3)  # bingkai merah
