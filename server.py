"""
Server monitoring drone: video VRX (+ deteksi model .pt) dan telemetry ArduPilot,
ditampilkan di browser lewat http://localhost:8000

Pakai:
    py server.py                              # cam auto, telemetry AUTO: cari radio/FC di COM manapun
    py server.py --mav COM5                   # paksa COM tertentu, baud 57600
    (sumber telemetry juga bisa diganti kapan saja dari halaman web: serial/radio, UDP, sim)
    py server.py --mav COM5 --baud 115200
    py server.py --mav udpin:0.0.0.0:14550    # forward dari Mission Planner (MAVLink Mirror) / SITL
    py server.py --cam 1 --model best.pt --conf 0.5 --device 0
    (model, confidence, imgsz juga bisa diatur dari halaman web tanpa restart)
    py server.py --no-model                   # matikan analisis

Endpoint:
    /            halaman monitoring
    /video       stream MJPEG (multipart/x-mixed-replace)
    /snapshot    JPEG frame terakhir
    /ws          WebSocket: JSON telemetry + status video/model, 10 Hz
    /api/state   JSON sekali (untuk debug / curl)
    /export/tracks.csv      objek terlacak (dibuat saat diminta)
    /export/telemetry.csv   log telemetry sesi ini (2 Hz, folder logs/)
    /export/estimates.csv   log estimasi lokasi tiap deteksi sesi ini (maks 5 Hz, folder logs/)
"""
import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sources import logger as csvlog
from sources.geolocate import Geolocator
from sources.telemetry import TelemetrySource, list_ports, snapshot_empty
from sources.tracker import ObjectTracker
from sources.video import Analyzer, VideoSource

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "best.pt")  # taruh best.pt kamu di sini
CONFIG_PATH = os.path.join(HERE, "config.json")           # pilihan telemetry terakhir

video: VideoSource = None
ARGS = None


def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(**kw):
    """Update sebagian config.json (gabung dengan yang sudah ada)."""
    cfg = load_config()
    cfg.update(kw)
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[server] gagal simpan config: {e}")


MODELS_DIR = os.path.join(HERE, "models")
MODEL_EXTS = (".pt", ".onnx", ".engine")


def list_models():
    """Model yang tersedia di folder models/ (nama file saja)."""
    try:
        return sorted(f for f in os.listdir(MODELS_DIR) if f.lower().endswith(MODEL_EXTS))
    except FileNotFoundError:
        return []


def _det_json(d):
    out = {"label": d["label"], "conf": round(d["conf"], 2), "box": list(d["box"]),
           "track_id": d.get("track_id")}
    g = d.get("geo")
    if g:
        out["geo"] = {"ok": g["ok"], "reason": g.get("reason", "")}
        if g["ok"]:
            out["geo"].update(lat=round(g["lat"], 7), lon=round(g["lon"], 7), dist=round(g["dist"], 1),
                              bearing=round(g["bearing"], 1), err_m=round(g["err_m"], 1),
                              depression=round(g["depression"], 1))
    return out


# estimasi lokasi objek: pose dari telemetry, parameter kamera dari config (bisa diubah dari web)
geo = Geolocator(pose_fn=lambda: telemetry.snapshot() if telemetry else {},
                 cam=load_config().get("geo", {}).get("cam"),
                 manual=load_config().get("geo", {}).get("manual"))


# pelacak objek: deteksi dalam radius R dari lokasi pertama = objek yang sama
_tcfg = load_config().get("track", {})
TRACKS_PATH = os.path.join(HERE, "tracks.json")   # daftar objek terlacak, permanen antar restart
LOGS_DIR = os.path.join(HERE, "logs")
telem_log: csvlog.CsvLog = None
est_log: csvlog.CsvLog = None


async def log_sampler():
    """Catat telemetry 2 Hz dan estimasi deteksi maks 5 Hz ke CSV sesi ini."""
    last_dets_id = None
    tick = 0
    while True:
        try:
            now = time.time()
            st = build_state()
            a = model.analyzer
            dets = a.detections if a else []
            if dets and id(dets) != last_dets_id:          # hanya frame analisis yang baru
                last_dets_id = id(dets)
                for row in csvlog.estimate_rows(st["model"]["detections"], st["geo"], now):
                    est_log.add(row)
            if tick % 2 == 0:                               # 0.2 s * 2 = 0.4 s ~ 2.5 Hz
                telem_log.add(csvlog.telemetry_row(st, now))
            tick += 1
        except Exception as e:
            print(f"[log] error: {e}")
        await asyncio.sleep(0.2)
tracker = ObjectTracker(radius_m=_tcfg.get("radius_m", 3.0), lost_s=_tcfg.get("lost_s", 2.0),
                        expire_s=_tcfg.get("expire_s", 0), store_path=TRACKS_PATH)


def track_set(cmd):
    tracker.set(cmd.get("radius_m"), cmd.get("lost_s"), cmd.get("expire_s"))
    save_config(track={"radius_m": tracker.radius_m, "lost_s": tracker.lost_s, "expire_s": tracker.expire_s})
    return "ok"


def geo_set(cmd):
    kw = {k: cmd[k] for k in ("hfov", "vfov", "tilt", "pan", "use_attitude", "anchor") if k in cmd}
    if "manual" in cmd:
        m = cmd["manual"]
        if m:
            try:
                m = {"lat": float(m["lat"]), "lon": float(m["lon"]),
                     "alt": float(m["alt"]), "heading": float(m.get("heading", 0))}
            except (KeyError, TypeError, ValueError):
                return "pose manual butuh lat, lon, alt, heading (angka)"
        kw["manual"] = m or None
    geo.set(**kw)
    snap = geo.snapshot()
    save_config(geo={"cam": snap["cam"], "manual": snap["manual"]})
    return "ok"


class ModelManager:
    """Pegang Analyzer aktif; model bisa diganti / dimatikan / diatur dari web tanpa restart."""

    def __init__(self):
        self.analyzer: Analyzer = None
        self.path = None       # nama file di models/, atau None = off
        self.conf = 0.4
        self.imgsz = 640
        self.device = None

    def load(self, name, conf=None, imgsz=None):
        name = os.path.basename((name or "").strip())
        if not name:
            return "nama model kosong"
        full = os.path.join(MODELS_DIR, name)
        if not os.path.exists(full):
            return f"model {name} tidak ada di folder models/"
        if conf is not None:
            self.conf = float(conf)
        if imgsz is not None:
            self.imgsz = int(imgsz)
        self._swap(Analyzer(video, model_path=full, conf=self.conf, imgsz=self.imgsz, device=self.device,
                            geolocator=geo, tracker=tracker))
        self.path = name
        save_config(model={"path": name, "conf": self.conf, "imgsz": self.imgsz, "enabled": True})
        print(f"[server] model -> {name} conf={self.conf} imgsz={self.imgsz}")
        return "ok"

    def off(self):
        self._swap(Analyzer(video, model_path=None, geolocator=geo, tracker=tracker))
        self.path = None
        save_config(model={"path": load_config().get("model", {}).get("path"),
                           "conf": self.conf, "imgsz": self.imgsz, "enabled": False})
        print("[server] model dimatikan")
        return "ok"

    def set_params(self, conf=None, imgsz=None):
        if conf is not None:
            self.conf = min(max(float(conf), 0.01), 0.99)
        if imgsz is not None:
            self.imgsz = max(32, int(imgsz) // 32 * 32)
        if self.analyzer:
            self.analyzer.set_params(self.conf, self.imgsz)
        save_config(model={"path": self.path or load_config().get("model", {}).get("path"),
                           "conf": self.conf, "imgsz": self.imgsz, "enabled": self.path is not None})
        return "ok"

    def _swap(self, new):
        old = self.analyzer
        self.analyzer = new
        video.analyzer = new
        new.start()
        if old:
            old.stop()

    def snapshot(self):
        a = self.analyzer
        return {
            "loaded": a is not None and a.model is not None,
            "status": a.status if a else "off",
            "path": self.path,
            "device": a.device_name if a else "-",
            "conf": self.conf,
            "imgsz": self.imgsz,
            "fps": round(a.fps, 1) if a else 0,
            "latency_ms": round(a.latency_ms, 1) if a else 0,
            "detections": [_det_json(d) for d in (a.detections if a else [])],
        }


model = ModelManager()


class TelemetryManager:
    """Pegang satu TelemetrySource aktif; bisa diganti kapan saja dari web (adaptif)."""

    def __init__(self):
        self.src: TelemetrySource = None
        self.conn = None
        self.baud = None
        self.prefer_key = load_config().get("hwid")  # device terakhir yang berhasil (VID:PID:SERIAL)
        self._saved_key = self.prefer_key

    def connect(self, conn, baud=57600):
        conn = (conn or "").strip()
        if not conn:
            return "koneksi kosong"
        self.disconnect()
        self.conn, self.baud = conn, int(baud)
        self.src = TelemetrySource(conn, baud=self.baud, prefer_key=self.prefer_key)
        self.src.start()
        save_config(mav=conn, baud=self.baud, hwid=self.prefer_key)
        print(f"[server] telemetry -> {conn} @ {self.baud}")
        return "ok"

    def _remember_device(self):
        """Saat mode auto berhasil connect, ingat hwid device-nya supaya nanti diprioritaskan walau nomor COM berubah."""
        s = self.src
        if s and s.status == "connected" and s.resolved_key and s.resolved_key != self._saved_key:
            self.prefer_key = self._saved_key = s.resolved_key
            save_config(mav=self.conn, baud=self.baud, hwid=s.resolved_key)
            print(f"[server] device telemetry diingat: {s.resolved_key} ({s.resolved})")

    def disconnect(self):
        if self.src is not None:
            self.src.stop()
            self.src = None
        self.conn = None
        print("[server] telemetry diputus")

    def snapshot(self):
        self._remember_device()
        s = self.src.snapshot() if self.src else snapshot_empty()
        s["telemetry"] = {
            "conn": self.conn,
            "baud": self.baud,
            "status": self.src.status if self.src else "disconnected",
            "resolved": self.src.resolved if self.src else None,        # COM aktual (mode auto)
            "resolved_baud": self.src.resolved_baud if self.src else None,
        }
        return s


telemetry = TelemetryManager()


def build_state():
    s = telemetry.snapshot()
    s["video"] = {
        "connected": video.connected,
        "fps": round(video.fps, 1),
        "size": f"{video.width}x{video.height}" if video.width else "-",
        "signal": video.signal_ok,                       # False = blue screen, frame terakhir ditahan
        "lost_s": round(time.time() - video.signal_lost_since, 1) if video.signal_lost_since else 0,
        "blue": round(video.blue, 2),
    }
    s["model"] = model.snapshot()
    s["geo"] = geo.snapshot()
    s["tracks"] = tracker.snapshot()
    s["logs"] = {"telemetry_rows": telem_log.rows if telem_log else 0,
                 "estimate_rows": est_log.rows if est_log else 0}
    s["server_time"] = time.time()
    return s


@asynccontextmanager
async def lifespan(app):
    global video
    video = VideoSource(cam_index=ARGS.cam, width=ARGS.width, height=ARGS.height,
                        jpeg_quality=ARGS.quality, hold_on_loss=not ARGS.no_hold,
                        blue_threshold=ARGS.blue_threshold)
    video.start()

    # model awal: argumen --model/--no-model > config.json (pengaturan terakhir dari web) > models/best.pt jika ada
    cfg = load_config()
    mcfg = cfg.get("model", {})
    model.device = ARGS.device
    model.conf = ARGS.conf if ARGS.conf is not None else mcfg.get("conf", 0.4)
    model.imgsz = ARGS.imgsz if ARGS.imgsz is not None else mcfg.get("imgsz", 640)
    if ARGS.no_model:
        model.off()
    else:
        name = ARGS.model or (mcfg.get("path") if mcfg.get("enabled", True) else None)             or ("best.pt" if os.path.exists(DEFAULT_MODEL) else None)
        if name:
            res = model.load(name)
            if res != "ok":
                print(f"[server] {res} -> jalan tanpa deteksi")
                model.off()
        else:
            print("[server] belum ada model di models/ -> jalan tanpa deteksi (bisa dipilih dari web)")
            model.off()

    # sumber telemetry awal: argumen --mav > config.json (pilihan terakhir dari web) > auto
    mav = ARGS.mav or cfg.get("mav") or "auto"
    baud = ARGS.baud or cfg.get("baud") or 57600
    if mav.lower() != "none":
        telemetry.connect(mav, baud)
    global telem_log, est_log
    telem_log = csvlog.CsvLog(LOGS_DIR, "telemetry", csvlog.TELEMETRY_HEADER)
    est_log = csvlog.CsvLog(LOGS_DIR, "estimates", csvlog.ESTIMATE_HEADER)
    print(f"[server] log CSV -> {telem_log.path} | {est_log.path}")
    sampler = asyncio.create_task(log_sampler())
    print(f"[server] buka http://localhost:{ARGS.port}")
    yield
    sampler.cancel()
    telem_log.flush()
    est_log.flush()
    video.stop()
    if model.analyzer:
        model.analyzer.stop()
    telemetry.disconnect()
    tracker.save()


app = FastAPI(title="Drone Monitor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/state")
def api_state():
    return build_state()


@app.get("/snapshot")
def snapshot():
    jpg = video.snapshot()
    if jpg is None:
        return Response(status_code=503, content="belum ada frame")
    return Response(content=jpg, media_type="image/jpeg",
                    headers={"Content-Disposition": f'inline; filename="vrx_{int(time.time())}.jpg"'})


async def mjpeg_gen():
    seq = 0
    loop = asyncio.get_running_loop()
    while True:
        # tunggu frame baru di thread pool supaya event loop tidak diblok
        seq, jpg = await loop.run_in_executor(None, video.latest, seq, 1.0)
        if jpg is None:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
               + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")


@app.get("/video")
def video_stream():
    return StreamingResponse(mjpeg_gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-cache, no-store",
                                      "Pragma": "no-cache"})


@app.get("/api/ports")
def api_ports():
    return list_ports()


def _csv_response(text, name):
    return Response(content=text.encode("utf-8-sig"), media_type="text/csv",  # BOM: Excel baca UTF-8 dengan benar
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/export/tracks.csv")
def export_tracks():
    snap = tracker.snapshot()
    text = csvlog.rows_to_csv(csvlog.TRACKS_HEADER, csvlog.tracks_rows(snap))
    return _csv_response(text, f"objek_terlacak_{time.strftime('%Y%m%d_%H%M%S')}.csv")


@app.get("/export/telemetry.csv")
def export_telemetry():
    telem_log.flush()
    return FileResponse(telem_log.path, media_type="text/csv",
                        filename=f"telemetry_{telem_log.session}.csv")


@app.get("/export/estimates.csv")
def export_estimates():
    est_log.flush()
    return FileResponse(est_log.path, media_type="text/csv",
                        filename=f"estimasi_lokasi_{est_log.session}.csv")


@app.get("/api/logs")
def api_logs():
    """Info log sesi ini + daftar file log lama di folder logs/."""
    files = sorted(f for f in os.listdir(LOGS_DIR) if f.endswith(".csv")) if os.path.isdir(LOGS_DIR) else []
    return {"session": telem_log.session, "telemetry_rows": telem_log.rows, "estimate_rows": est_log.rows,
            "files": files}


def handle_command(cmd: dict) -> dict:
    """
    Perintah dari browser lewat WebSocket:
      {"cmd":"connect","conn":"auto"}                cari radio/FC otomatis (COM manapun, baud 57600/115200)
      {"cmd":"connect","conn":"COM5","baud":57600}   serial: COM tertentu
      {"cmd":"connect","conn":"udpin:0.0.0.0:14550"} UDP (Mission Planner mirror / SITL)
      {"cmd":"connect","conn":"sim"}                 simulasi
      {"cmd":"disconnect"}
      {"cmd":"ports"}                                daftar COM port
      {"cmd":"models"}                               daftar model di models/
      {"cmd":"model_load","path":"best.pt","conf":0.4,"imgsz":640}
      {"cmd":"model_set","conf":0.6}                 ubah confidence / imgsz live
      {"cmd":"model_off"}
      {"cmd":"geo_set","hfov":110,"tilt":45,"pan":0,"use_attitude":true,"anchor":"bottom"}
      {"cmd":"geo_set","manual":{"lat":-6.2,"lon":106.8,"alt":50,"heading":0}}   pose uji tanpa GPS
      {"cmd":"geo_set","manual":null}                                            kembali ke telemetry
      {"cmd":"track_set","radius_m":3,"lost_s":2,"expire_s":0}                  radius asosiasi objek
      {"cmd":"track_clear"}                                                      hapus semua objek terlacak
      {"cmd":"track_remove","id":3}                                              hapus satu objek
    """
    c = cmd.get("cmd")
    if c == "connect":
        res = telemetry.connect(cmd.get("conn"), cmd.get("baud", 57600))
        return {"type": "ack", "cmd": c, "result": res}
    if c == "disconnect":
        telemetry.disconnect()
        return {"type": "ack", "cmd": c, "result": "ok"}
    if c == "ports":
        return {"type": "ports", "ports": list_ports()}
    if c == "models":
        return {"type": "models", "models": list_models()}
    if c == "model_load":
        return {"type": "ack", "cmd": c, "result": model.load(cmd.get("path"), cmd.get("conf"), cmd.get("imgsz"))}
    if c == "model_off":
        return {"type": "ack", "cmd": c, "result": model.off()}
    if c == "model_set":
        return {"type": "ack", "cmd": c, "result": model.set_params(cmd.get("conf"), cmd.get("imgsz"))}
    if c == "geo_set":
        return {"type": "ack", "cmd": c, "result": geo_set(cmd)}
    if c == "track_set":
        return {"type": "ack", "cmd": c, "result": track_set(cmd)}
    if c == "track_clear":
        tracker.clear()
        return {"type": "ack", "cmd": c, "result": "ok"}
    if c == "track_remove":
        ok = tracker.remove(int(cmd.get("id", 0)))
        return {"type": "ack", "cmd": c, "result": "ok" if ok else "id tidak ada"}
    return {"type": "ack", "cmd": c, "result": "perintah tidak dikenal"}


@app.websocket("/ws")
async def ws_state(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    async def sender():
        while True:
            await ws.send_text(json.dumps(build_state()))
            await asyncio.sleep(1.0 / ARGS.ws_hz)

    async def receiver():
        while True:
            raw = await ws.receive_text()
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # connect/disconnect membuka port & menutup thread -> jalankan di thread pool
            reply = await loop.run_in_executor(None, handle_command, cmd)
            await ws.send_text(json.dumps(reply))

    try:
        await asyncio.gather(sender(), receiver())
    except (WebSocketDisconnect, Exception):
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Drone monitor: VRX video + ArduPilot telemetry")
    p.add_argument("--cam", default=None,
                   help="index kamera VRX (default: auto-cari) | test (pola sintetis) | path video .mp4")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--quality", type=int, default=70, help="kualitas JPEG stream 1-100")
    p.add_argument("--no-hold", action="store_true", help="jangan tahan frame terakhir saat blue screen")
    p.add_argument("--blue-threshold", type=float, default=0.85,
                   help="rasio piksel biru agar frame dianggap blue screen (default 0.85)")
    p.add_argument("--mav", default=None,
                   help="auto | COM5 | udpin:0.0.0.0:14550 | tcp:127.0.0.1:5760 | sim | none "
                        "(default: pilihan terakhir di config.json, kalau belum ada: auto). "
                        "Bisa diganti kapan saja dari web.")
    p.add_argument("--baud", type=int, default=None, help="default 57600 (radio); USB FC biasanya 115200")
    p.add_argument("--model", default=None, help="nama file di models/ (default: pengaturan terakhir / best.pt)")
    p.add_argument("--no-model", action="store_true", help="mulai tanpa analisis (bisa dinyalakan dari web)")
    p.add_argument("--conf", type=float, default=None, help="confidence threshold (default: terakhir / 0.4)")
    p.add_argument("--imgsz", type=int, default=None, help="ukuran input model (default: terakhir / 640)")
    p.add_argument("--device", default=None, help="cpu | 0 (GPU CUDA pertama) | kosong = auto")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 agar bisa dibuka dari HP/laptop lain")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ws-hz", type=float, default=10.0)
    return p.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    if ARGS.cam is not None and ARGS.cam.isdigit():
        ARGS.cam = int(ARGS.cam)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")
