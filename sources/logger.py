"""
Log CSV per sesi (satu file per kali server jalan) untuk telemetry dan estimasi lokasi objek.

  logs/telemetry_YYYYmmdd_HHMMSS.csv
  logs/estimates_YYYYmmdd_HHMMSS.csv

Baris ditulis ke buffer lalu di-flush ke disk tiap ~1 detik supaya murah; file bisa
di-download kapan saja lewat endpoint server.
"""
import csv
import io
import os
import threading
import time
from datetime import datetime


def iso(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class CsvLog:
    def __init__(self, directory, prefix, header):
        os.makedirs(directory, exist_ok=True)
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"{prefix}_{self.session}.csv")
        self.header = header
        self.rows = 0
        self._buf = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    def add(self, row):
        with self._lock:
            self._buf.append(row)
            self.rows += 1
            if time.time() - self._last_flush >= 1.0:
                self._flush_locked()

    def flush(self):
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if not self._buf:
            return
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(self._buf)
        self._buf.clear()
        self._last_flush = time.time()


def rows_to_csv(header, rows):
    """Buat isi CSV (str) dari header + list baris."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(header)
    w.writerows(rows)
    return out.getvalue()


def _f(v, nd=6):
    return "" if v is None else (round(v, nd) if isinstance(v, float) else v)


TELEMETRY_HEADER = ["time", "unix", "link", "mode", "armed", "lat", "lon", "alt_rel_m", "alt_msl_m",
                    "heading_deg", "groundspeed_ms", "airspeed_ms", "climb_ms", "throttle_pct",
                    "roll_deg", "pitch_deg", "yaw_deg", "fix_type", "fix_name", "sats", "hdop",
                    "volt", "current_a", "batt_pct", "msg_rate", "video_signal", "model_fps",
                    "detections", "tracks_total", "tracks_seen"]


def telemetry_row(s, now):
    v, m, t = s.get("video", {}), s.get("model", {}), s.get("tracks", {})
    return [iso(now), round(now, 3), int(bool(s.get("link"))), s.get("mode"), int(bool(s.get("armed"))),
            _f(s.get("lat"), 7), _f(s.get("lon"), 7), _f(s.get("alt_rel"), 2), _f(s.get("alt_msl"), 2),
            _f(s.get("heading"), 1), _f(s.get("groundspeed"), 2), _f(s.get("airspeed"), 2),
            _f(s.get("climb"), 2), _f(s.get("throttle")), _f(s.get("roll"), 1), _f(s.get("pitch"), 1),
            _f(s.get("yaw"), 1), s.get("fix_type"), s.get("fix_name"), s.get("sats"), _f(s.get("hdop"), 2),
            _f(s.get("volt"), 2), _f(s.get("current"), 2), _f(s.get("batt_pct")), _f(s.get("msg_rate"), 1),
            int(bool(v.get("signal", True))), m.get("fps"), len(m.get("detections", [])),
            t.get("count"), t.get("active")]


ESTIMATE_HEADER = ["time", "unix", "track_id", "label", "conf", "x1", "y1", "x2", "y2",
                   "geo_ok", "reason", "obj_lat", "obj_lon", "dist_m", "bearing_deg", "err_m", "depression_deg",
                   "pose_source", "drone_lat", "drone_lon", "drone_alt_m", "drone_heading_deg",
                   "drone_roll_deg", "drone_pitch_deg", "hfov", "tilt", "pan"]


def estimate_rows(dets, geo_state, now):
    p, cam = geo_state.get("pose", {}), geo_state.get("cam", {})
    rows = []
    for d in dets:
        g = d.get("geo") or {}
        x1, y1, x2, y2 = d.get("box", [None] * 4)
        rows.append([iso(now), round(now, 3), d.get("track_id"), d.get("label"), d.get("conf"),
                     x1, y1, x2, y2, int(bool(g.get("ok"))), g.get("reason", ""),
                     _f(g.get("lat"), 7), _f(g.get("lon"), 7), _f(g.get("dist"), 1), _f(g.get("bearing"), 1),
                     _f(g.get("err_m"), 1), _f(g.get("depression"), 1),
                     p.get("source"), _f(p.get("lat"), 7), _f(p.get("lon"), 7), _f(p.get("alt"), 2),
                     _f(p.get("heading"), 1), _f(p.get("roll"), 1), _f(p.get("pitch"), 1),
                     cam.get("hfov"), cam.get("tilt"), cam.get("pan")])
    return rows


TRACKS_HEADER = ["id", "label", "status", "lat", "lon", "last_lat", "last_lon", "hits", "max_conf",
                 "moved_m", "max_moved_m", "first_seen", "last_seen", "located_at"]


def tracks_rows(snapshot):
    rows = []
    for t in snapshot.get("tracks", []):
        rows.append([t["id"], t["label"], "terlihat" if t["status"] == "seen" else "terkunci",
                     _f(t["lat"], 7), _f(t["lon"], 7), _f(t["last_lat"], 7), _f(t["last_lon"], 7),
                     t["hits"], t["max_conf"], t["moved_m"], t["max_moved_m"],
                     iso(t["first_seen"]) if t["first_seen"] else "",
                     iso(t["last_seen"]) if t["last_seen"] else "",
                     iso(t["located_at"]) if t.get("located_at") else ""])
    return rows
