"""
Estimasi lokasi objek (lat/lon) dari posisi piksel di frame + pose drone + sudut kamera.

Prinsip: sinar dari kamera lewat piksel objek diproyeksikan ke bidang tanah datar
(tanah dianggap setinggi titik home, alt_rel = tinggi di atas tanah).

Frame koordinat:
  kamera : x kanan, y bawah, z depan (pinhole)
  body   : FRD - x depan, y kanan, z bawah (konvensi ArduPilot)
  dunia  : NED - x utara, y timur, z bawah

Pose dari MAVLink: lat, lon (deg), alt (m di atas tanah), heading (deg, searah jarum jam dari utara),
roll (deg, + = sayap kanan turun), pitch (deg, + = hidung naik).
"""
import math
import threading

import numpy as np

EARTH_R = 6_371_000.0


# ----------------------------------------------------------------------------- rotasi
def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ----------------------------------------------------------------------------- inti
def pixel_to_ground(u, v, img_w, img_h, pose, cam):
    """
    u, v     : piksel objek (0,0 = kiri atas)
    pose     : dict lat, lon, alt, heading, roll, pitch
    cam      : dict hfov (deg), vfov (deg|None -> dari aspek), tilt (deg, 0 = datar, 90 = tegak ke bawah),
               pan (deg, + = kamera menoleh ke kanan), use_attitude (bool)
    return dict: ok, lat, lon, dist, bearing, depression, err_m, reason
    """
    alt = pose.get("alt")
    if pose.get("lat") is None or pose.get("lon") is None:
        return {"ok": False, "reason": "tidak ada posisi GPS"}
    if alt is None or alt < 0.5:
        return {"ok": False, "reason": "ketinggian < 0.5 m"}

    hfov = math.radians(cam["hfov"])
    fx = (img_w / 2) / math.tan(hfov / 2)
    if cam.get("vfov"):
        fy = (img_h / 2) / math.tan(math.radians(cam["vfov"]) / 2)
    else:
        fy = fx  # piksel persegi -> vfov mengikuti aspek
    cx, cy = img_w / 2, img_h / 2

    # sinar di frame kamera
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    # kamera -> body (tanpa tilt): z_cam -> x_body, x_cam -> y_body, y_cam -> z_body
    d_b0 = np.array([d_cam[2], d_cam[0], d_cam[1]])
    # pasang kamera: tilt ke bawah (rotasi negatif di sumbu y karena Ry(+) = hidung naik), lalu pan
    d_body = _rz(math.radians(cam.get("pan", 0.0))) @ _ry(-math.radians(cam.get("tilt", 0.0))) @ d_b0
    # body -> NED dengan attitude drone
    roll = math.radians(pose.get("roll") or 0.0) if cam.get("use_attitude", True) else 0.0
    pitch = math.radians(pose.get("pitch") or 0.0) if cam.get("use_attitude", True) else 0.0
    yaw = math.radians(pose.get("heading") or 0.0)
    d_ned = _rz(yaw) @ _ry(pitch) @ _rx(roll) @ d_body
    d_ned = d_ned / np.linalg.norm(d_ned)

    if d_ned[2] <= 1e-6:
        return {"ok": False, "reason": "sinar di atas horizon (tilt kamera terlalu kecil)"}

    t = alt / d_ned[2]
    n, e = t * d_ned[0], t * d_ned[1]
    dist = math.hypot(n, e)
    bearing = math.degrees(math.atan2(e, n)) % 360
    depression = math.degrees(math.asin(d_ned[2]))
    # sensitivitas: error sudut ~1 deg (attitude + pemasangan) -> error jarak di tanah
    err_m = alt * math.radians(1.0) / max(math.sin(math.radians(depression)) ** 2, 1e-6)

    lat0 = math.radians(pose["lat"])
    lat = pose["lat"] + math.degrees(n / EARTH_R)
    lon = pose["lon"] + math.degrees(e / (EARTH_R * math.cos(lat0)))
    return {"ok": True, "lat": lat, "lon": lon, "dist": dist, "bearing": bearing,
            "depression": depression, "err_m": err_m, "reason": ""}


# ----------------------------------------------------------------------------- pengelola
DEFAULT_CAM = {
    "hfov": 110.0,        # kamera FPV lensa 2.1 mm ~ 110-120 deg horizontal
    "vfov": None,         # None = dari aspek frame
    "tilt": 45.0,         # 0 = datar ke depan, 90 = tegak ke bawah
    "pan": 0.0,
    "use_attitude": True, # kompensasi roll/pitch drone (matikan kalau kamera di gimbal terstabilisasi)
    "anchor": "bottom",   # titik objek: bottom (kaki, tengah bawah kotak) | center
}


class Geolocator:
    """Simpan parameter kamera (bisa diubah dari web) dan beri lat/lon pada tiap deteksi."""

    def __init__(self, pose_fn, cam=None, manual=None):
        self.pose_fn = pose_fn        # callable -> snapshot telemetry
        self.cam = dict(DEFAULT_CAM)
        if cam:
            self.cam.update({k: v for k, v in cam.items() if k in DEFAULT_CAM})
        self.manual = manual          # dict lat, lon, alt, heading untuk uji tanpa GPS; None = pakai telemetry
        self.last_pose = {}
        self._lock = threading.Lock()

    def set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                if k == "manual":
                    self.manual = dict(v) if v else None
                elif k in DEFAULT_CAM and v is not None:
                    if k == "anchor":
                        self.cam[k] = "center" if v == "center" else "bottom"
                    elif k == "use_attitude":
                        self.cam[k] = bool(v)
                    elif k == "vfov":
                        self.cam[k] = float(v) if v else None
                    else:
                        self.cam[k] = float(v)
            self.cam["hfov"] = min(max(self.cam["hfov"], 10.0), 175.0)
            self.cam["tilt"] = min(max(self.cam["tilt"], -90.0), 90.0)

    def current_pose(self):
        if self.manual:
            p = dict(self.manual)
            p.setdefault("roll", 0.0)
            p.setdefault("pitch", 0.0)
            p["source"] = "manual"
            return p
        s = self.pose_fn() or {}
        heading = s.get("heading")
        if heading is None:
            heading = s.get("yaw")
        return {"lat": s.get("lat") if s.get("fix_type", 0) >= 2 else None,
                "lon": s.get("lon") if s.get("fix_type", 0) >= 2 else None,
                "alt": s.get("alt_rel"), "heading": heading,
                "roll": s.get("roll"), "pitch": s.get("pitch"), "source": "telemetry"}

    def annotate(self, dets, img_w, img_h):
        """Tambahkan key 'geo' ke tiap deteksi (in-place). Pose dibaca sekali per frame."""
        with self._lock:
            cam = dict(self.cam)
        pose = self.current_pose()
        self.last_pose = pose
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            u = (x1 + x2) / 2
            v = y2 if cam["anchor"] == "bottom" else (y1 + y2) / 2
            d["geo"] = pixel_to_ground(u, v, img_w, img_h, pose, cam)
        return dets

    def snapshot(self):
        with self._lock:
            cam = dict(self.cam)
        return {"cam": cam, "manual": self.manual, "pose": self.current_pose()}
