"""
Perencana dropping barang ke objek terlacak yang dipilih.

Barang yang dilepas dari drone yang bergerak tidak jatuh tegak lurus: ia masih membawa kecepatan
horizontal drone, jadi mendarat di DEPAN titik pelepasan. Supaya tepat di objek, barang harus dilepas
sebelum drone berada di atas objek, sejauh "lead":

    t_jatuh = (climb + sqrt(climb^2 + 2 g h)) / g        h = tinggi di atas tanah, climb = + naik
    lead    = v * (delay_s + carry * t_jatuh)            v = kecepatan tanah (vektor, arah lintasan)

  delay_s  jeda mekanisme (servo membuka, barang lepas) - selama itu barang masih ikut drone
  carry    faktor hambatan udara 0..1: 1 = tanpa hambatan (barang padat/berat), < 1 = barang ringan /
           bidang lebar yang cepat melambat. Kalibrasi dengan uji jatuh: carry = jarak_nyata / jarak_hitung.

Titik jatuh prediksi (kalau dilepas SEKARANG) = posisi drone + lead. Barang siap dijatuhkan kalau
titik jatuh prediksi berada dalam radius toleransi dari objek. Titik rilis = objek - lead.
Semua jarak horizontal di bidang datar lokal (NED), cukup akurat untuk jarak < beberapa km.
"""
import math
import threading

G = 9.80665
EARTH_R = 6_371_000.0

DEFAULTS = {"delay_s": 0.3, "carry": 1.0, "tol_m": 3.0}
MIN_SPEED = 0.5   # m/s; di bawah ini drone dianggap hover -> barang jatuh tegak di bawah drone


def ne_offset(lat0, lon0, lat, lon):
    """Offset (utara, timur) meter dari titik 0 ke titik lain (equirectangular)."""
    n = math.radians(lat - lat0) * EARTH_R
    e = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians((lat0 + lat) / 2))
    return n, e


def ne_to_latlon(lat0, lon0, n, e):
    lat = lat0 + math.degrees(n / EARTH_R)
    lon = lon0 + math.degrees(e / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon


def solve(pose, vel, target, delay_s, carry, tol_m):
    """
    pose   : {lat, lon, alt} drone (alt = tinggi di atas tanah, m)
    vel    : (vn, ve, vd) m/s NED, vd + = turun
    target : {lat, lon}
    return : dict hasil (lihat snapshot di bawah); ok False + reason kalau tidak bisa dihitung
    """
    lat, lon, alt = pose.get("lat"), pose.get("lon"), pose.get("alt")
    if lat is None or lon is None:
        return {"ok": False, "reason": "tidak ada posisi GPS drone"}
    if target.get("lat") is None:
        return {"ok": False, "reason": "objek belum punya lokasi"}

    tn, te = ne_offset(lat, lon, target["lat"], target["lon"])
    dist = math.hypot(tn, te)
    out = {"ok": True, "reason": "",
           "dist_m": dist, "bearing": math.degrees(math.atan2(te, tn)) % 360,
           "alt": alt, "slant_m": math.hypot(dist, alt) if alt is not None else None}

    vn, ve, vd = vel
    speed = math.hypot(vn, ve)
    out.update(speed=speed, course=math.degrees(math.atan2(ve, vn)) % 360 if speed >= MIN_SPEED else None)

    if alt is None or alt < 0.5:
        out.update(status="noalt", reason="ketinggian < 0.5 m / tidak ada")
        return out

    climb = -vd
    t_fall = (climb + math.sqrt(max(climb * climb + 2 * G * alt, 0.0))) / G
    t_lead = delay_s + carry * t_fall
    ln, le = vn * t_lead, ve * t_lead                 # titik jatuh prediksi relatif drone
    rn, re = tn - ln, te - le                         # titik rilis relatif drone (drone harus ke sini)
    # jarak titik jatuh prediksi ke objek = jarak drone ke titik rilis (vektor yang sama)
    miss = math.hypot(rn, re)
    out.update(t_fall=t_fall, lead_m=math.hypot(ln, le), miss_m=miss,
               impact=ne_to_latlon(lat, lon, ln, le), release=ne_to_latlon(lat, lon, rn, re))

    if speed >= MIN_SPEED:
        un, ue = vn / speed, ve / speed
        along = rn * un + re * ue                     # + = titik rilis masih di depan
        cross = re * un - rn * ue                     # + = titik rilis di kanan lintasan
        out.update(along_m=along, cross_m=cross, eta_s=along / speed if along > 0 else None)
    else:
        along = cross = None
        out.update(along_m=None, cross_m=None, eta_s=None)

    if miss <= tol_m:
        status = "ready"                              # lepas sekarang -> jatuh dalam toleransi
    elif along is None:
        status = "approach"                           # hover: geser ke atas titik rilis
    elif along < 0:
        status = "passed"                             # titik rilis sudah terlewat
    elif abs(cross) > tol_m:
        status = "offtrack"                           # belum segaris dengan titik rilis
    else:
        status = "approach"
    out["status"] = status
    return out


class DropPlanner:
    """Simpan objek sasaran + parameter drop (bisa diubah dari web), hitung geometri tiap snapshot."""

    def __init__(self, params=None):
        self._lock = threading.Lock()
        self.target_id = None
        self.p = dict(DEFAULTS)
        if params:
            self.set(**{k: params.get(k) for k in DEFAULTS})

    def set(self, delay_s=None, carry=None, tol_m=None):
        with self._lock:
            if delay_s is not None:
                self.p["delay_s"] = min(max(float(delay_s), 0.0), 5.0)
            if carry is not None:
                self.p["carry"] = min(max(float(carry), 0.0), 1.0)
            if tol_m is not None:
                self.p["tol_m"] = min(max(float(tol_m), 0.1), 100.0)

    def select(self, tid):
        with self._lock:
            self.target_id = int(tid) if tid not in (None, "", 0) else None

    def params(self):
        with self._lock:
            return dict(self.p)

    def snapshot(self, pose, vel, tracks):
        """pose: Geolocator.current_pose(); vel: (vn, ve, vd); tracks: list dari ObjectTracker.snapshot()."""
        with self._lock:
            tid, p = self.target_id, dict(self.p)
        out = {"target_id": tid, **p}
        if tid is None:
            return out
        t = next((t for t in tracks if t["id"] == tid), None)
        if t is None:                                  # objek dihapus dari daftar -> batalkan sasaran
            self.select(None)
            out["target_id"] = None
            return out
        out["target"] = {"id": t["id"], "label": t["label"], "lat": t["lat"], "lon": t["lon"], "status": t["status"]}
        r = solve(pose, vel, t, p["delay_s"], p["carry"], p["tol_m"])
        for k, v in r.items():                         # bulatkan supaya JSON ringkas
            if isinstance(v, float):
                r[k] = round(v, 2)
            elif isinstance(v, tuple):
                r[k] = [round(v[0], 7), round(v[1], 7)]
        out["plan"] = r
        return out
