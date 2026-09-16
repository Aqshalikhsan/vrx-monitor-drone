"""
Pelacakan objek dengan ID terkunci (lock) dan penyimpanan permanen.

Aturan:
  - Begitu sebuah objek terlihat, langsung dicatat sebagai objek baru (#1, #2, ...), walau
    lat/lon belum tersedia (lokasi diisi saat pertama kali ada estimasi GPS).
  - Lokasi objek = lat/lon saat PERTAMA kali ada estimasi (anchor). Itulah yang dilaporkan.
  - Deteksi berikutnya dianggap objek yang sama (ID tetap) kalau salah satu terpenuhi:
      a. kotak deteksi tumpang tindih dengan kotak terakhir objek yang baru saja terlihat
         (IoU >= iou_min, terakhir terlihat <= iou_window_s) -> kontinuitas visual
      b. lat/lon deteksi berada dalam radius R (default 3 m) dari anchor objek
    Objek boleh bergerak ke kanan/kiri selama masih di dalam radius.
  - Pencocokan satu-ke-satu (greedy, yang paling cocok dulu).
  - Objek TIDAK PERNAH dihapus otomatis (kecuali expire_s > 0 diset). Daftar disimpan ke file
    JSON dan dimuat kembali saat server start.

Status: "seen" (terlihat di frame terakhir) atau "locked" (tersimpan, terakhir terlihat X s lalu).
"""
import json
import math
import os
import threading
import time

EARTH_R = 6_371_000.0


def dist_m(lat1, lon1, lat2, lon2):
    """Jarak datar (m) untuk jarak pendek (equirectangular)."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    return EARTH_R * math.hypot(dlat, dlon)


def overlap(a, b):
    """
    Skor tumpang tindih dua kotak: maks(IoU, irisan / kotak terkecil).
    Irisan relatif kotak terkecil menangkap kasus kotak terpotong di tepi frame vs kotak penuh.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    iou_v = inter / (area_a + area_b - inter)
    cont = inter / min(area_a, area_b)
    return max(iou_v, 0.6 * cont)   # cont 0.5 -> skor 0.3 (= iou_min)


class Track:
    FIELDS = ("id", "label", "lat", "lon", "first_seen", "last_seen", "last_lat", "last_lon",
              "hits", "max_conf", "moved_m", "max_moved_m", "last_box", "located_at")

    def __init__(self, tid, label, conf, now, box, geo=None):
        self.id = tid                       # None = kandidat, belum dikonfirmasi
        self.label = label
        self.lat = self.lon = None          # anchor: lokasi pertama kali ada estimasi
        self.located_at = None
        self.first_seen = self.last_seen = now
        self.last_lat = self.last_lon = None
        self.hits = 1
        self.max_conf = conf
        self.moved_m = 0.0                  # jarak posisi terakhir dari anchor
        self.max_moved_m = 0.0
        self.last_box = list(box)
        self.seen_now = True
        if geo:
            self.set_anchor(geo, now)

    def set_anchor(self, geo, now):
        self.lat, self.lon = geo["lat"], geo["lon"]
        self.last_lat, self.last_lon = geo["lat"], geo["lon"]
        self.located_at = now

    def hit(self, d, now, geo_dist=None):
        self.seen_now = True
        self.last_seen = now
        self.hits += 1
        self.max_conf = max(self.max_conf, d["conf"])
        self.last_box = list(d["box"])
        g = d.get("geo")
        if g and g.get("ok"):
            if self.lat is None:
                self.set_anchor(g, now)
            else:
                self.last_lat, self.last_lon = g["lat"], g["lon"]
                dm = geo_dist if geo_dist is not None else dist_m(g["lat"], g["lon"], self.lat, self.lon)
                self.moved_m = dm
                self.max_moved_m = max(self.max_moved_m, dm)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.FIELDS}

    @classmethod
    def from_dict(cls, d):
        t = cls.__new__(cls)
        for k in cls.FIELDS:
            setattr(t, k, d.get(k))
        t.seen_now = False
        if t.last_box is None:
            t.last_box = [0, 0, 0, 0]
        return t


class ObjectTracker:
    def __init__(self, radius_m=3.0, lost_s=2.0, expire_s=0, max_tracks=1000,
                 iou_min=0.3, iou_window_s=1.5, iou_window_nogeo_s=10.0, store_path=None,
                 min_hits=3, pending_ttl_s=1.0):
        self.radius_m = float(radius_m)
        self.lost_s = float(lost_s)
        self.expire_s = float(expire_s)      # 0 = tidak pernah dihapus otomatis
        self.max_tracks = max_tracks
        self.iou_min = iou_min
        self.iou_window_s = iou_window_s
        self.iou_window_nogeo_s = iou_window_nogeo_s  # tanpa lokasi, kotak jadi satu-satunya identitas -> jendela lebih lama
        self.match_label = True
        self.min_hits = min_hits            # terlihat >= N kali sebelum dicatat sebagai objek
        self.pending_ttl_s = pending_ttl_s  # kandidat dibuang kalau tidak terlihat lagi selama ini
        self.store_path = store_path
        self._pending = []                  # kandidat (Track dengan id None)
        self._tracks = {}
        self._next_id = 1
        self._lock = threading.Lock()
        self._dirty = False
        self._last_save = 0.0
        if store_path:
            self._load()

    # ------------------------------------------------------------------ penyimpanan
    def _load(self):
        try:
            with open(self.store_path, encoding="utf-8") as f:
                data = json.load(f)
            for d in data.get("tracks", []):
                t = Track.from_dict(d)
                self._tracks[t.id] = t
            self._next_id = max([data.get("next_id", 1)] + [t.id + 1 for t in self._tracks.values()])
            if self._tracks:
                print(f"[tracker] {len(self._tracks)} objek dimuat dari {os.path.basename(self.store_path)}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[tracker] gagal memuat {self.store_path}: {e}")

    def _save_locked(self, force=False):
        """Simpan ke disk kalau ada perubahan; dibatasi maks 1x per 2 detik kecuali force."""
        if not self.store_path or not self._dirty:
            return
        now = time.time()
        if not force and now - self._last_save < 2.0:
            return
        try:
            tmp = self.store_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"next_id": self._next_id, "saved_at": now,
                           "tracks": [t.to_dict() for t in self._tracks.values()]}, f, indent=1)
            os.replace(tmp, self.store_path)
            self._dirty = False
            self._last_save = now
        except Exception as e:
            print(f"[tracker] gagal menyimpan: {e}")

    def save(self):
        with self._lock:
            self._save_locked(force=True)

    # ------------------------------------------------------------------ pengaturan
    def set(self, radius_m=None, lost_s=None, expire_s=None, min_hits=None):
        with self._lock:
            if radius_m is not None:
                self.radius_m = min(max(float(radius_m), 0.1), 1000.0)
            if lost_s is not None:
                self.lost_s = max(float(lost_s), 0.1)
            if expire_s is not None:
                self.expire_s = max(float(expire_s), 0.0)
            if min_hits is not None:
                self.min_hits = max(1, int(min_hits))

    def clear(self):
        with self._lock:
            self._tracks.clear()
            self._pending.clear()
            self._next_id = 1
            self._dirty = True
            self._save_locked(force=True)

    def remove(self, tid):
        with self._lock:
            if tid in self._tracks:
                del self._tracks[tid]
                self._dirty = True
                self._save_locked(force=True)
                return True
            return False

    # ------------------------------------------------------------------ inti
    def update(self, dets, now=None):
        """
        dets: list deteksi {'label','conf','box', 'geo'?}. Menambahkan 'track_id' ke tiap deteksi.
        """
        now = now or time.time()
        with self._lock:
            # semua kandidat asosiasi: objek terkonfirmasi + kandidat (index list = kunci sementara)
            pool = list(self._tracks.values()) + self._pending
            for t in pool:
                t.seen_now = False
            used_det, used_trk = set(), set()

            # (a) kontinuitas visual: kotak tumpang tindih dengan kotak terakhir objek yang baru terlihat
            pairs = []
            for i, d in enumerate(dets):
                d_geo = bool(d.get("geo", {}).get("ok"))
                for k, t in enumerate(pool):
                    if self.match_label and t.label != d["label"]:
                        continue
                    window = self.iou_window_s if (d_geo and t.lat is not None) else self.iou_window_nogeo_s
                    if t.id is None:
                        window = self.pending_ttl_s
                    if now - (t.last_seen or 0) > window:
                        continue
                    v = overlap(d["box"], t.last_box)
                    if v >= self.iou_min:
                        pairs.append((-v, i, k))
            pairs.sort()
            for _, i, k in pairs:
                if i in used_det or k in used_trk:
                    continue
                used_det.add(i)
                used_trk.add(k)
                pool[k].hit(dets[i], now)
                dets[i]["track_id"] = pool[k].id

            # (b) lokasi: dalam radius dari anchor
            pairs = []
            for i, d in enumerate(dets):
                if i in used_det:
                    continue
                g = d.get("geo")
                if not (g and g.get("ok")):
                    continue
                for k, t in enumerate(pool):
                    if k in used_trk or t.lat is None:
                        continue
                    if self.match_label and t.label != d["label"]:
                        continue
                    dm = dist_m(g["lat"], g["lon"], t.lat, t.lon)
                    if dm <= self.radius_m:
                        pairs.append((dm, i, k))
            pairs.sort()
            for dm, i, k in pairs:
                if i in used_det or k in used_trk:
                    continue
                used_det.add(i)
                used_trk.add(k)
                pool[k].hit(dets[i], now, geo_dist=dm)
                dets[i]["track_id"] = pool[k].id

            # (c) sisanya -> kandidat baru (belum dapat ID sampai terlihat min_hits kali)
            for i, d in enumerate(dets):
                if i in used_det:
                    continue
                g = d.get("geo")
                self._pending.append(Track(None, d["label"], d["conf"], now, d["box"],
                                           geo=g if (g and g.get("ok")) else None))
                d["track_id"] = None

            # konfirmasi kandidat yang cukup sering terlihat; buang yang basi
            keep = []
            for t in self._pending:
                if t.hits >= self.min_hits and len(self._tracks) < self.max_tracks:
                    t.id = self._next_id
                    self._next_id += 1
                    self._tracks[t.id] = t
                    self._dirty = True
                    for d in dets:                      # deteksi frame ini yang tadi cocok ke kandidat ini
                        if d.get("track_id") is None and t.seen_now and list(d["box"]) == t.last_box:
                            d["track_id"] = t.id
                elif now - t.last_seen <= self.pending_ttl_s:
                    keep.append(t)
            self._pending = keep

            if any(t.seen_now for t in self._tracks.values()):
                self._dirty = True
            if self.expire_s > 0:
                for tid in [tid for tid, t in self._tracks.items() if now - t.last_seen > self.expire_s]:
                    del self._tracks[tid]
                    self._dirty = True
            self._save_locked()

    # ------------------------------------------------------------------ keluaran
    def snapshot(self, now=None):
        now = now or time.time()
        with self._lock:
            items = []
            for t in sorted(self._tracks.values(), key=lambda t: t.id):
                age = now - (t.last_seen or 0)
                items.append({
                    "id": t.id, "label": t.label,
                    "lat": round(t.lat, 7) if t.lat is not None else None,
                    "lon": round(t.lon, 7) if t.lon is not None else None,
                    "last_lat": round(t.last_lat, 7) if t.last_lat is not None else None,
                    "last_lon": round(t.last_lon, 7) if t.last_lon is not None else None,
                    "status": "seen" if age < self.lost_s else "locked",
                    "since_s": round(age, 1),
                    "first_seen": t.first_seen, "last_seen": t.last_seen, "located_at": t.located_at,
                    "hits": t.hits, "max_conf": round(t.max_conf, 2),
                    "moved_m": round(t.moved_m, 1), "max_moved_m": round(t.max_moved_m, 1),
                })
            return {"radius_m": self.radius_m, "lost_s": self.lost_s, "expire_s": self.expire_s,
                    "min_hits": self.min_hits, "pending": len(self._pending),
                    "count": len(items), "active": sum(1 for i in items if i["status"] == "seen"),
                    "located": sum(1 for i in items if i["lat"] is not None),
                    "tracks": items}
