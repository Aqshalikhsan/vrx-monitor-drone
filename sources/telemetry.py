"""
Sumber telemetry MAVLink (ArduPilot) - jalan di thread sendiri.

Koneksi bisa:
    COM5                  serial telemetry radio / USB FC (baud default 57600)
    udpin:0.0.0.0:14550   terima forward dari Mission Planner (MAVLink Mirror) / SITL
    tcp:127.0.0.1:5760    SITL
    sim                   simulasi terbang melingkar, untuk tes tanpa hardware

State terbaru disimpan sebagai dict; ambil lewat snapshot() (thread-safe).
"""
import math
import threading
import time


def _empty_state():
    return {
        "sysid": None,
        "mode": "-",
        "armed": False,
        "lat": None, "lon": None,           # derajat
        "alt_msl": None, "alt_rel": None,   # meter
        "heading": None,                    # derajat
        "groundspeed": None, "airspeed": None, "climb": None,  # m/s
        "throttle": None,                   # %
        "roll": None, "pitch": None, "yaw": None,  # derajat
        "fix_type": 0, "sats": 0, "hdop": None,
        "volt": None, "current": None, "batt_pct": None,
        "msg_rate": 0.0,
        "last_hb": 0.0,
    }


FIX_NAMES = {0: "No GPS", 1: "No Fix", 2: "2D", 3: "3D", 4: "DGPS", 5: "RTK Float", 6: "RTK Fixed"}


# USB vendor ID yang lazim untuk radio telemetry / flight controller
KNOWN_VIDS = {
    0x10C4: "Silicon Labs CP210x (radio SiK/Holybro/3DR)",
    0x0403: "FTDI",
    0x1A86: "CH340",
    0x1209: "ArduPilot / Pixhawk",
    0x2DAE: "Hex Cube",
    0x26AC: "3DR Pixhawk",
    0x3162: "Holybro",
    0x0483: "STM32 VCP",
}


def list_ports():
    """Daftar COM port yang ada sekarang: [{"device": "COM5", "desc": "...", "hwid", "vid", "pid", "known"}]"""
    from serial.tools import list_ports as lp
    out = []
    for p in sorted(lp.comports(), key=lambda p: p.device):
        out.append({
            "device": p.device,
            "desc": p.description or "",
            "hwid": p.hwid or "",
            "vid": p.vid, "pid": p.pid,
            "serial": p.serial_number or "",
            "known": KNOWN_VIDS.get(p.vid) if p.vid else None,
        })
    return out


def port_key(p):
    """Identitas hardware yang tidak berubah walau nomor COM berubah: 'VID:PID:SERIAL'."""
    if not p.get("vid"):
        return None
    return f"{p['vid']:04X}:{p['pid']:04X}:{p.get('serial') or ''}"


def find_telemetry_port(prefer_key=None):
    """
    Pilih COM port telemetry secara otomatis:
      1. device yang sama dengan terakhir kali dipakai (prefer_key), berapapun nomor COM-nya
      2. port dengan vendor ID yang dikenal sebagai radio/FC
      3. kalau hanya ada satu port, pakai itu
    Return dict port atau None.
    """
    ports = list_ports()
    if prefer_key:
        for p in ports:
            if port_key(p) == prefer_key:
                return p
    for p in ports:
        if p["known"]:
            return p
    if len(ports) == 1:
        return ports[0]
    return None


def snapshot_empty():
    s = _empty_state()
    s["link"] = False
    s["fix_name"] = FIX_NAMES[0]
    return s


class TelemetrySource(threading.Thread):
    def __init__(self, conn, baud=57600, prefer_key=None):
        super().__init__(daemon=True, name="telemetry")
        self.conn = conn              # "auto" | "COM5" | "udpin:..." | "tcp:..." | "sim"
        self.baud = baud
        self.prefer_key = prefer_key  # hwid device terakhir (untuk mode auto)
        self._lock = threading.Lock()
        self.state = _empty_state()
        self._stop = threading.Event()
        self.status = "connecting"    # connecting | waiting | connected | error: ... | stopped
        self.resolved = None          # COM port yang benar-benar dipakai (mode auto)
        self.resolved_key = None      # hwid-nya, disimpan manager ke config saat connected
        self.resolved_baud = baud
        self._m = None

    def snapshot(self):
        with self._lock:
            s = dict(self.state)
        s["link"] = (time.time() - s["last_hb"]) < 3.0
        s["fix_name"] = FIX_NAMES.get(s["fix_type"], str(s["fix_type"]))
        return s

    def stop(self):
        """Hentikan thread dan lepaskan port SEKARANG (supaya port bisa dipakai koneksi baru)."""
        self._stop.set()
        m = self._m
        if m is not None:
            try:
                m.close()
            except Exception:
                pass
        self.status = "stopped"

    def _update(self, **kw):
        with self._lock:
            self.state.update(kw)

    # ------------------------------------------------------------------ run
    def run(self):
        if self.conn == "sim":
            self._run_sim()
        else:
            self._run_mavlink()

    def _resolve(self):
        """Tentukan target koneksi untuk percobaan ini. Return (conn, baud) atau None kalau belum ada device."""
        if self.conn != "auto":
            return self.conn, self.baud
        p = find_telemetry_port(self.prefer_key)
        if p is None:
            return None
        self.resolved, self.resolved_key = p["device"], port_key(p)
        return p["device"], self.resolved_baud

    def _run_mavlink(self):
        from pymavlink import mavutil

        is_serial = self.conn == "auto" or not (":" in self.conn)
        while not self._stop.is_set():
            m = None
            try:
                target = self._resolve()
                if target is None:
                    if self.status != "waiting":
                        print("[telemetry] auto: belum ada COM port radio/FC, menunggu dicolok ...")
                    self.status = "waiting"
                    self._stop.wait(2.0)
                    continue
                conn, baud = target
                self.status = "connecting"
                print(f"[telemetry] membuka {conn} (baud {baud}) ...")
                m = mavutil.mavlink_connection(conn, baud=baud, autoreconnect=False)
                self._m = m
                print("[telemetry] menunggu heartbeat ...")
                hb = self._wait_heartbeat(m, timeout=10)
                if self._stop.is_set():
                    break
                if hb is None:
                    m.close()
                    if is_serial and self.conn == "auto":
                        # radio biasanya 57600, USB langsung ke FC biasanya 115200 -> coba bergantian
                        self.resolved_baud = 115200 if baud == 57600 else 57600
                        print(f"[telemetry] tidak ada heartbeat di {baud}, coba baud {self.resolved_baud}")
                    else:
                        print("[telemetry] tidak ada heartbeat, coba lagi")
                    self.status = f"error: tidak ada heartbeat ({conn} @ {baud})"
                    continue
                print(f"[telemetry] heartbeat dari sysid={m.target_system} compid={m.target_component} "
                      f"via {conn} @ {baud}")
                self.resolved, self.resolved_baud = conn, baud
                self.status = "connected"
                self._update(sysid=m.target_system, last_hb=time.time())
                self._request_streams(m)
                self._read_loop(m)
                m.close()
            except Exception as e:  # port dicabut / tidak ada / dipakai aplikasi lain
                if self._stop.is_set():
                    break
                self.status = f"error: {e}"
                print(f"[telemetry] error: {e}")
                import traceback
                traceback.print_exc()
                if m is not None:
                    try:
                        m.close()
                    except Exception:
                        pass
                self._stop.wait(2.0)
        self._m = None
        self.status = "stopped"

    def _wait_heartbeat(self, m, timeout):
        """Seperti m.wait_heartbeat() tapi bisa dibatalkan lewat stop() tiap 1 detik."""
        deadline = time.time() + timeout
        while not self._stop.is_set() and time.time() < deadline:
            hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if hb is not None and hb.get_srcSystem() != 255:  # 255 = GCS lain
                m.target_system = hb.get_srcSystem()
                m.target_component = hb.get_srcComponent()
                return hb
        return None

    def _request_streams(self, m):
        """Minta FC mengirim pesan yang kita butuhkan pada rate tertentu."""
        from pymavlink import mavutil
        wanted = [  # (msg_id, interval_us)
            (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 200_000),  # 5 Hz
            (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 500_000),          # 2 Hz
            (mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, 200_000),
            (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 100_000),             # 10 Hz
            (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 1_000_000),         # 1 Hz
        ]
        for msg_id, us in wanted:
            m.mav.command_long_send(
                m.target_system, m.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, us, 0, 0, 0, 0, 0)
        # fallback untuk firmware lama yang belum mendukung SET_MESSAGE_INTERVAL
        m.mav.request_data_stream_send(
            m.target_system, m.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)

    def _read_loop(self, m):
        from pymavlink import mavutil
        n, t0 = 0, time.time()
        while not self._stop.is_set():
            msg = m.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                if time.time() - self.state["last_hb"] > 10:
                    print("[telemetry] link hilang > 10 s, reconnect")
                    return
                continue
            n += 1
            if time.time() - t0 >= 1.0:
                self._update(msg_rate=n / (time.time() - t0))
                n, t0 = 0, time.time()

            t = msg.get_type()
            if t == "HEARTBEAT":
                if msg.type == mavutil.mavlink.MAV_TYPE_GCS:
                    continue  # heartbeat dari GCS lain, bukan dari drone
                armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self._update(last_hb=time.time(), armed=armed,
                             mode=mavutil.mode_string_v10(msg))
            elif t == "GLOBAL_POSITION_INT":
                self._update(lat=msg.lat / 1e7, lon=msg.lon / 1e7,
                             alt_msl=msg.alt / 1000.0, alt_rel=msg.relative_alt / 1000.0,
                             heading=msg.hdg / 100.0 if msg.hdg != 65535 else None)
            elif t == "GPS_RAW_INT":
                self._update(fix_type=msg.fix_type, sats=msg.satellites_visible,
                             hdop=msg.eph / 100.0 if msg.eph != 65535 else None)
                # kalau GLOBAL_POSITION_INT belum keluar (EKF belum siap), pakai posisi GPS mentah
                if self.state["lat"] is None and msg.fix_type >= 2:
                    self._update(lat=msg.lat / 1e7, lon=msg.lon / 1e7)
            elif t == "VFR_HUD":
                self._update(groundspeed=msg.groundspeed, airspeed=msg.airspeed,
                             climb=msg.climb, throttle=msg.throttle)
            elif t == "ATTITUDE":
                self._update(roll=math.degrees(msg.roll), pitch=math.degrees(msg.pitch),
                             yaw=math.degrees(msg.yaw) % 360)
            elif t == "SYS_STATUS":
                self._update(volt=msg.voltage_battery / 1000.0,
                             current=msg.current_battery / 100.0 if msg.current_battery != -1 else None,
                             batt_pct=msg.battery_remaining if msg.battery_remaining != -1 else None)

    # ------------------------------------------------------------------ sim
    def _run_sim(self):
        """Terbang melingkar radius ~150 m di sekitar Monas, Jakarta."""
        print("[telemetry] MODE SIMULASI")
        self.status = "connected"
        lat0, lon0 = -6.1754, 106.8272
        t0 = time.time()
        self._update(sysid=1, mode="LOITER", armed=True, fix_type=3, sats=14, hdop=0.8)
        while not self._stop.is_set():
            t = time.time() - t0
            a = t * 0.15
            r = 150 / 111_320.0
            self._update(
                last_hb=time.time(),
                lat=lat0 + r * math.sin(a),
                lon=lon0 + r * math.cos(a) / math.cos(math.radians(lat0)),
                alt_rel=50 + 5 * math.sin(t / 3), alt_msl=58 + 5 * math.sin(t / 3),
                heading=(math.degrees(a) + 90) % 360, yaw=(math.degrees(a) + 90) % 360,
                groundspeed=22.5, airspeed=23.1, climb=1.6 * math.cos(t / 3), throttle=55,
                roll=25 + 3 * math.sin(t), pitch=2 * math.sin(t / 2),
                volt=15.8 - t * 0.002, current=12.3, batt_pct=max(0, int(95 - t / 10)),
                msg_rate=25.0,
            )
            self._stop.wait(0.1)
        self.status = "stopped"
