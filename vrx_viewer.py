"""
VRX viewer - menampilkan video dari receiver 5.8GHz (USB UVC "USB2.0 PC CAMERA").

Pakai:
    py vrx_viewer.py            # auto-cari kamera VRX
    py vrx_viewer.py 1          # paksa index kamera tertentu
Tombol:
    q / ESC  keluar
    s        simpan snapshot (vrx_snapshot_N.jpg)
"""
import sys
import time
import cv2

BACKEND = cv2.CAP_MSMF  # Media Foundation; CAP_DSHOW by-index tidak didukung di OpenCV 5


def open_cam(index):
    cap = cv2.VideoCapture(index, BACKEND)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap


def find_vrx():
    for i in range(6):
        cap = open_cam(i)
        if cap:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[+] kamera index {i} aktif ({w}x{h})")
            return i, cap
        print(f"[-] index {i} kosong")
    return None, None


def main():
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        cap = open_cam(idx)
    else:
        idx, cap = find_vrx()
    if cap is None:
        print("VRX tidak ditemukan. Pastikan USB tercolok dan tidak dipakai aplikasi lain.")
        sys.exit(1)

    win = f"VRX (cam {idx})"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    n_snap, frames, t0 = 0, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame gagal dibaca, coba lagi...")
            time.sleep(0.05)
            continue
        frames += 1
        fps = frames / max(time.time() - t0, 1e-6)
        cv2.putText(frame, f"{frame.shape[1]}x{frame.shape[0]}  {fps:.1f} fps",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            n_snap += 1
            fn = f"vrx_snapshot_{n_snap}.jpg"
            cv2.imwrite(fn, frame)
            print("disimpan:", fn)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
