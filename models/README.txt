Folder model deteksi. Taruh file YOLO (.pt / .onnx / .engine) di sini.

best.pt  -> model utama (default yang dimuat saat server start)

Cara ganti model / atur threshold:
  - Dari web (http://localhost:8000), kartu "Model deteksi": pilih file, geser Confidence,
    pilih imgsz, klik Load. Pengaturan disimpan ke config.json dan dipakai lagi saat start berikutnya.
  - Atau dari argumen: py server.py --model best.pt --conf 0.5 --imgsz 640 --device 0

Menambah model baru: salin file ke folder ini, klik tombol 🔄 di kartu Model (tanpa restart).

Kecepatan (RTX 5060):
  - Default pakai GPU + FP16 otomatis. imgsz 640 ~ 3-6 ms/frame untuk model nano.
  - Kalau butuh lebih cepat: imgsz 416/320. Kalau butuh objek kecil terdeteksi: 960.
  - Ekspor ke TensorRT untuk ~2x lebih cepat lagi:
      py -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='engine', half=True)"
    lalu pilih best.engine dari web.
