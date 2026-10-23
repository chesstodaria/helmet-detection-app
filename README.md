# Safety Helmet Detection System

Aplikasi desktop (Python) untuk mendeteksi objek terkait penggunaan helm keselamatan pada gambar pekerja konstruksi, memakai model YOLO (Ultralytics), OpenCV, dan antarmuka CustomTkinter. Hasil deteksi berupa bounding box, label kelas, dan confidence score. Aplikasi berjalan lokal tanpa web server maupun database.

> Sistem ini adalah **alat bantu** pengawasan, bukan pengganti pemeriksaan keselamatan oleh manusia.

## Fitur

| Fitur | Keterangan |
|---|---|
| Deteksi gambar | Pilih gambar JPG/JPEG/PNG, jalankan deteksi, bandingkan gambar asli dengan hasil, simpan hasil |
| Confidence threshold | Slider 0.10–0.90 (nilai awal 0.25) |
| Ringkasan | Nama file, status, jumlah objek per kelas, waktu inferensi, nama model, dan confidence yang dipakai |
| Penanganan error | File tidak ditemukan/format tidak didukung, model tidak ada atau gagal dimuat, inferensi gagal, tidak ada objek, lokasi simpan tidak bisa diakses |
| Video (tambahan) | Proses file video dan simpan video hasil deteksi |
| Webcam live (tambahan) | Deteksi langsung dari kamera, dengan FPS yang diukur di perangkat |
| Evaluasi model | `evaluate.py`: Precision, Recall, mAP@0.5, mAP@0.5:0.95, confusion matrix, ekspor CSV |

Aplikasi **tidak** menyimpulkan "tidak memakai helm" dan tidak menghitung persentase kepatuhan. Model mendeteksi kelas `head`, `helmet`, dan `person`; tidak adanya kotak `helmet` pada seseorang belum membuktikan ia tidak memakai helm.

## Struktur proyek

```
app.py                          titik masuk aplikasi
detector.py                     pembungkus model YOLO (memuat bobot, menjalankan deteksi)
evaluate.py                     evaluasi model pada data beranotasi (FR-09)
resume_train.py                 melanjutkan training dari checkpoint last.pt
requirements.txt                dependensi dengan versi
views/main_window.py            tampilan (widget CustomTkinter)
controllers/app_controller.py   menghubungkan aksi UI dengan service
services/detection_service.py   inferensi gambar/frame
services/video_service.py       pemrosesan video di thread terpisah
services/webcam_service.py      deteksi webcam di thread terpisah
utils/image_utils.py            pemuatan gambar dan konversi warna
models/best.pt                  bobot model hasil training (tidak disertakan di repositori)
```

## Instalasi (Windows PowerShell)

Diuji dengan Python 3.14 di Windows. Jalankan dari folder proyek:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Pasang PyTorch **terlebih dahulu** (supaya `ultralytics` tidak memasang versi CPU):

```powershell
# GPU NVIDIA (CUDA 12.6)
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126

# atau CPU saja
python -m pip install torch==2.14.1 torchvision==0.29.1
```

Lalu pasang dependensi lain dan cek GPU:

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

Gunakan `python -m pip` (bukan `pip` langsung) supaya selalu memakai Python milik venv aktif.

Letakkan bobot model hasil training di `models/best.pt`. Model dimuat dari file tersebut; training tidak perlu diulang setiap aplikasi dijalankan.

## Menjalankan aplikasi

```powershell
python app.py
```

Alur utama: **Pilih Gambar → Jalankan Deteksi → lihat hasil** (opsional: Simpan Hasil). Gambar asli tidak pernah diubah; hasil disimpan sebagai file baru dengan nama dan lokasi pilihan pengguna.

## Evaluasi model

Evaluasi memakai split `test` yang tidak dipakai saat training, dengan label ground-truth dari dataset:

```powershell
python evaluate.py --data "D:\path\ke\dataset\data.yaml" --split test
```

Keluaran di `evaluation/<nama_run>/`: `metrics_summary.csv`, `metrics_per_class.csv`, `run_info.json` (model, data, parameter, versi library), `confusion_matrix*.png`, kurva PR/P/R/F1, dan contoh prediksi. Jangan melaporkan hasil split `val` sebagai performa akhir, karena split itu dipakai untuk memilih `best.pt`.

## Informasi model dan training

<!-- TODO: isi dari runs/.../finetune_v1/args.yaml dan results.csv -->

| Item | Nilai |
|---|---|
| Arsitektur / bobot awal | _(isi, mis. YOLOv8s pretrained COCO)_ |
| Epoch (selesai / dijadwalkan) | _(isi)_ |
| Ukuran gambar (`imgsz`) | _(isi)_ |
| Batch size | _(isi)_ |
| Kelas | `head`, `helmet`, `person` |
| Perangkat training | _(isi, mis. NVIDIA GeForce RTX 4050 Laptop GPU)_ |

## Dataset

<!-- TODO: isi sumber dan lisensi sesuai halaman/README dataset (NFR-06, FR-10) -->

- Nama dataset: Hard Hat Workers
- Sumber (URL): _(isi)_
- Lisensi: _(isi)_
- Pembagian data: _(isi jumlah gambar train / val / test)_

Gambar yang dipakai untuk pengujian harus terpisah dari gambar training; pemeriksaan kebocoran data (nomor gambar asli yang muncul di train dan test) didokumentasikan di laporan.

## Hasil evaluasi

<!-- TODO: salin dari evaluation/<nama_run>/metrics_summary.csv dan metrics_per_class.csv -->

| Metrik | Nilai |
|---|---|
| Precision | _(isi)_ |
| Recall | _(isi)_ |
| mAP@0.5 | _(isi)_ |
| mAP@0.5:0.95 | _(isi)_ |

## Contoh input dan output

<!-- TODO: taruh screenshot aktual di docs/screenshots/ lalu tautkan di sini -->

_(isi dengan contoh gambar input, hasil deteksi, dan kasus tanpa objek terdeteksi)_

## Keterbatasan

- Hasil sangat bergantung pada kualitas dataset, anotasi, dan variasi gambar; performa pada dataset uji belum otomatis mewakili kondisi konstruksi nyata.
- Kinerja dapat turun pada sudut pandang ekstrem, pencahayaan buruk, objek kecil atau jauh, dan oklusi (helm tertutup objek lain).
- Confidence score bukan jaminan bahwa deteksi benar.
- Jumlah deteksi adalah jumlah kotak, bukan jumlah orang unik. Pada video dan webcam, orang yang sama dihitung ulang di setiap frame.
- Model hanya mengenali kelas `head`, `helmet`, dan `person`. Status "tidak memakai helm" tidak disimpulkan otomatis.
- Fitur video dan webcam adalah tambahan; kecepatan real-time belum dijamin dan bergantung pada perangkat (angka FPS tampil di aplikasi).
- Sistem tidak dimaksudkan untuk peringatan otomatis atau penilaian kepatuhan keselamatan tanpa pemeriksaan manusia.

## Referensi dan teknologi

Ultralytics YOLO (`ultralytics`), PyTorch, OpenCV (`opencv-python`), CustomTkinter, Pillow, Matplotlib, PyYAML. Dataset dan bobot pretrained dicantumkan pada bagian Dataset dan Informasi model di atas.
