"""Coordinates UI actions and application services."""
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox
import threading
import queue

from services.detection_service import DetectionService
from services.video_service import VideoService
from services.webcam_service import WebcamService
from utils.image_utils import bgr_to_pil

WEBCAM_CAMERA_INDEX = 0  # ganti ke 1, 2, ... jika memakai kamera eksternal
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def format_detection_summary(detections):
    """Ringkasan jumlah objek per kelas, misalnya 'Jumlah objek: 3 (head: 1, helmet: 2)'."""
    if not detections:
        return "Tidak ada objek yang terdeteksi"
    counts = Counter(d["class_name"] for d in detections)
    detail = ", ".join(f"{name}: {n}" for name, n in sorted(counts.items()))
    return f"Jumlah objek: {len(detections)} ({detail})"


class AppController:
    def __init__(self, view):
        self.view = view
        self.model_path = Path(__file__).resolve().parents[1] / "models" / "best.pt"
        self.video_queue = queue.Queue()
        self.video_processing = False
        self.webcam_service = None
        self.webcam_active = False
        self.last_webcam_frame = None
        try:
            self.detection_service = DetectionService(self.model_path)
            self.video_service = VideoService(self.detection_service)
            if self.detection_service.is_ready:
                self.view.set_model_status("Model: best.pt berhasil dimuat")
                self.view.set_status("Status: Siap")
            else:
                self.view.set_model_status("Model: best.pt belum ditemukan")
                self.view.set_status("Status: menunggu file model")
        except Exception as error:
            self.detection_service = None
            self.video_service = None
            self.view.set_model_status("Model: gagal dimuat")
            self.view.set_status("Status: Model bermasalah")
            messagebox.showerror("Gagal Memuat Model", f"Model tidak dapat dimuat.\n\n{error}")

        view.on_select_image = self.select_image
        view.on_detect_image = self.detect_image
        view.on_save_result = self.save_result
        view.on_reset = self.reset
        view.on_select_video = self.select_video
        view.on_toggle_webcam = self.toggle_webcam
        view.protocol("WM_DELETE_WINDOW", self.on_close)
        view.after(100, self.poll_video_queue)
        view.after(30, self.poll_webcam_queue)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Pilih gambar",
            filetypes=[("File gambar", "*.jpg *.jpeg *.png"), ("Semua file", "*.*")]
        )
        if not path:
            return
        # FR-01: validasi keberadaan file dan format sebelum dimuat.
        if not Path(path).is_file():
            messagebox.showerror("File Tidak Ditemukan", f"File tidak ditemukan:\n{path}")
            return
        if Path(path).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            messagebox.showerror(
                "Format Tidak Didukung",
                "Format file tidak didukung. Gunakan gambar JPG, JPEG, atau PNG."
            )
            return
        try:
            from utils.image_utils import load_rgb_image
            image = load_rgb_image(path)
            self.view.set_original_image(image)
            self.view.clear_result_preview()
            self.view.set_file_label(f"File: {Path(path).name}")
            self.view.set_status("Status: Gambar berhasil dimuat")
            self.view.set_result_text("Jumlah objek: —")
            self.view.set_time_text("Waktu inferensi: —")
            self.view.set_review_text("Peringatan: menunggu proses deteksi")
            self.view.enable_save(False)
            self.current_image_path = path
            self.current_result = None
        except Exception as error:
            messagebox.showerror("Gagal Membuka Gambar", f"Gambar tidak dapat dibuka.\n\n{error}")

    def detect_image(self):
        path = getattr(self, "current_image_path", None)
        if not path:
            messagebox.showwarning("Pilih Gambar", "Pilih gambar terlebih dahulu.")
            return
        if not self.detection_service or not self.detection_service.is_ready:
            messagebox.showwarning("Model Belum Tersedia", f"Letakkan model di:\n{self.model_path}")
            return
        self.view.set_detect_enabled(False)
        self.view.set_status("Status: Sedang memproses...")
        self.view.update_idletasks()
        try:
            confidence = self.view.get_confidence()
            output = self.detection_service.detect_image(path, confidence)
            self.current_result = output["annotated_image"]
            self.view.set_result_image(self.current_result)
            detections = output["detections"]
            self.view.set_result_text(format_detection_summary(detections))
            self.view.set_time_text(
                f"Waktu inferensi: {output['inference_seconds']:.3f} detik  |  "
                f"Model: {self.model_path.name}  |  Confidence: {confidence:.2f}"
            )
            self.view.set_status("Status: Deteksi selesai")
            if detections:
                self.view.set_review_text("Peringatan: periksa hasil deteksi secara manual")
            else:
                self.view.set_review_text("Tidak ada objek terdeteksi. Ini bukan bukti pasti bahwa tidak ada helm.")
            self.view.enable_save(True)
        except Exception as error:
            self.view.set_status("Status: Deteksi gagal")
            messagebox.showerror("Deteksi Gagal", f"Model gagal memproses gambar.\n\n{error}")
        finally:
            self.view.set_detect_enabled(True)

    def save_result(self):
        result = getattr(self, "current_result", None)
        if result is None:
            messagebox.showwarning("Belum Ada Hasil", "Jalankan deteksi terlebih dahulu.")
            return
        # `or "webcam"`: snapshot webcam tidak punya file sumber (current_image_path bisa None).
        source = getattr(self, "current_image_path", None) or "webcam"
        path = filedialog.asksaveasfilename(
            title="Simpan hasil deteksi", defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile=f"{Path(source).stem}_hasil.jpg"
        )
        if not path:
            return
        try:
            result.save(path)
            self.view.set_status("Status: Hasil berhasil disimpan")
            messagebox.showinfo("Berhasil", f"Hasil deteksi disimpan di:\n{path}")
        except Exception as error:
            messagebox.showerror("Gagal Menyimpan", f"Hasil tidak dapat disimpan.\n\n{error}")

    def reset(self):
        self.current_image_path = None
        self.current_result = None
        self.view.reset_view()

    def select_video(self):
        if not self.detection_service or not self.detection_service.is_ready:
            messagebox.showwarning("Model Belum Tersedia", "Model best.pt belum berhasil dimuat.")
            return
        if self.video_processing:
            messagebox.showinfo("Video Sedang Diproses", "Tunggu sampai video saat ini selesai.")
            return
        video_path = filedialog.askopenfilename(
            title="Pilih video", filetypes=[("File video", "*.mp4 *.avi *.mov *.mkv"), ("Semua file", "*.*")]
        )
        if not video_path:
            return
        output_path = filedialog.asksaveasfilename(
            title="Simpan video hasil deteksi", defaultextension=".mp4",
            initialfile=f"{Path(video_path).stem}_hasil.mp4", filetypes=[("Video MP4", "*.mp4")]
        )
        if not output_path:
            return
        self.video_processing = True
        self.view.set_video_controls_enabled(False)
        self.view.set_status("Status: Menyiapkan video...")
        self.view.set_result_text("Jumlah objek: menunggu hasil video")
        self.view.set_time_text("Frame: menyiapkan...")
        self.view.set_progress(0, "Progres video: 0%")
        self.view.set_review_text("Peringatan: hasil deteksi perlu ditinjau manual")
        threading.Thread(
            target=self.video_service.process,
            args=(video_path, output_path, self.view.get_confidence(), self.video_queue),
            daemon=True
        ).start()

    def poll_video_queue(self):
        try:
            while True:
                item = self.video_queue.get_nowait()
                if item[0] == "progress":
                    _, percent, frame_number, total_frames = item
                    self.view.set_status(f"Status: Memproses video... {percent}%")
                    self.view.set_time_text(f"Frame: {frame_number:,}/{total_frames:,}")
                    self.view.set_progress(percent / 100, f"Progres video: {percent}%")
                elif item[0] == "done":
                    _, summary, output_path = item
                    self.video_processing = False
                    self.view.set_video_controls_enabled(True)
                    self.view.set_status("Status: Video selesai diproses")
                    self.view.set_result_text(summary)
                    self.view.set_time_text("Pemrosesan selesai")
                    self.view.set_progress(1, "Progres video: 100% — selesai")
                    self.view.set_review_text("Peringatan: jumlah deteksi bukan jumlah orang unik. Periksa hasil secara manual.")
                    messagebox.showinfo("Video Selesai", f"Video hasil deteksi disimpan di:\n{output_path}")
                elif item[0] == "error":
                    self.video_processing = False
                    self.view.set_video_controls_enabled(True)
                    self.view.set_status("Status: Pemrosesan video gagal")
                    self.view.set_progress(0, "Progres video: gagal")
                    messagebox.showerror("Gagal Memproses Video", item[1])
        except queue.Empty:
            pass
        finally:
            if self.view.winfo_exists():
                self.view.after(100, self.poll_video_queue)

    # ---------- Webcam live ----------
    def toggle_webcam(self):
        if self.webcam_active:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        if not self.detection_service or not self.detection_service.is_ready:
            messagebox.showwarning("Model Belum Tersedia", f"Letakkan model di:\n{self.model_path}")
            return
        if self.video_processing:
            messagebox.showinfo("Video Sedang Diproses", "Tunggu sampai video selesai, lalu nyalakan webcam.")
            return
        if self.webcam_service is None:
            self.webcam_service = WebcamService(self.detection_service.detector, WEBCAM_CAMERA_INDEX)
        self.current_result = None
        self.last_webcam_frame = None
        self.webcam_active = True
        self.view.set_webcam_running(True)
        self.view.set_status("Status: Membuka kamera...")
        self.view.set_result_text("Jumlah objek: —")
        self.view.set_time_text("FPS: —")
        self.view.set_review_text(
            "Peringatan: jumlah deteksi adalah jumlah kotak pada frame saat ini, "
            "bukan jumlah orang unik dan bukan bukti kepatuhan. Periksa secara manual."
        )
        self.webcam_service.start(confidence=self.view.get_confidence())

    def stop_webcam(self, error_message=None):
        if self.webcam_service:
            self.webcam_service.stop()
        self.webcam_active = False
        self.view.set_webcam_running(False)
        if self.last_webcam_frame is not None:
            # Frame terakhir tetap tampil dan bisa disimpan lewat tombol "Simpan Hasil".
            self.current_result = self.last_webcam_frame
            self.view.enable_save(True)
            self.view.set_status("Status: Webcam dihentikan (frame terakhir bisa disimpan)")
        else:
            self.view.clear_result_preview()
            self.view.set_status("Status: Webcam dihentikan")
        if error_message:
            self.view.set_status("Status: Webcam bermasalah")
            messagebox.showerror("Webcam Bermasalah", error_message)

    def _show_webcam_frame(self, item):
        image = bgr_to_pil(item["image"])
        self.last_webcam_frame = image
        self.view.set_live_frame(image)
        counts = item["counts"]
        total = sum(counts.values())
        detail = ", ".join(f"{name}: {n}" for name, n in sorted(counts.items())) or "tidak ada"
        self.view.set_status("Status: Webcam live berjalan")
        self.view.set_result_text(f"Objek pada frame ini: {total} ({detail})")
        self.view.set_time_text(f"FPS: {item['fps']:.1f}  |  Inferensi: {item['inference_ms']:.0f} ms")

    def poll_webcam_queue(self):
        try:
            if self.webcam_active and self.webcam_service:
                # Slider confidence berlaku langsung saat webcam berjalan.
                self.webcam_service.set_confidence(self.view.get_confidence())
                latest, error_message, stopped = None, None, False
                while True:
                    try:
                        item = self.webcam_service.events.get_nowait()
                    except queue.Empty:
                        break
                    if item["type"] == "frame":
                        latest = item  # hanya frame terbaru yang ditampilkan
                    elif item["type"] == "error":
                        error_message = item["message"]
                    elif item["type"] == "stopped":
                        stopped = True
                if latest is not None:
                    self._show_webcam_frame(latest)
                if stopped or error_message:
                    self.stop_webcam(error_message)
        finally:
            if self.view.winfo_exists():
                self.view.after(30, self.poll_webcam_queue)

    def on_close(self):
        """Lepas kamera sebelum jendela ditutup."""
        self.webcam_active = False
        if self.webcam_service:
            self.webcam_service.stop()
        self.view.destroy()
