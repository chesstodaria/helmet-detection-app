"""Layanan deteksi webcam live.

Webcam dibaca dan diproses di thread terpisah supaya GUI tidak freeze.
Hasilnya dikirim lewat `events` (queue.Queue); GUI cukup membacanya dari
main thread, misalnya dengan `after(30, ...)`. Service ini sengaja tidak
mengimpor customtkinter, jadi tidak terikat ke tampilan tertentu.

Jenis event yang dikirim:
    {"type": "frame", "image": <BGR ndarray>, "counts": {nama_kelas: jumlah},
     "fps": float, "inference_ms": float}
    {"type": "error", "message": str}
    {"type": "stopped"}
"""
import queue
import sys
import threading
from collections import Counter
from time import perf_counter

import cv2


class WebcamService:
    def __init__(self, detector, camera_index: int = 0):
        # detector: instance HelmetDetector (detector.py)
        self.detector = detector
        self.camera_index = camera_index
        self.events: queue.Queue = queue.Queue(maxsize=2)

        self._confidence = 0.4
        self._stop_event = threading.Event()
        self._thread = None

    # ---------- kontrol dari GUI ----------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_confidence(self, value: float) -> None:
        """Bisa dipanggil kapan saja, termasuk saat webcam berjalan (slider)."""
        self._confidence = float(value)

    def start(self, camera_index=None, confidence=None) -> None:
        if self.is_running:
            return
        if camera_index is not None:
            self.camera_index = camera_index
        if confidence is not None:
            self._confidence = float(confidence)

        self._clear_events()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None

    # ---------- internal ----------

    def _open_camera(self):
        # CAP_DSHOW membuat kamera terbuka jauh lebih cepat di Windows.
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.camera_index)
        return cap

    def _publish(self, event: dict) -> None:
        """Kirim event; jika antrean penuh, buang yang paling lama
        agar GUI selalu menerima frame terbaru (tidak menumpuk)."""
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            try:
                self.events.put_nowait(event)
            except queue.Full:
                pass

    def _clear_events(self) -> None:
        while True:
            try:
                self.events.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        cap = self._open_camera()
        if not cap.isOpened():
            cap.release()
            self._publish({
                "type": "error",
                "message": f"Kamera {self.camera_index} tidak bisa dibuka. "
                           "Pastikan kamera terpasang dan tidak sedang dipakai aplikasi lain.",
            })
            self._publish({"type": "stopped"})
            return

        fps = 0.0
        last_tick = perf_counter()

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._publish({
                        "type": "error",
                        "message": "Gagal membaca frame dari kamera.",
                    })
                    break

                # Cermin horizontal supaya terasa seperti bercermin.
                frame = cv2.flip(frame, 1)

                result = self.detector.detect(frame, self._confidence)

                counts = Counter(d["class_name"] for d in result["detections"])

                now = perf_counter()
                instant_fps = 1.0 / max(now - last_tick, 1e-6)
                last_tick = now
                # Haluskan FPS supaya angkanya tidak loncat-loncat.
                fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps

                self._publish({
                    "type": "frame",
                    "image": result["annotated_image"],  # BGR, sudah ada kotaknya
                    "counts": dict(counts),
                    "fps": fps,
                    "inference_ms": result["inference_seconds"] * 1000,
                })
        except Exception as exc:  # pragma: no cover - pengaman agar thread tidak mati diam-diam
            self._publish({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            cap.release()
            self._publish({"type": "stopped"})
