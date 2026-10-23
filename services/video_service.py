"""Video processing service. It never reads or modifies GUI widgets."""
from pathlib import Path
import cv2


class VideoService:
    def __init__(self, detection_service):
        self.detection_service = detection_service

    def process(self, video_path, output_path, confidence, event_queue):
        """Process video and report ('progress', ...) / ('done', ...) / ('error', ...)."""
        cap = None
        writer = None
        output_path = str(Path(output_path).with_suffix(".mp4"))
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError("Video sumber tidak dapat dibuka.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or width <= 0 or height <= 0:
                raise RuntimeError("Informasi video tidak valid.")

            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                (width, height), True
            )
            if not writer.isOpened():
                raise RuntimeError(
                    "Gagal membuat video output. Coba lokasi penyimpanan lain."
                )

            frame_number = 0
            counts = {"head": 0, "helmet": 0, "person": 0}
            last_percent = -1
            while True:
                success, frame = cap.read()
                if not success:
                    break
                result = self.detection_service.predict_frame(frame, confidence)
                writer.write(result.plot())
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    if class_name in counts:
                        counts[class_name] += 1
                frame_number += 1
                percent = min(100, int(frame_number * 100 / total_frames)) if total_frames > 0 else 0
                if percent != last_percent or frame_number % 10 == 0:
                    event_queue.put(("progress", percent, frame_number, total_frames))
                    last_percent = percent

            if frame_number == 0:
                raise RuntimeError("Tidak ada frame yang berhasil dibaca.")
            summary = (
                f"Frame diproses: {frame_number}\n"
                f"Deteksi head: {counts['head']}\n"
                f"Deteksi helmet: {counts['helmet']}\n"
                f"Deteksi person: {counts['person']}"
            )
            cap.release(); cap = None
            writer.release(); writer = None
            event_queue.put(("done", summary, output_path))
        except Exception as error:
            event_queue.put(("error", str(error)))
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
