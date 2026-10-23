
from pathlib import Path
from time import perf_counter

from ultralytics import YOLO


class HelmetDetector:
    """Memuat model YOLO dan menjalankan deteksi pada gambar."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"File model tidak ditemukan: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def detect(self, image_path: str, confidence: float = 0.25):
        """Mengembalikan hasil deteksi dan waktu inferensi."""
        start = perf_counter()

        results = self.model.predict(
            source=image_path,
            conf=confidence,
            verbose=False,
        )

        elapsed = perf_counter() - start
        result = results[0]

        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            coordinates = [
                round(float(value), 2)
                for value in box.xyxy[0].tolist()
            ]

            detections.append({
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": score,
                "bbox": coordinates,
            })

        # Gambar hasil anotasi dari YOLO, dalam format BGR OpenCV.
        annotated_image = result.plot()

        return {
            "detections": detections,
            "annotated_image": annotated_image,
            "inference_seconds": elapsed,
            "image_size": result.orig_shape,
        }
