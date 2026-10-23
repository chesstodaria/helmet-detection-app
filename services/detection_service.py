"""Image detection service; keeps model inference out of the UI layer."""
from pathlib import Path
from detector import HelmetDetector
from utils.image_utils import bgr_to_pil


class DetectionService:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.detector = None
        if self.model_path.is_file():
            self.detector = HelmetDetector(str(self.model_path))

    @property
    def is_ready(self):
        return self.detector is not None

    def detect_image(self, image_path, confidence):
        if not self.is_ready:
            raise RuntimeError(f"Model belum tersedia: {self.model_path}")
        output = self.detector.detect(str(image_path), confidence=float(confidence))
        return {
            "annotated_image": bgr_to_pil(output["annotated_image"]),
            "detections": output["detections"],
            "inference_seconds": output["inference_seconds"],
        }

    def predict_frame(self, frame, confidence):
        """Return a YOLO result for a BGR video frame."""
        if not self.is_ready:
            raise RuntimeError("Model belum berhasil dimuat.")
        return self.detector.model.predict(
            source=frame, conf=float(confidence), verbose=False
        )[0]
