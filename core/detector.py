import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from core.config import CLASS_NAMES, CONF_THRESHOLD, TARGET_CLASS_ID, resolve_active_model

# BGR — ennemi vert, allié bleu
DEBUG_CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 128, 0),
}


class YoloDetector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = CONF_THRESHOLD,
    ):
        self.model_path = Path(model_path) if model_path else resolve_active_model()
        self._is_engine = self.model_path.suffix == ".engine"
        self.conf_threshold = conf_threshold
        self._model = YOLO(str(self.model_path))

    def detect(self, frame: np.ndarray) -> list[dict]:
        predict_kwargs: dict = {
            "verbose": False,
            "conf": self.conf_threshold,
            "device": 0,
        }
        if not self._is_engine:
            predict_kwargs["quantize"] = "fp16"

        results = self._model(frame, **predict_kwargs)

        detections: list[dict] = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                x, y, w, h = box.xywh[0].tolist()
                detections.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "w": float(w),
                        "h": float(h),
                        "conf": float(box.conf[0]),
                        "class_id": class_id,
                        "class_name": self._class_name(class_id),
                    }
                )

        return detections

    @staticmethod
    def _class_name(class_id: int) -> str:
        if 0 <= class_id < len(CLASS_NAMES):
            return CLASS_NAMES[class_id]
        return f"cls_{class_id}"

    def draw_debug(
        self, frame: np.ndarray, detections: list[dict]
    ) -> np.ndarray:
        debug_frame = frame.copy()

        for det in detections:
            x = int(det["x"])
            y = int(det["y"])
            w = int(det["w"])
            h = int(det["h"])
            class_id = det.get("class_id", 0)
            color = DEBUG_CLASS_COLORS.get(class_id, (0, 255, 255))

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(debug_frame, (x, y), 3, (0, 0, 255), -1)

            label = f"{det.get('class_name', '?')} {det['conf']:.2f}"
            cv2.putText(
                debug_frame,
                label,
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

            if class_id == TARGET_CLASS_ID:
                cv2.drawMarker(
                    debug_frame,
                    (x, y),
                    (255, 0, 0),
                    cv2.MARKER_CROSS,
                    8,
                    1,
                )

        return debug_frame


if __name__ == "__main__":
    from core.capture import ScreenCapture
    from core.config import FOV_SIZE

    capture = ScreenCapture()
    detector = YoloDetector()

    print(f"Modèle : {detector.model_path}")
    print(f"Classes : {', '.join(CLASS_NAMES)}")
    print(f"FOV {FOV_SIZE}x{FOV_SIZE} centré — région: {capture.region}")
    print("Appuyez sur 'q' dans la fenêtre Debug pour quitter.")

    window_name = "Debug"
    frame_count = 0
    fps_timer = time.perf_counter()

    try:
        while True:
            frame = capture.get_latest_frame()
            if frame is None:
                continue

            detections = detector.detect(frame)
            debug_frame = detector.draw_debug(frame, detections)
            cv2.imshow(window_name, debug_frame)

            frame_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f} | Détections: {len(detections)}")
                frame_count = 0
                fps_timer = time.perf_counter()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        capture.release()
