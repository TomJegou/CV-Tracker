import time

import cv2
import dxcam
import numpy as np


class ScreenCapture:
    FOV_SIZE = 400

    def __init__(self, output_idx: int = 0):
        self._camera = dxcam.create(
            output_idx=output_idx,
            output_color="BGR",
        )
        if self._camera is None:
            raise RuntimeError("Impossible d'initialiser dxcam.")

        screen_width = self._camera.width
        screen_height = self._camera.height

        half = self.FOV_SIZE // 2
        center_x = screen_width // 2
        center_y = screen_height // 2

        self._region = (
            center_x - half,
            center_y - half,
            center_x + half,
            center_y + half,
        )

    @property
    def region(self) -> tuple[int, int, int, int]:
        return self._region

    def get_latest_frame(self) -> np.ndarray | None:
        return self._camera.grab(region=self._region)


if __name__ == "__main__":
    capture = ScreenCapture()
    print(f"FOV 400x400 centré — région: {capture.region}")
    print("Appuyez sur 'q' dans la fenêtre Debug pour quitter.")

    window_name = "Debug"
    frame_count = 0
    fps_timer = time.perf_counter()

    try:
        while True:
            frame = capture.get_latest_frame()
            if frame is None:
                continue

            cv2.imshow(window_name, frame)

            frame_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}")
                frame_count = 0
                fps_timer = time.perf_counter()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        if not capture._camera.is_released:
            capture._camera.release()
