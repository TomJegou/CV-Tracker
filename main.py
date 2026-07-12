import cv2

from core.capture import ScreenCapture
from core.detector import YoloDetector
from core.mouse import MouseController
from core.targeting import TargetingSystem


def main() -> None:
    capture = ScreenCapture()
    detector = YoloDetector()
    targeting = TargetingSystem()
    mouse = MouseController()

    fov_center = targeting.fov_size // 2
    window_name = "CV-Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 1000) # Affiche la fenêtre en 800x800

    try:
        while True:
            frame = capture.get_latest_frame()
            if frame is None:
                continue

            detections = detector.detect(frame)
            best_target = targeting.get_best_target(detections)

            if best_target:
                mouse.move(
                    best_target["dx"],
                    best_target["dy"],
                    best_target["distance"],
                )

            debug_frame = detector.draw_debug(frame, detections)
            if best_target:
                cv2.line(
                    debug_frame,
                    (fov_center, fov_center),
                    (int(best_target["x"]), int(best_target["y"])),
                    (255, 0, 0),
                    2,
                )

            cv2.imshow(window_name, debug_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if not capture._camera.is_released:
            capture._camera.release()


if __name__ == "__main__":
    main()
