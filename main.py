import cv2

from core.config import (
    AIM_ASSIST,
    AIM_ASSIST_REQUIRE_LMB,
    DATA_MINING_COOLDOWN,
    DATA_MINING_SAVE_DIR,
    DEBUG,
    ENABLE_DATA_MINING,
    FOV_SIZE,
)
from core.capture import ScreenCapture
from core.collector import DataCollector
from core.detector import YoloDetector
from core.mouse import MouseController, is_left_click_pressed, is_left_mouse_pressed
from core.targeting import TargetingSystem


def main() -> None:
    capture = ScreenCapture()
    detector = YoloDetector()
    print(f"Modèle : {detector.model_path}")
    targeting = TargetingSystem()
    mouse = MouseController() if AIM_ASSIST else None
    collector = (
        DataCollector(DATA_MINING_SAVE_DIR, DATA_MINING_COOLDOWN)
        if ENABLE_DATA_MINING
        else None
    )

    fov_center = FOV_SIZE // 2
    window_name = "CV-Tracker"

    if DEBUG:
        print("Mode DEBUG — fenêtre OpenCV active, appuyez sur 'q' pour quitter.")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 1000)
    else:
        print("Mode production — pas de rendu visuel, Ctrl+C pour quitter.")

    if AIM_ASSIST:
        if AIM_ASSIST_REQUIRE_LMB:
            print("Aim assist : activé (clic gauche maintenu)")
        else:
            print("Aim assist : activé")
    else:
        print("Aim assist : désactivé (détection seule)")

    if ENABLE_DATA_MINING:
        print(f"Data mining : activé → {DATA_MINING_SAVE_DIR}/")

    try:
        while True:
            frame = capture.get_latest_frame()
            if frame is None:
                continue

            detections = detector.detect(frame)
            best_target = targeting.get_best_target(detections)

            if AIM_ASSIST and best_target:
                trigger_active = not AIM_ASSIST_REQUIRE_LMB or is_left_mouse_pressed()
                if trigger_active:
                    mouse.move(
                        best_target["dx"],
                        best_target["dy"],
                        best_target["distance"],
                    )

            if ENABLE_DATA_MINING:
                if detections:
                    collector.add_image(frame, reason="detection")
                if is_left_click_pressed():
                    collector.add_image(frame, reason="click")

            if DEBUG:
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
        if DEBUG:
            cv2.destroyAllWindows()
        if not capture._camera.is_released:
            capture._camera.release()


if __name__ == "__main__":
    main()
