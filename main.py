import time

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
from core.mouse import MouseController
from core.pipeline import AimPipeline
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

    pipeline = AimPipeline(
        capture,
        detector,
        targeting,
        mouse,
        collector,
        aim_assist=AIM_ASSIST,
        aim_assist_require_lmb=AIM_ASSIST_REQUIRE_LMB,
        enable_data_mining=ENABLE_DATA_MINING,
        debug=DEBUG,
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

    print("Pipeline découplée : capture | detect | mouse")

    pipeline.start()

    try:
        while True:
            if DEBUG:
                packet = pipeline.get_debug_frame(timeout=0.05)
                if packet is not None:
                    debug_frame = detector.draw_debug(packet.frame, packet.detections)
                    if packet.best_target:
                        cv2.line(
                            debug_frame,
                            (fov_center, fov_center),
                            (
                                int(packet.best_target["x"]),
                                int(packet.best_target["y"]),
                            ),
                            (255, 0, 0),
                            2,
                        )
                    cv2.imshow(window_name, debug_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Garde le process vivant ; capture/detect/mouse tournent en background
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if DEBUG:
            cv2.destroyAllWindows()
        if not capture._camera.is_released:
            capture._camera.release()


if __name__ == "__main__":
    main()
