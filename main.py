import time

import cv2

from core import config
from core.pipeline import AimPipeline


def _print_status(pipeline: AimPipeline) -> None:
    print(f"Modèle : {pipeline.detector.model_path}")

    if config.DEBUG:
        print("Mode DEBUG — fenêtre OpenCV active, appuyez sur 'q' pour quitter.")
    else:
        print("Mode production — pas de rendu visuel, Ctrl+C pour quitter.")

    if config.AIM_ASSIST:
        trigger = "clic gauche" if config.AIM_ASSIST_REQUIRE_LMB else "toujours actif"
        print(f"Aim : activé — mode={config.AIM_MODE} ({trigger})")
    else:
        print("Aim : désactivé (détection seule)")

    if config.ENABLE_DATA_MINING:
        print(
            f"Data mining : activé → {config.DATA_MINING_SAVE_DIR}/ "
            f"(fp [{config.DATA_MINING_UNCERTAIN_MIN:.2f}-{config.DATA_MINING_UNCERTAIN_MAX:.2f}], "
            f"fn conf<{config.DATA_MINING_FN_MAX_CONF:.2f} + LMB+RMB)"
        )

    print("Pipeline découplée : capture | detect | mouse")


def _run_debug_ui(pipeline: AimPipeline) -> None:
    window_name = "CV-Tracker"
    fov_center = config.FOV_SIZE // 2
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 1000)

    while True:
        packet = pipeline.get_debug_frame(timeout=0.05)
        if packet is not None:
            debug_frame = pipeline.detector.draw_debug(packet.frame, packet.detections)
            if packet.best_target:
                cv2.line(
                    debug_frame,
                    (fov_center, fov_center),
                    (int(packet.best_target["x"]), int(packet.best_target["y"])),
                    (255, 0, 0),
                    2,
                )
            cv2.imshow(window_name, debug_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> None:
    pipeline = AimPipeline.create()
    _print_status(pipeline)
    pipeline.start()

    try:
        if config.DEBUG:
            _run_debug_ui(pipeline)
        else:
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if config.DEBUG:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
