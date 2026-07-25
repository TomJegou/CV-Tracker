import argparse
import sys
import time
from pathlib import Path

import cv2
from serial import SerialException

from core import config
from core.keys import describe_aim_trigger
from core.mouse import close_arduino_mouse
from core.pipeline import AimPipeline, PipelineError


def _print_status(pipeline: AimPipeline) -> None:
    print(f"Modèle : {pipeline.detector.model_path}")

    if config.DEBUG:
        print("Mode DEBUG — fenêtre OpenCV active, appuyez sur 'q' pour quitter.")
    else:
        print("Mode production — pas de rendu visuel, Ctrl+C pour quitter.")

    if config.AIM_ASSIST:
        trigger = describe_aim_trigger()
        print(f"Aim : activé — mode={config.AIM_MODE} ({trigger})")
    else:
        print("Aim : désactivé (détection seule)")

    if config.NO_RECOIL:
        print(
            f"No-recoil : activé — jitter X/Y "
            f"{config.NO_RECOIL_JITTER_MIN}–{config.NO_RECOIL_JITTER_MAX} px "
            f"(LMB+RMB, via Arduino)"
        )
        if config.NO_RECOIL_DEBUG:
            print("No-recoil DEBUG : logs [no-recoil] toutes les 0.5 s — "
                  "vérifie LMB=1 RMB=1 firing=1 en tirant")

    if config.ENABLE_DATA_MINING and pipeline.data_mining_dir is not None:
        infer = min(config.CONF_THRESHOLD, config.DATA_MINING_CONF)
        print(
            f"Data mining : activé → {pipeline.data_mining_dir}/ "
            f"(YOLO conf≥{infer:.2f}, "
            f"aim conf≥{config.CONF_THRESHOLD:.2f}, "
            f"fp [{config.DATA_MINING_UNCERTAIN_MIN:.2f}-{config.DATA_MINING_UNCERTAIN_MAX:.2f}], "
            f"fn conf<{config.DATA_MINING_FN_MAX_CONF:.2f} + LMB+RMB)"
        )

    print("Pipeline découplée : capture | detect | mouse")
    print(
        f"Classes : {', '.join(config.CLASS_NAMES)} — "
        f"cible aim : {config.CLASS_NAMES[config.TARGET_CLASS_ID]}"
    )


def _run_debug_ui(pipeline: AimPipeline) -> None:
    window_name = "CV-Tracker"
    fov_center = config.FOV_SIZE // 2
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 1000)

    while pipeline.is_running():
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline CV-Tracker.")
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=None,
        help=(
            "Chemin modèle (.pt / .engine). "
            "Défaut : dernier models/apex_* (.engine prioritaire)"
        ),
    )
    args = parser.parse_args()

    pipeline: AimPipeline | None = None
    exit_code = 0
    try:
        pipeline = AimPipeline.create(model_path=args.model)
        _print_status(pipeline)
        pipeline.start()

        if config.DEBUG:
            _run_debug_ui(pipeline)
        else:
            while pipeline.is_running():
                time.sleep(0.1)

        pipeline.raise_if_failed()
    except PipelineError as exc:
        print(f"\nArrêt : {exc}")
        exit_code = 1
    except FileNotFoundError as exc:
        print(exc)
        exit_code = 1
    except SerialException as exc:
        print(exc)
        exit_code = 1
    except KeyboardInterrupt:
        pass
    finally:
        if pipeline is not None:
            pipeline.stop()
        else:
            close_arduino_mouse()
        if config.DEBUG:
            cv2.destroyAllWindows()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
