import queue
import threading
from dataclasses import dataclass

import numpy as np

from core import config
from core.capture import ScreenCapture
from core.collector import DataCollector
from core.detector import YoloDetector
from core.mouse import MouseController, is_left_mouse_pressed
from core.targeting import TargetingSystem


def put_latest(q: queue.Queue, item) -> None:
    """Remplace le contenu de la queue pour ne garder que la valeur la plus récente."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)


@dataclass
class DebugFrame:
    frame: np.ndarray
    detections: list[dict]
    best_target: dict | None


class AimPipeline:
    def __init__(
        self,
        capture: ScreenCapture,
        detector: YoloDetector,
        targeting: TargetingSystem,
        mouse: MouseController | None = None,
        collector: DataCollector | None = None,
        *,
        aim_assist: bool = config.AIM_ASSIST,
        aim_assist_require_lmb: bool = config.AIM_ASSIST_REQUIRE_LMB,
        enable_data_mining: bool = config.ENABLE_DATA_MINING,
        debug: bool = config.DEBUG,
    ):
        self._capture = capture
        self._detector = detector
        self._targeting = targeting
        self._mouse = mouse
        self._collector = collector
        self._aim_assist = aim_assist
        self._aim_assist_require_lmb = aim_assist_require_lmb
        self._enable_data_mining = enable_data_mining
        self._debug = debug

        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._target_queue: queue.Queue[dict | None] = queue.Queue(maxsize=1)
        self._debug_queue: queue.Queue[DebugFrame] = queue.Queue(maxsize=1)

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    @classmethod
    def create(cls) -> "AimPipeline":
        capture = ScreenCapture()
        detector = YoloDetector()
        targeting = TargetingSystem()
        mouse = MouseController() if config.AIM_ASSIST else None
        collector = DataCollector() if config.ENABLE_DATA_MINING else None
        return cls(capture, detector, targeting, mouse, collector)

    @property
    def detector(self) -> YoloDetector:
        return self._detector

    def start(self) -> None:
        if self._threads:
            return

        workers = [
            ("capture", self._capture_loop),
            ("detect", self._detect_loop),
        ]
        if self._aim_assist and self._mouse is not None:
            workers.append(("mouse", self._mouse_loop))

        self._stop.clear()
        for name, target in workers:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()
        self._capture.release()

    def get_debug_frame(self, timeout: float = 0.05) -> DebugFrame | None:
        try:
            return self._debug_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            frame = self._capture.get_latest_frame()
            if frame is None:
                continue
            put_latest(self._frame_queue, frame.copy())

    def _detect_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            detections = self._detector.detect(frame)
            best_target = self._targeting.get_best_target(detections)
            put_latest(self._target_queue, best_target)

            if self._enable_data_mining and self._collector is not None:
                self._collector.consider(
                    frame,
                    detections,
                    clicking=is_left_mouse_pressed(),
                )

            if self._debug:
                put_latest(
                    self._debug_queue,
                    DebugFrame(
                        frame=frame,
                        detections=detections,
                        best_target=best_target,
                    ),
                )

    def _mouse_loop(self) -> None:
        assert self._mouse is not None

        while not self._stop.is_set():
            try:
                target = self._target_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if target is None:
                continue

            if self._aim_assist_require_lmb and not is_left_mouse_pressed():
                continue

            self._mouse.apply(target["dx"], target["dy"], target["distance"])
