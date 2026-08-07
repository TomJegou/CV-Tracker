import queue
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core import config
from core.capture import ScreenCapture
from core.collector import DataCollector
from core.detector import YoloDetector
from core.mouse import AimFirePull, MouseController, RecoilCompensator, move_mouse
from core.keys import (
    is_ads_and_firing,
    is_aim_trigger_active,
    is_left_mouse_pressed,
    is_right_mouse_pressed,
)
from core.settings import SETTINGS
from core.targeting import TargetingSystem

MOUSE_QUEUE_TIMEOUT_S = 0.05


class PipelineError(RuntimeError):
    """Un thread worker a échoué : la pipeline a été arrêtée."""


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


def _take_latest(q: queue.Queue):
    """Vide la queue et retourne le dernier item, ou None si vide."""
    latest = None
    try:
        while True:
            latest = q.get_nowait()
    except queue.Empty:
        pass
    return latest


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
        mouse: MouseController,
        collector: DataCollector | None = None,
        recoil: RecoilCompensator | None = None,
        fire_pull: AimFirePull | None = None,
    ):
        self._capture = capture
        self._detector = detector
        self._targeting = targeting
        self._mouse = mouse
        self._collector = collector
        self._recoil = recoil
        self._fire_pull = fire_pull
        self._capture_idle_sleep_s = config.CAPTURE_IDLE_SLEEP_S
        # Toujours tick fusionné : jitter / pull / aim peuvent s'activer à chaud.
        self._mouse_tick_s = config.NO_RECOIL_TICK_S
        self._no_recoil_debug = bool(config.NO_RECOIL_DEBUG)

        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._target_queue: queue.Queue[dict | None] = queue.Queue(maxsize=1)
        self._debug_queue: queue.Queue[DebugFrame] = queue.Queue(maxsize=1)

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._stopped = False
        self._failure: tuple[str, Exception] | None = None
        self._failure_lock = threading.Lock()
        self._collector_lock = threading.Lock()

    @classmethod
    def create(cls, model_path: Path | str | None = None) -> "AimPipeline":
        capture = ScreenCapture()
        detector = YoloDetector(model_path=model_path)
        targeting = TargetingSystem()
        # COM + compensateurs toujours prêts pour activation à chaud.
        mouse = MouseController()
        recoil = RecoilCompensator()
        fire_pull = AimFirePull()
        # DataCollector paresseux : cree a la premiere activation mining.
        collector = DataCollector() if SETTINGS.ENABLE_DATA_MINING else None
        return cls(capture, detector, targeting, mouse, collector, recoil, fire_pull)

    @property
    def detector(self) -> YoloDetector:
        return self._detector

    @property
    def capture_region(self) -> tuple[int, int, int, int]:
        return self._capture.region

    @property
    def data_mining_dir(self) -> Path | None:
        return self._collector.save_dir if self._collector is not None else None

    def is_running(self) -> bool:
        return self._started and not self._stop.is_set()

    def raise_if_failed(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is None:
            return
        name, exc = failure
        raise PipelineError(f"Thread '{name}' a échoué : {exc}") from exc

    def start(self) -> None:
        if self._stopped:
            raise RuntimeError(
                "AimPipeline est à usage unique : après stop(), recrée une "
                "instance via AimPipeline.create()."
            )
        if self._started:
            return

        self._mouse.open()

        workers = [
            ("capture", self._capture_loop),
            ("detect", self._detect_loop),
            ("mouse", self._mouse_loop),
        ]

        self._stop.clear()
        for name, loop in workers:
            thread = threading.Thread(
                target=self._guard,
                args=(name, loop),
                name=name,
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        self._started = True

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()
        self._capture.release()
        if self._collector is not None:
            self._collector.stop()
        self._mouse.close()

    def get_debug_frame(self, timeout: float = 0.05) -> DebugFrame | None:
        try:
            return self._debug_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _guard(self, name: str, loop: Callable[[], None]) -> None:
        try:
            loop()
        except Exception as exc:
            # Exception pendant un arrêt demandé = bruit de shutdown
            # (ex. grab() après capture.release() si le join a timeout).
            if self._stop.is_set():
                return
            with self._failure_lock:
                if self._failure is None:
                    self._failure = (name, exc)
            print(f"\n[pipeline] Thread '{name}' a crashé :")
            traceback.print_exc()
            self._stop.set()

    def _ensure_collector(self) -> DataCollector | None:
        """Cree le DataCollector a la premiere activation mining (thread-safe)."""
        if not SETTINGS.ENABLE_DATA_MINING:
            return None
        if self._collector is not None:
            return self._collector
        with self._collector_lock:
            if self._collector is None:
                self._collector = DataCollector()
                print(f"[pipeline] Data mining active -> {self._collector.save_dir}/")
            return self._collector

    def _capture_loop(self) -> None:
        idle_sleep = self._capture_idle_sleep_s
        while not self._stop.is_set():
            frame = self._capture.get_latest_frame()
            if frame is None:
                if idle_sleep > 0:
                    time.sleep(idle_sleep)
                continue
            put_latest(self._frame_queue, frame.copy())

    def _detect_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            aim_conf = SETTINGS.CONF_THRESHOLD
            mining_on = SETTINGS.ENABLE_DATA_MINING
            infer_conf = (
                min(aim_conf, config.DATA_MINING_CONF) if mining_on else aim_conf
            )

            detections = self._detector.detect(frame, conf=infer_conf)
            aim_detections = (
                self._detector.filter_by_conf(detections, aim_conf)
                if infer_conf < aim_conf
                else detections
            )
            best_target = self._targeting.get_best_target(aim_detections)
            if SETTINGS.AIM_ASSIST:
                put_latest(self._target_queue, best_target)

            if mining_on:
                collector = self._ensure_collector()
                if collector is not None:
                    collector.consider(
                        frame,
                        detections,
                        clicking=is_ads_and_firing(),
                    )

            if config.DEBUG or SETTINGS.OVERLAY:
                put_latest(
                    self._debug_queue,
                    DebugFrame(
                        frame=frame,
                        detections=detections,
                        best_target=best_target,
                    ),
                )

    def _mouse_loop(self) -> None:
        """Aim + jitter + fire-pull : tick fixe, un write Serial, flags live."""
        tick = self._mouse_tick_s
        debug = self._no_recoil_debug
        last_debug = 0.0
        recoil_sent = 0
        jitter_was_on = False
        pull_was_on = False

        while not self._stop.is_set():
            loop_start = time.perf_counter()
            target = _take_latest(self._target_queue)

            if self._stop.is_set():
                break

            now = time.perf_counter()
            aim_x = aim_y = 0
            if (
                SETTINGS.AIM_ASSIST
                and target is not None
                and is_aim_trigger_active(
                    require_lmb=SETTINGS.AIM_ASSIST_REQUIRE_LMB,
                    require_rmb=SETTINGS.AIM_ASSIST_REQUIRE_RMB,
                    require_both=SETTINGS.AIM_ASSIST_REQUIRE_BOTH,
                )
            ):
                aim_x, aim_y = self._mouse.compute_move(
                    target["dx"], target["dy"], target["distance"]
                )

            recoil_x = recoil_y = 0
            pull_y = 0
            ads_firing = is_ads_and_firing()
            jitter_on = SETTINGS.ACTIVE_JITTER
            pull_on = SETTINGS.ACTIVE_PULL_DOWN

            if jitter_was_on and not jitter_on and self._recoil is not None:
                self._recoil.reset()
            if pull_was_on and not pull_on and self._fire_pull is not None:
                self._fire_pull.reset()
            jitter_was_on = jitter_on
            pull_was_on = pull_on

            if ads_firing:
                if jitter_on and self._recoil is not None:
                    recoil_x, recoil_y = self._recoil.tick(now)
                if pull_on and self._fire_pull is not None:
                    pull_y = self._fire_pull.tick(now)
            else:
                if self._recoil is not None:
                    self._recoil.reset()
                if self._fire_pull is not None:
                    self._fire_pull.reset()

            out_x = aim_x + recoil_x
            out_y = aim_y + recoil_y + pull_y
            if out_x != 0 or out_y != 0:
                move_mouse(out_x, out_y)
                if recoil_x != 0 or recoil_y != 0:
                    recoil_sent += 1

            if debug and now - last_debug >= 0.5:
                last_debug = now
                pull_rate = (
                    self._fire_pull.current_rate
                    if self._fire_pull is not None and pull_on
                    else 0.0
                )
                print(
                    f"[mouse] LMB={int(is_left_mouse_pressed())} "
                    f"RMB={int(is_right_mouse_pressed())} "
                    f"ads_fire={int(ads_firing)} "
                    f"aim=<{aim_x},{aim_y}> jitter=<{recoil_x},{recoil_y}> "
                    f"pull_y={pull_y} pull_rate={pull_rate:.0f} "
                    f"jitter_ticks={recoil_sent}"
                )
                recoil_sent = 0

            elapsed = time.perf_counter() - loop_start
            sleep_s = tick - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
