import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.config import (
    CLASS_NAMES,
    DATA_MINING_COOLDOWN_FN,
    DATA_MINING_COOLDOWN_FP,
    DATA_MINING_FN_MAX_CONF,
    DATA_MINING_SAVE_DIR,
    DATA_MINING_UNCERTAIN_MAX,
    DATA_MINING_UNCERTAIN_MIN,
    TARGET_CLASS_ID,
)


class DataCollector:
    def __init__(
        self,
        save_dir: Path = DATA_MINING_SAVE_DIR,
        uncertain_min: float = DATA_MINING_UNCERTAIN_MIN,
        uncertain_max: float = DATA_MINING_UNCERTAIN_MAX,
        fn_max_conf: float = DATA_MINING_FN_MAX_CONF,
        cooldown_fp: float = DATA_MINING_COOLDOWN_FP,
        cooldown_fn: float = DATA_MINING_COOLDOWN_FN,
    ):
        self._save_dir = Path(save_dir)
        self._uncertain_min = uncertain_min
        self._uncertain_max = uncertain_max
        self._fn_max_conf = fn_max_conf
        self._cooldowns = {
            "fp_suspect": cooldown_fp,
            "fn_suspect": cooldown_fn,
            # Allié détecté (potentiel FP ou décor / ennemi mal classé en allie)
            "ally_fp_suspect": cooldown_fp,
            # Tir actif mais pas d'ennemi détecté : l'ennemi pourrait être classé "allie"
            "enemy_as_ally_suspect": cooldown_fn,
        }
        self._last_capture: dict[str, float] = {}
        self._queue: queue.Queue[tuple[np.ndarray, str]] = queue.Queue()
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._worker = threading.Thread(
            target=self._process_queue,
            name="data-collector",
            daemon=True,
        )
        self._worker.start()

    def consider(
        self,
        image: np.ndarray,
        detections: list[dict],
        *,
        clicking: bool,
    ) -> None:
        """Enqueue seulement les frames utiles pour corriger FP / FN.

        `clicking` doit être True quand LMB + RMB sont maintenus (ADS + tir).
        """
        enemies = [d for d in detections if d.get("class_id") == TARGET_CLASS_ID]
        ally_class_id = CLASS_NAMES.index("allie") if "allie" in CLASS_NAMES else 1
        allies = [d for d in detections if d.get("class_id") == ally_class_id]
        best_enemy_conf = max((d["conf"] for d in enemies), default=0.0)
        best_ally_conf = max((d["conf"] for d in allies), default=0.0)

        if enemies and any(
            self._uncertain_min <= det["conf"] <= self._uncertain_max
            for det in enemies
        ):
            self._add_image(image, "fp_suspect")

        # Allié détecté avec une confiance "suspecte" : potentiel faux positif
        if allies and any(
            self._uncertain_min <= det["conf"] <= self._uncertain_max
            for det in allies
        ):
            self._add_image(image, "ally_fp_suspect")

        # Tir actif mais pas d'ennemi détecté : l'ennemi pourrait être classé "allie"
        if clicking and best_enemy_conf < self._fn_max_conf and allies:
            # On garde uniquement les cas où le modèle "pense" réellement à un allie
            if best_ally_conf >= self._uncertain_min:
                self._add_image(image, "enemy_as_ally_suspect")

        if clicking and best_enemy_conf < self._fn_max_conf:
            self._add_image(image, "fn_suspect")

    def _add_image(self, image: np.ndarray, reason: str) -> None:
        now = time.perf_counter()
        cooldown = self._cooldowns.get(reason, 0.5)
        last = self._last_capture.get(reason, 0.0)
        if now - last < cooldown:
            return

        self._last_capture[reason] = now
        self._queue.put((image.copy(), reason))

    def _process_queue(self) -> None:
        while True:
            image, reason = self._queue.get()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            filename = f"{timestamp}_{reason}.jpg"
            cv2.imwrite(str(self._save_dir / filename), image)
            self._queue.task_done()
