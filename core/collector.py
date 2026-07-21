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
    DATA_MINING_UNCERTAIN_MAX,
    DATA_MINING_UNCERTAIN_MIN,
    TARGET_CLASS_ID,
)
from core.dataset_paths import create_data_mining_session_dir

# Raisons sans box YOLO (faux négatif suspect) → .txt vide
_EMPTY_LABEL_REASONS = frozenset({"fn_suspect"})


def detections_to_yolo_text(
    detections: list[dict],
    image_width: int,
    image_height: int,
) -> str:
    """Convertit des détections pixel (xywh) en lignes YOLO normalisées."""
    if image_width <= 0 or image_height <= 0 or not detections:
        return ""

    lines: list[str] = []
    for det in detections:
        class_id = int(det.get("class_id", 0))
        x_center = float(det["x"]) / image_width
        y_center = float(det["y"]) / image_height
        width = float(det["w"]) / image_width
        height = float(det["h"]) / image_height
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    return "\n".join(lines) + "\n"


class DataCollector:
    def __init__(
        self,
        save_dir: Path | None = None,
        uncertain_min: float = DATA_MINING_UNCERTAIN_MIN,
        uncertain_max: float = DATA_MINING_UNCERTAIN_MAX,
        fn_max_conf: float = DATA_MINING_FN_MAX_CONF,
        cooldown_fp: float = DATA_MINING_COOLDOWN_FP,
        cooldown_fn: float = DATA_MINING_COOLDOWN_FN,
    ):
        self._save_dir = Path(save_dir) if save_dir else create_data_mining_session_dir()
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
        # (image, reason, label_text) — I/O disque uniquement dans le worker
        self._queue: queue.Queue[tuple[np.ndarray, str, str]] = queue.Queue()
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._worker = threading.Thread(
            target=self._process_queue,
            name="data-collector",
            daemon=True,
        )
        self._worker.start()

    @property
    def save_dir(self) -> Path:
        return self._save_dir

    def consider(
        self,
        image: np.ndarray,
        detections: list[dict],
        *,
        clicking: bool,
    ) -> None:
        """Enqueue seulement les frames utiles pour corriger FP / FN.

        `clicking` doit être True quand LMB + RMB sont maintenus (ADS + tir).
        Les labels YOLO sont préparés ici (léger) ; l'écriture disque reste async.
        """
        enemies = [d for d in detections if d.get("class_id") == TARGET_CLASS_ID]
        ally_class_id = CLASS_NAMES.index("allie") if "allie" in CLASS_NAMES else 1
        allies = [d for d in detections if d.get("class_id") == ally_class_id]
        best_enemy_conf = max((d["conf"] for d in enemies), default=0.0)
        best_ally_conf = max((d["conf"] for d in allies), default=0.0)

        height, width = image.shape[:2]
        yolo_text = detections_to_yolo_text(detections, width, height)

        if enemies and any(
            self._uncertain_min <= det["conf"] <= self._uncertain_max
            for det in enemies
        ):
            self._add_image(image, "fp_suspect", yolo_text)

        # Allié détecté avec une confiance "suspecte" : potentiel faux positif
        if allies and any(
            self._uncertain_min <= det["conf"] <= self._uncertain_max
            for det in allies
        ):
            self._add_image(image, "ally_fp_suspect", yolo_text)

        # Tir actif mais pas d'ennemi détecté : l'ennemi pourrait être classé "allie"
        if clicking and best_enemy_conf < self._fn_max_conf and allies:
            if best_ally_conf >= self._uncertain_min:
                self._add_image(image, "enemy_as_ally_suspect", yolo_text)

        if clicking and best_enemy_conf < self._fn_max_conf:
            self._add_image(image, "fn_suspect", "")

    def _add_image(self, image: np.ndarray, reason: str, label_text: str) -> None:
        now = time.perf_counter()
        cooldown = self._cooldowns.get(reason, 0.5)
        last = self._last_capture.get(reason, 0.0)
        if now - last < cooldown:
            return

        if reason in _EMPTY_LABEL_REASONS:
            label_text = ""

        self._last_capture[reason] = now
        self._queue.put((image.copy(), reason, label_text))

    def _process_queue(self) -> None:
        while True:
            image, reason, label_text = self._queue.get()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            stem = f"{timestamp}_{reason}"
            cv2.imwrite(str(self._save_dir / f"{stem}.jpg"), image)
            (self._save_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")
            self._queue.task_done()
