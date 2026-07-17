import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.config import DATA_MINING_COOLDOWN, DATA_MINING_SAVE_DIR


class DataCollector:
    def __init__(
        self,
        save_dir: Path = DATA_MINING_SAVE_DIR,
        cooldown: float = DATA_MINING_COOLDOWN,
    ):
        self._save_dir = Path(save_dir)
        self._cooldown = cooldown
        self._last_capture_time = 0.0
        self._queue: queue.Queue[tuple[np.ndarray, str]] = queue.Queue()
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._worker = threading.Thread(
            target=self._process_queue,
            name="data-collector",
            daemon=True,
        )
        self._worker.start()

    def add_image(self, image: np.ndarray, reason: str) -> None:
        now = time.perf_counter()
        if now - self._last_capture_time < self._cooldown:
            return

        self._last_capture_time = now
        self._queue.put((image.copy(), reason))

    def _process_queue(self) -> None:
        while True:
            image, reason = self._queue.get()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            filename = f"{timestamp}_{reason}.jpg"
            cv2.imwrite(str(self._save_dir / filename), image)
            self._queue.task_done()
