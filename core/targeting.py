import math

from core.config import FOV_SIZE, TARGET_CLASS_ID


class TargetingSystem:
    def __init__(
        self,
        fov_size: int = FOV_SIZE,
        target_class_id: int = TARGET_CLASS_ID,
    ):
        self._center = fov_size // 2
        self._target_class_id = target_class_id

    def get_best_target(self, detections: list[dict]) -> dict | None:
        enemies = [
            det for det in detections if det.get("class_id") == self._target_class_id
        ]
        if not enemies:
            return None

        scored: list[dict] = []
        for det in enemies:
            target = det.copy()
            dx = target["x"] - self._center
            dy = target["y"] - self._center
            distance = math.sqrt(dx * dx + dy * dy)

            target["dx"] = dx
            target["dy"] = dy
            target["distance"] = distance
            scored.append(target)

        scored.sort(key=lambda t: t["distance"])
        return scored[0]
