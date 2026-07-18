import math

from core.config import FOV_SIZE


class TargetingSystem:
    def __init__(self, fov_size: int = FOV_SIZE):
        self._center = fov_size // 2

    def get_best_target(self, detections: list[dict]) -> dict | None:
        if not detections:
            return None

        scored: list[dict] = []
        for det in detections:
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
