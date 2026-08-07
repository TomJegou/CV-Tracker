import math

from core.config import FOV_SIZE, TARGET_CLASS_ID
from core.settings import SETTINGS


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

        aim_point_x = float(SETTINGS.AIM_POINT_X)
        aim_point_y = float(SETTINGS.AIM_POINT_Y)

        scored: list[dict] = []
        for det in enemies:
            target = det.copy()
            # YOLO xywh = centre de box → offset vers le point configuré
            aim_x = target["x"] + (aim_point_x - 0.5) * target["w"]
            aim_y = target["y"] + (aim_point_y - 0.5) * target["h"]
            dx = aim_x - self._center
            dy = aim_y - self._center
            distance = math.sqrt(dx * dx + dy * dy)

            target["x"] = aim_x
            target["y"] = aim_y
            target["dx"] = dx
            target["dy"] = dy
            target["distance"] = distance
            scored.append(target)

        scored.sort(key=lambda t: t["distance"])
        return scored[0]
