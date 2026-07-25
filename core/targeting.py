import math

from core.config import AIM_POINT_X, AIM_POINT_Y, FOV_SIZE, TARGET_CLASS_ID


class TargetingSystem:
    def __init__(
        self,
        fov_size: int = FOV_SIZE,
        target_class_id: int = TARGET_CLASS_ID,
        aim_point_x: float = AIM_POINT_X,
        aim_point_y: float = AIM_POINT_Y,
    ):
        self._center = fov_size // 2
        self._target_class_id = target_class_id
        self._aim_point_x = aim_point_x
        self._aim_point_y = aim_point_y

    def get_best_target(self, detections: list[dict]) -> dict | None:
        enemies = [
            det for det in detections if det.get("class_id") == self._target_class_id
        ]
        if not enemies:
            return None

        scored: list[dict] = []
        for det in enemies:
            target = det.copy()
            # YOLO xywh = centre de box → offset vers le point configuré
            aim_x = target["x"] + (self._aim_point_x - 0.5) * target["w"]
            aim_y = target["y"] + (self._aim_point_y - 0.5) * target["h"]
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
