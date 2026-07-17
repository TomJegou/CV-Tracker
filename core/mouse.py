import ctypes

from core.config import AIM_MODE, MAGNETIC_RADIUS, MAX_SMOOTHING

MOUSEEVENTF_MOVE = 0x0001
VK_LBUTTON = 0x01


def is_left_mouse_pressed() -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)


class MouseController:
    def __init__(
        self,
        mode: str = AIM_MODE,
        max_smoothing: float = MAX_SMOOTHING,
        magnetic_radius: float = MAGNETIC_RADIUS,
    ):
        if mode not in ("lock", "assist"):
            raise ValueError(f"AIM_MODE invalide : {mode!r} (attendu 'lock' ou 'assist')")
        self.mode = mode
        self.max_smoothing = max_smoothing
        self.magnetic_radius = magnetic_radius

    def apply(self, dx: float, dy: float, distance: float, *, fresh: bool) -> None:
        if self.mode == "lock":
            if fresh:
                self._snap(dx, dy)
            return
        self._assist(dx, dy, distance)

    def _snap(self, dx: float, dy: float) -> None:
        move_x = int(round(dx))
        move_y = int(round(dy))
        if move_x == 0 and move_y == 0:
            return
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)

    def _assist(self, dx: float, dy: float, distance: float) -> None:
        if distance > self.magnetic_radius:
            return

        dynamic_smooth = self.max_smoothing * (1 - (distance / self.magnetic_radius))
        move_x = int(dx * dynamic_smooth)
        move_y = int(dy * dynamic_smooth)

        if move_x == 0 and move_y == 0:
            return

        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
