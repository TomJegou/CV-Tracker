import ctypes

MOUSEEVENTF_MOVE = 0x0001


class MouseController:
    def __init__(self, max_smoothing: float = 0.8, magnetic_radius: float = 150.0):
        self.max_smoothing = max_smoothing
        self.magnetic_radius = magnetic_radius

    def move(self, dx: float, dy: float, distance: float) -> None:
        if distance > self.magnetic_radius:
            return

        dynamic_smooth = self.max_smoothing * (1 - (distance / self.magnetic_radius))
        move_x = int(dx * dynamic_smooth)
        move_y = int(dy * dynamic_smooth)

        if move_x == 0 and move_y == 0:
            return

        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
