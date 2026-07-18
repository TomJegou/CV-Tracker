from ctypes import POINTER, Structure, Union, byref, c_ulong, sizeof, windll
from ctypes import wintypes

from core.config import (
    AIM_DEBUG_MOVES,
    AIM_MODE,
    LOCK_SCALE,
    MAGNETIC_RADIUS,
    MAX_SMOOTHING,
)

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
VK_LBUTTON = 0x01


class MOUSEINPUT(Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", POINTER(c_ulong)),
    )


class _INPUTUNION(Union):
    _fields_ = (("mi", MOUSEINPUT),)


class INPUT(Structure):
    _anonymous_ = ("_input",)
    _fields_ = (
        ("type", wintypes.DWORD),
        ("_input", _INPUTUNION),
    )


def is_left_mouse_pressed() -> bool:
    return bool(windll.user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)


def _send_relative_move(dx: int, dy: int) -> None:
    if dx == 0 and dy == 0:
        return
    inp = INPUT(
        type=INPUT_MOUSE,
        mi=MOUSEINPUT(dx=dx, dy=dy, mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=None),
    )
    windll.user32.SendInput(1, byref(inp), sizeof(INPUT))


class MouseController:
    def __init__(
        self,
        mode: str = AIM_MODE,
        lock_scale: float = LOCK_SCALE,
        max_smoothing: float = MAX_SMOOTHING,
        magnetic_radius: float = MAGNETIC_RADIUS,
        debug_moves: bool = AIM_DEBUG_MOVES,
    ):
        if mode not in ("lock", "assist"):
            raise ValueError(f"AIM_MODE invalide : {mode!r} (attendu 'lock' ou 'assist')")
        self.mode = mode
        self.lock_scale = lock_scale
        self.max_smoothing = max_smoothing
        self.magnetic_radius = magnetic_radius
        self.debug_moves = debug_moves

    def apply(self, dx: float, dy: float, distance: float) -> None:
        if self.mode == "lock":
            self._lock(dx, dy)
        else:
            self._assist(dx, dy, distance)

    def _lock(self, dx: float, dy: float) -> None:
        """Snap direct : vecteur complet vers la cible (effet téléportation)."""
        move_x = int(round(dx * self.lock_scale))
        move_y = int(round(dy * self.lock_scale))
        if move_x == 0 and move_y == 0:
            return

        _send_relative_move(move_x, move_y)
        if self.debug_moves:
            print(f"[aim] SNAP dx={dx:.0f} dy={dy:.0f} -> move=({move_x},{move_y})")

    def _assist(self, dx: float, dy: float, distance: float) -> None:
        if distance > self.magnetic_radius:
            return

        dynamic_smooth = self.max_smoothing * (1 - (distance / self.magnetic_radius))
        move_x = int(dx * dynamic_smooth)
        move_y = int(dy * dynamic_smooth)
        _send_relative_move(move_x, move_y)
