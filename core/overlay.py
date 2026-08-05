"""Overlay FOV click-through (Windows) — V1 layered OpenCV."""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import cv2
import numpy as np

from core.config import (
    CLASS_NAMES,
    FOV_SIZE,
    MAGNETIC_RADIUS,
    OVERLAY_SHOW_CROSSHAIR,
    OVERLAY_SHOW_MAGNETIC_RADIUS,
    TARGET_CLASS_ID,
)
from core.detector import DEBUG_CLASS_COLORS
from core.pipeline import AimPipeline, DebugFrame

# Fond chromakey (BGR) — rendu transparent via LWA_COLORKEY
_CHROMA_BGR = (255, 0, 255)  # magenta
_CHROMA_COLORREF = 0x00FF00FF  # 0x00BBGGRR

user32 = ctypes.windll.user32

GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
LWA_COLORKEY = 0x00000001
HWND_TOPMOST = -1
SWP_SHOWWINDOW = 0x0040
VK_F8 = 0x77

user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
user32.SetWindowLongW.restype = wintypes.LONG
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = wintypes.LONG
user32.SetLayeredWindowAttributes.argtypes = [
    wintypes.HWND,
    wintypes.COLORREF,
    wintypes.BYTE,
    wintypes.DWORD,
]
user32.SetLayeredWindowAttributes.restype = wintypes.BOOL
user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]
user32.SetWindowPos.restype = wintypes.BOOL
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.FindWindowW.restype = wintypes.HWND
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = wintypes.SHORT


def _key_down(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


class FovOverlay:
    """Fenêtre borderless topmost + click-through, calée sur le FOV capture."""

    WINDOW_NAME = "CV-Tracker Overlay"

    def __init__(
        self,
        region: tuple[int, int, int, int],
        *,
        fov_size: int = FOV_SIZE,
    ):
        left, top, right, bottom = region
        self._left = int(left)
        self._top = int(top)
        self._width = int(right - left)
        self._height = int(bottom - top)
        self._fov_size = fov_size
        self._center = fov_size // 2
        self._hwnd: int | None = None
        self._visible = True
        self._f8_was_down = False
        self._canvas = np.full(
            (self._height, self._width, 3),
            _CHROMA_BGR,
            dtype=np.uint8,
        )

    def open(self) -> None:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self._width, self._height)
        cv2.moveWindow(self.WINDOW_NAME, self._left, self._top)
        # Premier paint pour forcer la création du HWND
        cv2.imshow(self.WINDOW_NAME, self._canvas)
        cv2.waitKey(1)
        self._hwnd = self._find_hwnd()
        if self._hwnd:
            self._apply_window_styles(self._hwnd)

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except cv2.error:
            pass
        self._hwnd = None

    def _find_hwnd(self) -> int | None:
        hwnd = user32.FindWindowW(None, self.WINDOW_NAME)
        return int(hwnd) if hwnd else None

    def _apply_window_styles(self, hwnd: int) -> None:
        user32.SetWindowLongW(hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE)
        ex = (
            WS_EX_LAYERED
            | WS_EX_TRANSPARENT
            | WS_EX_TOPMOST
            | WS_EX_TOOLWINDOW
        )
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex)
        user32.SetLayeredWindowAttributes(hwnd, _CHROMA_COLORREF, 0, LWA_COLORKEY)
        user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            self._left,
            self._top,
            self._width,
            self._height,
            SWP_SHOWWINDOW,
        )

    def _poll_toggle(self) -> None:
        down = _key_down(VK_F8)
        if down and not self._f8_was_down:
            self._visible = not self._visible
            if self._hwnd:
                # Re-topmost à chaque show (jeux borderless peuvent passer devant)
                if self._visible:
                    user32.SetWindowPos(
                        self._hwnd,
                        HWND_TOPMOST,
                        self._left,
                        self._top,
                        self._width,
                        self._height,
                        SWP_SHOWWINDOW,
                    )
        self._f8_was_down = down

    def render(self, packet: DebugFrame | None) -> None:
        self._poll_toggle()
        canvas = self._canvas
        canvas[:] = _CHROMA_BGR

        if not self._visible:
            cv2.imshow(self.WINDOW_NAME, canvas)
            cv2.waitKey(1)
            return

        # Cadre FOV
        cv2.rectangle(
            canvas,
            (0, 0),
            (self._width - 1, self._height - 1),
            (0, 255, 255),
            1,
        )
        c = self._center
        if OVERLAY_SHOW_CROSSHAIR:
            cv2.drawMarker(
                canvas,
                (c, c),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                12,
                1,
            )
        if OVERLAY_SHOW_MAGNETIC_RADIUS:
            radius = int(round(MAGNETIC_RADIUS))
            if radius > 0:
                cv2.circle(canvas, (c, c), radius, (0, 200, 255), 1)

        if packet is not None:
            for det in packet.detections:
                self._draw_detection(canvas, det)
            if packet.best_target is not None:
                c = self._center
                tx = int(packet.best_target["x"])
                ty = int(packet.best_target["y"])
                cv2.line(canvas, (c, c), (tx, ty), (0, 128, 255), 2)
                cv2.circle(canvas, (tx, ty), 4, (0, 128, 255), -1)

        cv2.imshow(self.WINDOW_NAME, canvas)
        cv2.waitKey(1)
        # Certains WM re-appliquent le chrome après imshow
        if self._hwnd:
            user32.SetWindowPos(
                self._hwnd,
                HWND_TOPMOST,
                self._left,
                self._top,
                self._width,
                self._height,
                SWP_SHOWWINDOW,
            )

    def _draw_detection(self, canvas: np.ndarray, det: dict) -> None:
        x = float(det["x"])
        y = float(det["y"])
        w = float(det["w"])
        h = float(det["h"])
        class_id = int(det.get("class_id", 0))
        color = DEBUG_CLASS_COLORS.get(class_id, (0, 255, 255))

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        name = det.get("class_name") or (
            CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else "?"
        )
        label = f"{name} {det['conf']:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        if class_id == TARGET_CLASS_ID:
            cv2.drawMarker(
                canvas,
                (int(x), int(y)),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                8,
                1,
            )


def run_overlay_loop(pipeline: AimPipeline, *, max_fps: float = 60.0) -> None:
    """Boucle principale overlay — quitte sur Ctrl+C (pipeline stop) ou fermeture."""
    overlay = FovOverlay(pipeline.capture_region)
    overlay.open()
    frame_interval = 1.0 / max(1.0, max_fps)
    try:
        while pipeline.is_running():
            loop_start = time.perf_counter()
            packet = pipeline.get_debug_frame(timeout=0.02)
            overlay.render(packet)
            elapsed = time.perf_counter() - loop_start
            sleep_s = frame_interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        overlay.close()
