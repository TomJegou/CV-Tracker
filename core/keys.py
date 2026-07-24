"""Lecture d'état des boutons souris (Windows) — distinct de l'injection Arduino."""
from ctypes import windll

VK_LBUTTON = 0x01
VK_RBUTTON = 0x02


def is_left_mouse_pressed() -> bool:
    return bool(windll.user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)


def is_right_mouse_pressed() -> bool:
    return bool(windll.user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000)


def is_ads_and_firing() -> bool:
    """Clic droit (ADS) + clic gauche (tir) maintenus ensemble."""
    return is_left_mouse_pressed() and is_right_mouse_pressed()
