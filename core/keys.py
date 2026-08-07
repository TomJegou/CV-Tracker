"""Lecture d'état des boutons souris (Windows) — distinct de l'injection Arduino."""
from ctypes import windll

from core.settings import SETTINGS

VK_LBUTTON = 0x01
VK_RBUTTON = 0x02


def is_left_mouse_pressed() -> bool:
    return bool(windll.user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)


def is_right_mouse_pressed() -> bool:
    return bool(windll.user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000)


def is_ads_and_firing() -> bool:
    """Clic droit (ADS) + clic gauche (tir) maintenus ensemble."""
    return is_left_mouse_pressed() and is_right_mouse_pressed()


def is_aim_trigger_active(
    *,
    require_lmb: bool | None = None,
    require_rmb: bool | None = None,
    require_both: bool | None = None,
) -> bool:
    """True si l'aim (lock/assist) doit s'appliquer selon les flags config.

    - require_both : LMB et RMB maintenus (prioritaire)
    - sinon : OU des boutons dont le flag est True
    - aucun flag : toujours actif
    """
    if require_lmb is None:
        require_lmb = SETTINGS.AIM_ASSIST_REQUIRE_LMB
    if require_rmb is None:
        require_rmb = SETTINGS.AIM_ASSIST_REQUIRE_RMB
    if require_both is None:
        require_both = SETTINGS.AIM_ASSIST_REQUIRE_BOTH

    if require_both:
        return is_left_mouse_pressed() and is_right_mouse_pressed()

    if not require_lmb and not require_rmb:
        return True

    if require_lmb and is_left_mouse_pressed():
        return True
    if require_rmb and is_right_mouse_pressed():
        return True
    return False


def describe_aim_trigger(
    *,
    require_lmb: bool | None = None,
    require_rmb: bool | None = None,
    require_both: bool | None = None,
) -> str:
    """Libellé court pour le statut au démarrage."""
    if require_lmb is None:
        require_lmb = SETTINGS.AIM_ASSIST_REQUIRE_LMB
    if require_rmb is None:
        require_rmb = SETTINGS.AIM_ASSIST_REQUIRE_RMB
    if require_both is None:
        require_both = SETTINGS.AIM_ASSIST_REQUIRE_BOTH

    if require_both:
        return "LMB+RMB (ET)"
    parts: list[str] = []
    if require_lmb:
        parts.append("LMB")
    if require_rmb:
        parts.append("RMB")
    if not parts:
        return "toujours actif"
    if len(parts) == 1:
        return parts[0]
    return " ou ".join(parts)
