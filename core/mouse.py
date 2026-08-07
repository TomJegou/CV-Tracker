"""Contrôle souris via Arduino Leonardo (Serial <dx,dy>) — plus de SendInput."""
from __future__ import annotations

import atexit
import random
import time

import serial
from serial import SerialException, SerialTimeoutException

from core.arduino_port import resolve_arduino_port
from core.config import (
    AIM_DEBUG_MOVES,
    AIM_FIRE_PULL_DECAY_S,
    AIM_FIRE_PULL_DY_PER_S,
    AIM_FIRE_PULL_PEAK_DURATION_S,
    AIM_FIRE_PULL_PEAK_DY_PER_S,
    AIM_MODE,
    ARDUINO_BAUD,
    ARDUINO_OPEN_RETRIES,
    ARDUINO_OPEN_RETRY_S,
    ARDUINO_PORT,
    ARDUINO_SETTLE_S,
    LOCK_SCALE,
    NO_RECOIL_JITTER_MAX,
    NO_RECOIL_JITTER_MIN,
)
from core.settings import SETTINGS


class ArduinoMouse:
    """Envoie des deltas relatifs à l'Arduino sur le port série (non-bloquant)."""

    def __init__(
        self,
        port: str | None = ARDUINO_PORT,
        baudrate: int = ARDUINO_BAUD,
        *,
        settle_s: float = ARDUINO_SETTLE_S,
        open_retries: int = ARDUINO_OPEN_RETRIES,
        open_retry_s: float = ARDUINO_OPEN_RETRY_S,
    ):
        self.port = resolve_arduino_port(port)
        self.baudrate = baudrate
        self._settle_s = settle_s
        self._open_retries = max(1, open_retries)
        self._open_retry_s = open_retry_s
        self._serial: serial.Serial | None = None
        self._open()

    def _open(self) -> None:
        last_err: Exception | None = None
        for attempt in range(1, self._open_retries + 1):
            try:
                self._serial = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=0,
                    write_timeout=0,
                )
                break
            except SerialException as exc:
                last_err = exc
                if attempt < self._open_retries:
                    time.sleep(self._open_retry_s)
                else:
                    raise SerialException(
                        f"Impossible d'ouvrir {self.port!r} après "
                        f"{self._open_retries} tentatives ({exc}). "
                        f"Ferme le Moniteur Série Arduino / un autre script "
                        f"qui tient le COM, puis réessaie."
                    ) from exc

        if self._serial is None:
            raise SerialException(f"Impossible d'ouvrir {self.port!r}: {last_err}")

        # Leonardo reset CDC à l'ouverture — laisser le sketch démarrer
        if self._settle_s > 0:
            time.sleep(self._settle_s)
        try:
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
        except (SerialException, OSError):
            pass

    def move(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        if self._serial is None or not self._serial.is_open:
            return

        payload = f"<{int(dx)},{int(dy)}>\n".encode("ascii")
        try:
            self._serial.write(payload)
        except SerialTimeoutException:
            # Préférer rater une frame plutôt que bloquer l'aim
            return
        except (SerialException, OSError):
            return

    def close(self) -> None:
        if self._serial is not None:
            try:
                if self._serial.is_open:
                    self._serial.close()
            except (SerialException, OSError):
                pass
            self._serial = None


# ---------------------------------------------------------------------------
# Instance globale lazy + API move_mouse
# ---------------------------------------------------------------------------
_arduino: ArduinoMouse | None = None
_suppress_open = False


def get_arduino_mouse() -> ArduinoMouse:
    global _arduino
    if _arduino is None:
        if _suppress_open:
            raise SerialException(
                "Port Arduino fermé (pipeline stop). Relance via MouseController.open()."
            )
        _arduino = ArduinoMouse()
    return _arduino


def close_arduino_mouse() -> None:
    """Ferme le Serial et empêche une réouverture accidentelle (race thread mouse)."""
    global _arduino, _suppress_open
    _suppress_open = True
    if _arduino is not None:
        _arduino.close()
        _arduino = None


atexit.register(close_arduino_mouse)


def move_mouse(dx: int, dy: int) -> None:
    if _suppress_open and _arduino is None:
        return
    get_arduino_mouse().move(dx, dy)


class MouseController:
    """Logique aim (lock / assist) → deltas envoyés à l'Arduino."""

    def __init__(
        self,
        mode: str = AIM_MODE,
        lock_scale: float = LOCK_SCALE,
        debug_moves: bool = AIM_DEBUG_MOVES,
    ):
        if mode not in ("lock", "assist"):
            raise ValueError(f"AIM_MODE invalide : {mode!r} (attendu 'lock' ou 'assist')")
        self.mode = mode
        self.lock_scale = lock_scale
        self.debug_moves = debug_moves

    def open(self) -> None:
        """Autorise et ouvre le Serial (appelé au start pipeline)."""
        global _suppress_open
        _suppress_open = False
        get_arduino_mouse()

    def close(self) -> None:
        close_arduino_mouse()

    def apply(self, dx: float, dy: float, distance: float) -> bool:
        """Applique l'aim. True si un delta non nul a été soumis."""
        move_x, move_y = self.compute_move(dx, dy, distance)
        if move_x == 0 and move_y == 0:
            return False
        move_mouse(move_x, move_y)
        if self.debug_moves:
            print(f"[aim] SNAP dx={dx:.0f} dy={dy:.0f} -> <{move_x},{move_y}>")
        return True

    def compute_move(self, dx: float, dy: float, distance: float) -> tuple[int, int]:
        """Calcule le delta aim sans l'envoyer (pour fusion avec no-recoil)."""
        if self.mode == "lock":
            return self._lock_delta(dx, dy)
        return self._assist_delta(dx, dy, distance)

    def _lock_delta(self, dx: float, dy: float) -> tuple[int, int]:
        move_x = int(round(dx * self.lock_scale))
        move_y = int(round(dy * self.lock_scale))
        if move_x == 0 and move_y == 0:
            return 0, 0
        return move_x, move_y

    def _assist_delta(self, dx: float, dy: float, distance: float) -> tuple[int, int]:
        magnetic_radius = float(SETTINGS.MAGNETIC_RADIUS)
        if magnetic_radius <= 0 or distance > magnetic_radius:
            return 0, 0

        dynamic_smooth = float(SETTINGS.MAX_SMOOTHING) * (
            1 - (distance / magnetic_radius)
        )
        move_x = int(dx * dynamic_smooth)
        move_y = int(dy * dynamic_smooth)
        if move_x == 0 and move_y == 0:
            return 0, 0
        return move_x, move_y


class AimFirePull:
    """Pull-down courbe (peak -> plateau) tant que LMB+RMB sont maintenus."""

    def __init__(
        self,
        dy_per_s: float = AIM_FIRE_PULL_DY_PER_S,
        *,
        peak_dy_per_s: float = AIM_FIRE_PULL_PEAK_DY_PER_S,
        peak_duration_s: float = AIM_FIRE_PULL_PEAK_DURATION_S,
        decay_s: float = AIM_FIRE_PULL_DECAY_S,
        max_dt_s: float = 0.05,
    ):
        self._dy_per_s = float(dy_per_s)
        self._peak_dy_per_s = float(peak_dy_per_s)
        self._peak_duration_s = max(0.0, float(peak_duration_s))
        self._decay_s = max(0.0, float(decay_s))
        self._max_dt_s = max_dt_s
        self._carry = 0.0
        self._elapsed_s = 0.0
        self._last_tick: float | None = None

    @property
    def current_rate(self) -> float:
        """Taux de pull courant (px/s) selon le temps écoulé depuis le début du tir."""
        return self._rate_at(self._elapsed_s)

    def reset(self) -> None:
        self._carry = 0.0
        self._elapsed_s = 0.0
        self._last_tick = None

    def _rate_at(self, elapsed_s: float) -> float:
        peak = self._peak_dy_per_s
        sustain = self._dy_per_s
        if self._peak_duration_s <= 0.0 or peak <= sustain:
            return sustain
        if elapsed_s <= self._peak_duration_s:
            return peak
        if self._decay_s <= 0.0:
            return sustain
        t = elapsed_s - self._peak_duration_s
        if t >= self._decay_s:
            return sustain
        # Interpolation linéaire peak -> plateau
        return peak + (sustain - peak) * (t / self._decay_s)

    def tick(self, now: float) -> int:
        if max(self._dy_per_s, self._peak_dy_per_s) <= 0.0:
            return 0
        if self._last_tick is None:
            self._last_tick = now
            return 0
        dt = min(now - self._last_tick, self._max_dt_s)
        self._last_tick = now
        if dt <= 0.0:
            return 0
        rate = self._rate_at(self._elapsed_s)
        self._elapsed_s += dt
        self._carry += rate * dt
        move_y = int(self._carry)
        self._carry -= move_y
        return move_y


class RecoilCompensator:
    """Jitter aim Apex : tremblement sec sur X et Y pendant LMB+RMB."""

    def __init__(
        self,
        jitter_min: int = NO_RECOIL_JITTER_MIN,
        jitter_max: int = NO_RECOIL_JITTER_MAX,
    ):
        lo = max(1, int(jitter_min))
        hi = max(lo, int(jitter_max))
        self._jitter_min = lo
        self._jitter_max = hi
        self._pending_return = False
        self._dx = 0
        self._dy = 0

    def reset(self) -> None:
        self._pending_return = False
        self._dx = 0
        self._dy = 0

    def tick(self, _now: float = 0.0) -> tuple[int, int]:
        """Tick impair : vecteur aléatoire ±amp ; tick pair : inverse exact (dérive ~0)."""
        if self._pending_return:
            self._pending_return = False
            return -self._dx, -self._dy

        amp_x = random.randint(self._jitter_min, self._jitter_max)
        amp_y = random.randint(self._jitter_min, self._jitter_max)
        self._dx = amp_x if random.random() < 0.5 else -amp_x
        self._dy = amp_y if random.random() < 0.5 else -amp_y
        self._pending_return = True
        return self._dx, self._dy
