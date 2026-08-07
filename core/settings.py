"""Réglages runtime modifiables à chaud (fenêtre paramètres + settings.json)."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from core import config

SETTINGS_PATH = config.ROOT_DIR / "settings.json"
_AUTOSAVE_DEBOUNCE_S = 0.4

SettingType = Literal["bool", "float"]


@dataclass(frozen=True)
class SettingSpec:
    name: str
    type: SettingType
    label: str
    group: str
    min: float | None = None
    max: float | None = None
    step: float | None = None


SETTING_SPECS: tuple[SettingSpec, ...] = (
    SettingSpec(
        "OVERLAY",
        "bool",
        "Overlay FOV in-game",
        "Overlay",
    ),
    SettingSpec(
        "OVERLAY_SHOW_CROSSHAIR",
        "bool",
        "Overlay : croix centrale",
        "Overlay",
    ),
    SettingSpec(
        "OVERLAY_SHOW_MAGNETIC_RADIUS",
        "bool",
        "Overlay : cercle magnetique",
        "Overlay",
    ),
    SettingSpec(
        "CONF_THRESHOLD",
        "float",
        "Seuil de confiance (aim)",
        "Détection",
        min=0.05,
        max=0.95,
        step=0.01,
    ),
    SettingSpec(
        "AIM_ASSIST",
        "bool",
        "Aim assist",
        "Aim",
    ),
    SettingSpec(
        "AIM_ASSIST_REQUIRE_LMB",
        "bool",
        "Aim : exiger LMB",
        "Aim",
    ),
    SettingSpec(
        "AIM_ASSIST_REQUIRE_RMB",
        "bool",
        "Aim : exiger RMB",
        "Aim",
    ),
    SettingSpec(
        "AIM_ASSIST_REQUIRE_BOTH",
        "bool",
        "Aim : LMB et RMB (ET)",
        "Aim",
    ),
    SettingSpec(
        "MAX_SMOOTHING",
        "float",
        "Lissage max (assist)",
        "Aim",
        min=0.0,
        max=3.0,
        step=0.05,
    ),
    SettingSpec(
        "MAGNETIC_RADIUS",
        "float",
        "Rayon magnétique (px)",
        "Aim",
        min=10.0,
        max=float(config.FOV_SIZE // 2),
        step=1.0,
    ),
    SettingSpec(
        "AIM_POINT_X",
        "float",
        "Point visé X (0–1)",
        "Aim",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    SettingSpec(
        "AIM_POINT_Y",
        "float",
        "Point visé Y (0–1)",
        "Aim",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    SettingSpec(
        "ACTIVE_JITTER",
        "bool",
        "Jitter (LMB+RMB)",
        "Spray",
    ),
    SettingSpec(
        "ACTIVE_PULL_DOWN",
        "bool",
        "Pull-down (LMB+RMB)",
        "Spray",
    ),
    SettingSpec(
        "ENABLE_DATA_MINING",
        "bool",
        "Data mining FP/FN",
        "Mining",
    ),
)

_SPEC_BY_NAME: dict[str, SettingSpec] = {s.name: s for s in SETTING_SPECS}


def _defaults_from_config() -> dict[str, Any]:
    return {spec.name: getattr(config, spec.name) for spec in SETTING_SPECS}


def _coerce(spec: SettingSpec, value: Any) -> bool | float:
    if spec.type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
        raise ValueError(f"{spec.name}: bool attendu, reçu {value!r}")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{spec.name}: float attendu, reçu {value!r}") from exc
    if spec.min is not None:
        number = max(spec.min, number)
    if spec.max is not None:
        number = min(spec.max, number)
    return number


class RuntimeSettings:
    """Store de réglages partagé (lecture hot-path sans lock)."""

    def __init__(self) -> None:
        self._autosave_lock = threading.Lock()
        self._dirty = False
        self._last_change = 0.0
        self._stop_autosave = threading.Event()
        self._listeners: list[Callable[[str, Any], None]] = []
        self._apply_dict(_defaults_from_config(), mark_dirty=False)
        self._load_json_overlay()
        self._autosave_thread = threading.Thread(
            target=self._autosave_loop,
            name="settings-autosave",
            daemon=True,
        )
        self._autosave_thread.start()

    def _apply_dict(self, values: dict[str, Any], *, mark_dirty: bool) -> None:
        for spec in SETTING_SPECS:
            if spec.name not in values:
                continue
            try:
                coerced = _coerce(spec, values[spec.name])
            except ValueError as exc:
                print(f"[settings] Ignoré {spec.name}: {exc}")
                continue
            object.__setattr__(self, spec.name, coerced)
        if mark_dirty:
            self._mark_dirty()

    def _load_json_overlay(self) -> None:
        path = SETTINGS_PATH
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("racine JSON doit être un objet")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(
                f"[settings] {path.name} invalide -> valeurs config.py ({exc})"
            )
            return
        known = {k: v for k, v in raw.items() if k in _SPEC_BY_NAME}
        unknown = sorted(set(raw) - set(_SPEC_BY_NAME))
        if unknown:
            print(f"[settings] Clés inconnues ignorées : {', '.join(unknown)}")
        self._apply_dict(known, mark_dirty=False)
        print(f"[settings] Chargé {path.name} ({len(known)} clé(s))")

    def get(self, name: str) -> Any:
        if name not in _SPEC_BY_NAME:
            raise KeyError(name)
        return getattr(self, name)

    def set(self, name: str, value: Any) -> Any:
        spec = _SPEC_BY_NAME.get(name)
        if spec is None:
            raise KeyError(name)
        coerced = _coerce(spec, value)
        current = getattr(self, name)
        if current == coerced:
            return coerced
        object.__setattr__(self, name, coerced)
        self._mark_dirty()
        for listener in list(self._listeners):
            try:
                listener(name, coerced)
            except Exception as exc:  # noqa: BLE001 — UI listener ne doit pas casser
                print(f"[settings] Listener error on {name}: {exc}")
        return coerced

    def reset_to_config(self) -> None:
        self._apply_dict(_defaults_from_config(), mark_dirty=True)
        for spec in SETTING_SPECS:
            value = getattr(self, spec.name)
            for listener in list(self._listeners):
                try:
                    listener(spec.name, value)
                except Exception as exc:  # noqa: BLE001
                    print(f"[settings] Listener error on {spec.name}: {exc}")

    def add_listener(self, callback: Callable[[str, Any], None]) -> None:
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str, Any], None]) -> None:
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def as_dict(self) -> dict[str, Any]:
        return {spec.name: getattr(self, spec.name) for spec in SETTING_SPECS}

    def _mark_dirty(self) -> None:
        with self._autosave_lock:
            self._dirty = True
            self._last_change = time.monotonic()

    def _autosave_loop(self) -> None:
        while not self._stop_autosave.wait(0.1):
            with self._autosave_lock:
                if not self._dirty:
                    continue
                if time.monotonic() - self._last_change < _AUTOSAVE_DEBOUNCE_S:
                    continue
                snapshot = self.as_dict()
                self._dirty = False
            self._write_json(snapshot)

    def flush(self) -> None:
        """Écrit immédiatement si dirty (appelé au shutdown)."""
        with self._autosave_lock:
            if not self._dirty:
                return
            snapshot = self.as_dict()
            self._dirty = False
        self._write_json(snapshot)

    def shutdown(self) -> None:
        self.flush()
        self._stop_autosave.set()
        self._autosave_thread.join(timeout=1.0)

    @staticmethod
    def _write_json(snapshot: dict[str, Any]) -> None:
        path = SETTINGS_PATH
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as exc:
            print(f"[settings] Échec écriture {path.name}: {exc}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass


SETTINGS = RuntimeSettings()
