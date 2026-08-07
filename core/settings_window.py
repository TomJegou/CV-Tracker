"""Fenêtre tkinter de paramètres runtime (thread dédié)."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

from core.settings import SETTING_SPECS, SettingSpec, SETTINGS


def _format_float(spec: SettingSpec, value: float) -> str:
    if spec.step is not None and spec.step >= 1.0:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _is_float_partial(text: str) -> bool:
    """True si `text` est un float valide ou une saisie partielle autorisée."""
    if text == "" or text == "-" or text == "." or text == "-.":
        return True
    try:
        float(text)
        return True
    except ValueError:
        return False


class SettingsWindow:
    """UI paramètres : un thread, un Tk, fermeture = arrêt programme."""

    def __init__(self) -> None:
        self.closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._root: tk.Tk | None = None
        self._vars: dict[str, tk.Variable] = {}
        self._entries: dict[str, ttk.Entry] = {}
        self._updating_ui = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="settings-ui",
            daemon=True,
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        root = tk.Tk()
        self._root = root
        root.title("CV-Tracker — Paramètres")
        root.resizable(False, True)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Validation : n'accepte que des floats (ou saisie partielle).
        vcmd = (root.register(self._validate_float_key), "%P")

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        current_group: str | None = None
        group_frame: ttk.LabelFrame | None = None
        for spec in SETTING_SPECS:
            if spec.group != current_group:
                current_group = spec.group
                group_frame = ttk.LabelFrame(outer, text=current_group, padding=8)
                group_frame.pack(fill=tk.X, pady=(0, 8))
            assert group_frame is not None
            self._build_row(group_frame, spec, vcmd)

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(
            buttons,
            text="Réinitialiser (config.py)",
            command=self._on_reset,
        ).pack(side=tk.LEFT)
        ttk.Label(
            buttons,
            text="Autosave -> settings.json",
            foreground="#666",
        ).pack(side=tk.RIGHT)

        SETTINGS.add_listener(self._on_settings_changed)
        root.mainloop()
        SETTINGS.remove_listener(self._on_settings_changed)
        self._root = None
        self.closed.set()

    def _build_row(
        self,
        parent: ttk.LabelFrame,
        spec: SettingSpec,
        vcmd: tuple,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text=spec.label, width=28, anchor=tk.W).pack(side=tk.LEFT)
        initial = SETTINGS.get(spec.name)

        if spec.type == "bool":
            var = tk.BooleanVar(value=bool(initial))
            self._vars[spec.name] = var
            ttk.Checkbutton(
                row,
                variable=var,
                command=lambda n=spec.name, v=var: self._on_bool(n, v),
            ).pack(side=tk.RIGHT)
            return

        text = _format_float(spec, float(initial))
        var = tk.StringVar(value=text)
        self._vars[spec.name] = var
        entry = ttk.Entry(
            row,
            textvariable=var,
            width=10,
            justify=tk.RIGHT,
            validate="key",
            validatecommand=vcmd,
        )
        entry.pack(side=tk.RIGHT)
        self._entries[spec.name] = entry
        entry.bind(
            "<Return>",
            lambda _e, n=spec.name: self._commit_float(n),
        )
        entry.bind(
            "<FocusOut>",
            lambda _e, n=spec.name: self._commit_float(n),
        )

    @staticmethod
    def _validate_float_key(proposed: str) -> bool:
        return _is_float_partial(proposed)

    def _on_bool(self, name: str, var: tk.BooleanVar) -> None:
        if self._updating_ui:
            return
        SETTINGS.set(name, bool(var.get()))

    def _commit_float(self, name: str) -> None:
        if self._updating_ui:
            return
        var = self._vars.get(name)
        if var is None:
            return
        spec = next(s for s in SETTING_SPECS if s.name == name)
        raw = str(var.get()).strip()
        try:
            value = float(raw)
        except ValueError:
            # Saisie invalide / partielle → revenir à la valeur courante.
            self._set_float_var(name, float(SETTINGS.get(name)))
            return
        coerced = float(SETTINGS.set(name, value))
        self._set_float_var(name, coerced)

    def _set_float_var(self, name: str, value: float) -> None:
        var = self._vars.get(name)
        if var is None:
            return
        spec = next(s for s in SETTING_SPECS if s.name == name)
        text = _format_float(spec, value)
        self._updating_ui = True
        try:
            var.set(text)
        finally:
            self._updating_ui = False

    def _on_reset(self) -> None:
        SETTINGS.reset_to_config()
        self._sync_widgets_from_settings()

    def _on_settings_changed(self, name: str, value: Any) -> None:
        root = self._root
        if root is None:
            return
        # Les listeners peuvent être appelés hors thread Tk -> marshaler.
        try:
            root.after(0, lambda: self._apply_external(name, value))
        except tk.TclError:
            pass

    def _apply_external(self, name: str, value: Any) -> None:
        var = self._vars.get(name)
        if var is None:
            return
        spec = next((s for s in SETTING_SPECS if s.name == name), None)
        if spec is None:
            return
        self._updating_ui = True
        try:
            if spec.type == "bool":
                var.set(bool(value))
            else:
                # Ne pas écraser une saisie en cours dans le champ focusé.
                entry = self._entries.get(name)
                if entry is not None and entry is entry.focus_get():
                    return
                var.set(_format_float(spec, float(value)))
        finally:
            self._updating_ui = False

    def _sync_widgets_from_settings(self) -> None:
        self._updating_ui = True
        try:
            for spec in SETTING_SPECS:
                var = self._vars.get(spec.name)
                if var is None:
                    continue
                value = SETTINGS.get(spec.name)
                if spec.type == "bool":
                    var.set(bool(value))
                else:
                    var.set(_format_float(spec, float(value)))
        finally:
            self._updating_ui = False

    def _on_close(self) -> None:
        root = self._root
        # Detruire les Variables sur le thread Tk avant destroy (evite Tcl_AsyncDelete).
        self._vars.clear()
        self._entries.clear()
        if root is not None:
            try:
                root.quit()
                root.destroy()
            except tk.TclError:
                pass
        self._root = None
        self.closed.set()


def start_settings_window() -> SettingsWindow:
    window = SettingsWindow()
    window.start()
    return window
