"""Détection / résolution du port série Arduino (Leonardo / clones)."""
from __future__ import annotations

from serial.tools import list_ports

# VID Arduino officiels courants (Leonardo = 0x2341, parfois bootloader 0x2E8A)
_ARDUINO_VIDS = frozenset({0x2341, 0x2E8A})
# Clones / bridges USB-série fréquents (CH340, CP210x) — fallback faible
_CLONE_VIDS = frozenset({0x1A86, 0x10C4})
_NAME_HINTS = ("arduino", "leonardo", "usb serial device")


def list_serial_port_summaries() -> list[str]:
    """Lignes lisibles pour messages d'erreur / --list."""
    lines: list[str] = []
    for port in list_ports.comports():
        vid = f"{port.vid:04X}" if port.vid is not None else "----"
        pid = f"{port.pid:04X}" if port.pid is not None else "----"
        desc = port.description or "?"
        lines.append(f"{port.device}  [{vid}:{pid}]  {desc}")
    return lines


def find_arduino_ports() -> list[str]:
    """Ports candidats, du plus probable au moins probable."""
    strong: list[str] = []
    weak: list[str] = []

    for port in list_ports.comports():
        blob = " ".join(
            filter(
                None,
                (
                    port.description,
                    port.manufacturer,
                    port.product,
                    port.hwid,
                ),
            )
        ).lower()
        name_hit = any(hint in blob for hint in _NAME_HINTS)
        vid = port.vid

        if name_hit or (vid is not None and vid in _ARDUINO_VIDS):
            strong.append(port.device)
        elif vid is not None and vid in _CLONE_VIDS:
            weak.append(port.device)

    # Dédup en gardant l'ordre
    seen: set[str] = set()
    ordered: list[str] = []
    for device in strong + weak:
        if device not in seen:
            seen.add(device)
            ordered.append(device)
    return ordered


def resolve_arduino_port(port: str | None) -> str:
    """Résout le port à ouvrir.

    - ``None``, ``\"\"`` ou ``\"auto\"`` → détection automatique
    - autre valeur → utilisée telle quelle (ex. ``\"COM5\"``)
    """
    if port is not None:
        explicit = str(port).strip()
        if explicit and explicit.lower() != "auto":
            return explicit

    candidates = find_arduino_ports()
    if len(candidates) == 1:
        return candidates[0]

    available = list_serial_port_summaries()
    available_txt = "\n".join(f"  {line}" for line in available) or "  (aucun)"

    if not candidates:
        raise RuntimeError(
            "Aucun port Arduino détecté automatiquement.\n"
            "Branche le Leonardo, ferme le Moniteur Série, ou fixe "
            "ARDUINO_PORT=\"COMx\" dans core/config.py.\n"
            f"Ports visibles :\n{available_txt}"
        )

    raise RuntimeError(
        f"Plusieurs ports Arduino candidats : {', '.join(candidates)}.\n"
        "Fixe ARDUINO_PORT=\"COMx\" dans core/config.py pour trancher.\n"
        f"Ports visibles :\n{available_txt}"
    )
