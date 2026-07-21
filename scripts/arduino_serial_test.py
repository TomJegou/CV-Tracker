"""Envoie des deltas X/Y simulés à l'Arduino Leonardo via USB Serial.

Protocole : "<X,Y>\\n"  (ex. "<12,-34>\\n")
Baud rate : 115200

Usage :
    python scripts/arduino_serial_test.py --list
    python scripts/arduino_serial_test.py --port COM3
    python scripts/arduino_serial_test.py --port COM3 --hz 60 --duration 10
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("pyserial manquant — installe avec : pip install pyserial")
    sys.exit(1)

BAUD_RATE = 115200


def list_serial_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def format_move(dx: int, dy: int) -> bytes:
    return f"<{dx},{dy}>\n".encode("ascii")


def send_move(ser: serial.Serial, dx: int, dy: int) -> None:
    ser.write(format_move(dx, dy))


def random_delta(max_abs: int) -> int:
    value = random.randint(-max_abs, max_abs)
    if value == 0:
        value = random.choice((-1, 1))
    return value


def run_test(
    port: str,
    *,
    hz: float = 30.0,
    duration: float | None = None,
    max_delta: int = 50,
) -> None:
    interval = 1.0 / hz
    deadline = time.perf_counter() + duration if duration is not None else None

    with serial.Serial(port, BAUD_RATE, timeout=0.1) as ser:
        # La Leonardo peut mettre un court instant à réinitialiser le CDC après open()
        time.sleep(2.0)
        ser.reset_input_buffer()

        print(f"Connecté à {port} @ {BAUD_RATE} baud")
        print("Format : <X,Y> — Ctrl+C pour arrêter\n")

        sent = 0
        try:
            while deadline is None or time.perf_counter() < deadline:
                loop_start = time.perf_counter()

                dx = random_delta(max_delta)
                dy = random_delta(max_delta)
                send_move(ser, dx, dy)
                sent += 1

                while ser.in_waiting:
                    line = ser.readline().decode("ascii", errors="replace").strip()
                    if line:
                        print(f"  Arduino ← {line}")

                print(f"  PC → <{dx},{dy}>  ({sent} envois)")
                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0.0, interval - elapsed))
        except KeyboardInterrupt:
            print(f"\nArrêt — {sent} vecteur(s) envoyé(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test série Arduino Leonardo — envoi de deltas <X,Y>.",
    )
    parser.add_argument(
        "--port",
        "-p",
        help="Port série (ex. COM3). Auto-détecté si un seul port disponible.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="Lister les ports série disponibles",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="Fréquence d'envoi en Hz (défaut : 30)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Durée du test en secondes (défaut : infini)",
    )
    parser.add_argument(
        "--max-delta",
        type=int,
        default=50,
        help="Amplitude max des deltas simulés (défaut : 50 px)",
    )
    args = parser.parse_args()

    ports = list_serial_ports()

    if args.list:
        if ports:
            print("Ports série disponibles :")
            for device in ports:
                print(f"  {device}")
        else:
            print("Aucun port série détecté.")
        return

    port = args.port
    if port is None:
        if len(ports) == 1:
            port = ports[0]
            print(f"Port auto-détecté : {port}")
        elif not ports:
            print("Aucun port série trouvé. Branche l'Arduino et relance avec --list.")
            sys.exit(1)
        else:
            print("Plusieurs ports détectés — précise --port :")
            for device in ports:
                print(f"  {device}")
            sys.exit(1)

    run_test(
        port,
        hz=args.hz,
        duration=args.duration,
        max_delta=args.max_delta,
    )


if __name__ == "__main__":
    main()
