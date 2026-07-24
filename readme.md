# CV-Tracker — Computer Vision Real-Time Tracking

Pipeline Python basse latence : capture FOV écran → YOLO multiclasse → ciblage → injection souris via **Arduino Leonardo** (HID hardware, pas `SendInput`).

---

## 1. Objectif

Analyser le centre de l’écran en temps réel, détecter des cibles (`ennemi` / `allie`), et optionnellement appliquer des deltas de souris fusionnés avec la souris physique sur un Leonardo + USB Host Shield.

Cible perf : queues bornées, GPU NVIDIA (YOLO / TensorRT FP16).

---

## 2. Stack

| Composant | Technologie |
|---|---|
| Langage | Python 3.10+ |
| Capture | `dxcam` (DXGI Desktop Duplication) |
| Inférence | Ultralytics YOLO → export TensorRT (`.engine`) |
| Aim (PC) | `pyserial` → protocole `<dx,dy>\n` @ 115200 |
| Aim (firmware) | Arduino Leonardo + USB Host Shield → `Mouse.h` |
| Annotation | LabelImg (YOLO) |

---

## 3. Matériel aim (requis si `AIM_ASSIST=True`)

1. Arduino **Leonardo** (ATmega32U4, HID natif)
2. **USB Host Shield** (compatible)
3. Souris physique branchée sur le shield (Superlight **filaire** recommandé)
4. Leonardo branché en USB au PC (port COM Windows)

Flash **uniquement** : [`arduino/mouse_fusion/mouse_fusion.ino`](arduino/mouse_fusion/mouse_fusion.ino)

> `arduino/serial_mouse/` est un prototype obsolète (parse le Serial mais n’appelle pas `Mouse.move`).

Test rapide sans pipeline :

```bash
python scripts/arduino_serial_test.py --list
python scripts/arduino_serial_test.py --port COM5
```

---

## 4. Architecture

| Module | Fichier | Rôle |
|---|---|---|
| Config | `core/config.py` | FOV, flags, seuils, COM Arduino, train |
| Chemins dataset | `core/dataset_paths.py` | `v*` / `data_mining_{NNN}` |
| Chemins modèles | `core/model_paths.py` | `models/apex_{NNN}/` |
| Capture | `core/capture.py` | FOV centré via dxcam (BGR) |
| Inférence | `core/detector.py` | YOLO multiclasse + debug draw |
| Ciblage | `core/targeting.py` | Ennemi le plus proche du réticule |
| Touches | `core/keys.py` | État LMB/RMB (`GetAsyncKeyState`) |
| Souris | `core/mouse.py` | Serial → Leonardo (`lock` / `assist`) |
| Data mining | `core/collector.py` | FP/FN suspects → `data_mining_{NNN}/` |
| Pipeline | `core/pipeline.py` | Threads capture / detect / mouse |

```
Thread capture  → frame_queue (size=1)
Thread detect   → YOLO + targeting + (mining) + debug_queue
Thread mouse    → MouseController → Arduino <dx,dy>   [si AIM_ASSIST]
Thread main     → fenêtre OpenCV                      [si DEBUG]
```

Protocole Serial (PC → Leonardo) : `<dx,dy>\n` (ex. `<12,-34>\n`), baud `ARDUINO_BAUD` (115200).  
Le sketch fusionne ces deltas avec le rapport HID de la souris physique.

---

## 5. Configuration (`core/config.py`)

| Flag | Effet |
|---|---|
| `DEBUG` | Fenêtre OpenCV + overlays |
| `AIM_ASSIST` | Ouvre le COM et démarre le thread mouse |
| `AIM_ASSIST_REQUIRE_LMB` | Aim seulement si clic gauche maintenu |
| `AIM_MODE` | `"lock"` (snap) ou `"assist"` (friction magnétique) |
| `AIM_DEBUG_MOVES` | Log stdout des SNAP |
| `ENABLE_DATA_MINING` | Collecte async FP/FN |
| `ARDUINO_PORT` | Ex. `"COM5"` — **à adapter** |
| `ARDUINO_SETTLE_S` | Pause après open (reset CDC Leonardo) |
| `ARDUINO_OPEN_RETRIES` | Relances si COM verrouillé (Ctrl+C) |

Defaults runtime = préférences machine : vérifier `ARDUINO_PORT`, et désactiver `AIM_ASSIST` / `DEBUG` si tu lances sans Leonardo.

---

## 6. Lancement

```bash
python main.py                              # Dernier models/apex_* (.engine prioritaire)
python main.py --model path/to/best.pt      # Override

python scripts/extract_frames.py            # Derush → prochain images_extraites/vN/
python scripts/auto_label.py                # Pré-anno (skip si .txt déjà présent)
python scripts/auto_label.py --latest
python scripts/split_dataset.py             # Fusionne v* + data_mining_* → train/val
python scripts/train.py                     # Crée models/apex_{NNN}/
python scripts/train.py --list
python scripts/export_engine.py             # TensorRT du dernier apex_*
```

Ctrl+C → `pipeline.stop()` ferme le Serial. Sous Windows le COM peut rester verrouillé ~1–3 s ; l’ouverture retente automatiquement.

---

## 7. Workflow entraînement

1. **Data mining** — `ENABLE_DATA_MINING=True` → `data/images_extraites/data_mining_{NNN}/`  
   Chaque image est déjà accompagnée d’un `.txt` YOLO (boxes détectées, ou vide pour `fn_suspect`).
2. **Correction** — LabelImg sur la session (ou supprimer les `.txt` puis `auto_label.py` pour régénérer une pré-anno)
3. **Split** — `python scripts/split_dataset.py`
4. **Train** — `python scripts/train.py` → `models/apex_{NNN}/` (fine-tune depuis le dernier `best.pt`, sinon `yolov8n.pt`)
5. **Export** — `python scripts/export_engine.py`

### Data mining — raisons

| Raison | Condition |
|---|---|
| `fp_suspect` | Ennemi conf. dans la bande incertaine |
| `ally_fp_suspect` | Allié conf. dans la bande incertaine |
| `enemy_as_ally_suspect` | Tir (LMB+RMB), pas d’ennemi confiant, allié détecté |
| `fn_suspect` | Tir (LMB+RMB), conf. ennemi trop basse |

---

## 8. Dépannage aim

| Symptôme | Piste |
|---|---|
| `PermissionError` / accès refusé sur COMx | Attendre 2–3 s après Ctrl+C ; fermer le Moniteur Série Arduino |
| Souris morte / pas de fusion | Sketch `mouse_fusion` flashé ? Host Shield OK ? Souris sur le shield ? |
| Python envoie des SNAP mais rien en jeu | Mauvais `ARDUINO_PORT` ; tester avec `arduino_serial_test.py` |
| COM change après reset Leonardo | Vérifier le Gestionnaire de périphériques → mettre à jour `ARDUINO_PORT` |

---

## 9. Directives

- Performance first — queues size=1, pas d’I/O bloquant dans la boucle detect
- Config = flags / seuils ; chemins versionnés = découverte dynamique
- Aim = hardware HID uniquement (plus de `SendInput`)
- Valider capture / détecteur isolément (`python -m core.capture`, `python -m core.detector`) avant le pipeline complet
