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
| Aim (firmware) | Arduino Leonardo → `Mouse.h` (2 modes, voir §3) |
| Annotation | LabelImg (YOLO) |

### Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Adapte l’index CUDA (`cu118` / `cu124` / `cu130`) à ton driver. TensorRT pour `.engine` s’installe à part (kit NVIDIA).

---

## 3. Matériel aim (requis si `AIM_ASSIST`, `NO_RECOIL` ou fire-pull)

Deux setups possibles (même protocole Serial `<dx,dy>\n` @ 115200, même Python).

### A — Deux souris (recommandé si tu veux G HUB)

1. Souris gamer → **port USB du PC**
2. Leonardo → **autre port USB** (HID + COM)
3. Host Shield **non requis**

Flash : [`arduino/serial_aim/serial_aim.ino`](arduino/serial_aim/serial_aim.ino)

Windows fusionne les deux HID sur un seul curseur : tu joues avec ta souris, l’aim pousse via le Leonardo.

### B — Une souris (fusion Host Shield)

1. Leonardo + **USB Host Shield**
2. Souris physique sur le shield (filaire recommandé)
3. Leonardo → PC

Flash : [`arduino/mouse_fusion/mouse_fusion.ino`](arduino/mouse_fusion/mouse_fusion.ino)

> G HUB ne voit plus la souris (elle n’est plus branchée au PC).

### Test rapide

```bash
python scripts/arduino_serial_test.py --list
python scripts/arduino_serial_test.py --port COM5
```

Règle `ARDUINO_PORT` dans `core/config.py`.

---

## 4. Architecture

| Module | Fichier | Rôle |
|---|---|---|
| Config | `core/config.py` | FOV, flags, seuils, COM Arduino, train |
| Chemins dataset | `core/dataset_paths.py` | `data_mining_{NNN}` |
| Chemins modèles | `core/model_paths.py` | `models/apex_{NNN}/` |
| Capture | `core/capture.py` | FOV centré via dxcam (BGR) |
| Inférence | `core/detector.py` | YOLO multiclasse + debug draw |
| Ciblage | `core/targeting.py` | Ennemi le plus proche du réticule |
| Touches | `core/keys.py` | État LMB/RMB (`GetAsyncKeyState`) |
| Souris | `core/mouse.py` | Serial → Leonardo (`lock` / `assist` + jitter + fire-pull) |
| Data mining | `core/collector.py` | FP/FN suspects → `data_mining_{NNN}/` |
| Pipeline | `core/pipeline.py` | Threads capture / detect / mouse |

```
Thread capture  → frame_queue (size=1)
Thread detect   → YOLO + targeting + (mining) + debug_queue
Thread mouse    → MouseController → Arduino <dx,dy>   [si AIM_ASSIST / NO_RECOIL / fire-pull]
Thread main     → fenêtre OpenCV                      [si DEBUG]
```

Protocole Serial (PC → Leonardo) : `<dx,dy>\n` (ex. `<12,-34>\n`), baud `ARDUINO_BAUD` (115200).  
Avec `mouse_fusion`, ces deltas sont fusionnés avec la souris sur le Host Shield ; avec `serial_aim`, le Leonardo n’envoie que l’aim (2ᵉ souris HID).  
`NO_RECOIL` active le **jitter aim** Apex : tremblement sec aller/retour sur X et Y dès LMB+RMB, via Arduino (peut se cumuler avec l’aim).

---

## 5. Configuration (`core/config.py`)

| Flag | Effet |
|---|---|
| `DEBUG` | Fenêtre OpenCV + overlays |
| `AIM_ASSIST` | Ouvre le COM et démarre le thread mouse |
| `AIM_ASSIST_REQUIRE_LMB` | Aim si clic gauche (OU avec RMB si les deux True) |
| `AIM_ASSIST_REQUIRE_RMB` | Aim si clic droit |
| `AIM_ASSIST_REQUIRE_BOTH` | Aim seulement si LMB **et** RMB (prioritaire) |
| `AIM_MODE` | `"lock"` (snap) ou `"assist"` (friction magnétique) |
| `MAGNETIC_RADIUS` | Rayon d’attraction (mode assist) |
| `AIM_POINT_X` / `AIM_POINT_Y` | Point visé dans la box (0–1 ; 0.5 = centre) |
| `AIM_FIRE_PULL_PEAK_DY_PER_S` | Pull initial fort (début de spray, px/s) |
| `AIM_FIRE_PULL_PEAK_DURATION_S` | Durée du peak avant transition |
| `AIM_FIRE_PULL_DECAY_S` | Transition linéaire peak → plateau |
| `AIM_FIRE_PULL_DY_PER_S` | Pull plateau (fin de spray ; 0 = off si peak aussi à 0) |
| `AIM_DEBUG_MOVES` | Log stdout des SNAP |
| `NO_RECOIL` | Jitter aim X/Y sur LMB+RMB (via Arduino) |
| `NO_RECOIL_JITTER_MIN` / `NO_RECOIL_JITTER_MAX` | Amplitude du tremblement (px) |
| `NO_RECOIL_TICK_S` | Période thread mouse fusionné (jitter / fire-pull) |
| `NO_RECOIL_DEBUG` | Logs périodiques LMB/RMB / deltas mouse |
| `ENABLE_DATA_MINING` | Collecte async FP/FN |
| `CONF_THRESHOLD` | Seuil aim / targeting |
| `DATA_MINING_CONF` | Plancher YOLO si mining ON (≤ aim ; une seule passe) |
| `DATA_MINING_UNCERTAIN_*` | Bande FP suspect `[MIN, MAX)` — `MAX` = `CONF_THRESHOLD` |
| `DATA_MINING_FN_MAX_CONF` | FN / confusion si meilleure conf ennemi < ce seuil + LMB+RMB |
| `DATA_MINING_COOLDOWN_FP` / `_FN` | Cooldown anti-spam entre captures (s) |
| `REINFER_LOW_CONF` | Plancher YOLO pour la ré-inférence mining |
| `REINFER_HIGH_CONF` | Seuil auto-accept (toutes boxes ≥ → AUTO) |
| `REINFER_ALWAYS_REVIEW_REASONS` | Raisons forcées en REVIEW (ex. confusion de classe) |
| `REINFER_MANUAL_EDIT_MARGIN_S` | Marge mtime pour détecter une correction LabelImg |
| `CAPTURE_IDLE_SLEEP_S` | Sleep si `grab()` = None (anti busy-loop) |
| `ARDUINO_PORT` | Ex. `"COM5"` — **à adapter** |
| `ARDUINO_SETTLE_S` | Pause après open (reset CDC Leonardo) |
| `ARDUINO_OPEN_RETRIES` | Relances si COM verrouillé (Ctrl+C) |

Defaults runtime = préférences machine : vérifier `ARDUINO_PORT`, et désactiver `AIM_ASSIST` / `DEBUG` si tu lances sans Leonardo.

---

## 6. Lancement

```bash
python main.py                              # Dernier models/apex_* (.engine prioritaire)
python main.py --model path/to/best.pt      # Override

python scripts/auto_label.py                # Pré-anno (skip si .txt déjà présent)
python scripts/auto_label.py --latest
python scripts/auto_label.py --force        # Écrase les .txt mining / existants
python scripts/auto_label.py --latest -f
python scripts/reinfer_mining.py            # Dry-run triage (rapport CSV)
python scripts/reinfer_mining.py --latest --apply
python scripts/reinfer_mining.py --restore --only-manual  # Remonte seulement LabelImg
python scripts/reinfer_mining.py --restore  # Remonte tout review/
python scripts/split_dataset.py             # Fusionne toutes les sessions data_mining_*
python scripts/split_dataset.py --latest    # Uniquement la dernière session data_mining_*
python scripts/split_dataset.py -d chemin/  # Source(s) explicite(s)
python scripts/train.py                     # Crée models/apex_{NNN}/
python scripts/train.py --list
python scripts/export_engine.py             # TensorRT du dernier apex_*
```

Ctrl+C → `pipeline.stop()` ferme le Serial. Sous Windows le COM peut rester verrouillé ~1–3 s ; l’ouverture retente automatiquement.

`AimPipeline` est à usage unique : après `stop()`, recrée une instance via `AimPipeline.create()`. Un crash dans un thread worker (capture / detect / mouse) arrête toute la pipeline, affiche le traceback, et `main.py` sort avec le code 1.

---

## 7. Workflow entraînement

1. **Data mining** — `ENABLE_DATA_MINING=True` → `data/images_extraites/data_mining_{NNN}/`  
   YOLO tourne à `DATA_MINING_CONF` ; l’aim ne voit que `conf ≥ CONF_THRESHOLD`.  
   Chaque image est déjà accompagnée d’un `.txt` YOLO (boxes détectées, ou vide pour `fn_suspect`).
2. **Ré-inférence** — `python scripts/reinfer_mining.py --apply`  
   Repasse le dernier `best.pt` : AUTO (`.txt` réécrit), NEGATIF (hard neg), REVIEW (`→ review/`).  
   Corrige LabelImg sur un échantillon de `data_mining_*/review/`, puis  
   `python scripts/reinfer_mining.py --restore --only-manual`  
   (les non relues partent dans `skipped/`, hors dataset).  
   `--restore` sans `--only-manual` remonte tout `review/`.
3. **Correction** — LabelImg sur la session (ou le résidu `review/`), ou `auto_label.py --force` pour régénérer la pré-anno YOLO
4. **Split** — `python scripts/split_dataset.py` (toutes les sessions ; `--latest` pour la dernière session mining)
5. **Train** — `python scripts/train.py` → `models/apex_{NNN}/` (fine-tune depuis le dernier `best.pt`, sinon `yolov8n.pt`)
6. **Export** — `python scripts/export_engine.py`

### Data mining — raisons

| Raison | Condition |
|---|---|
| `fp_suspect` | Ennemi conf. dans la bande incertaine `[MIN, MAX)` |
| `ally_fp_suspect` | Allié conf. dans la bande incertaine `[MIN, MAX)` |
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
