# CV-Tracker : Computer Vision Real-Time Tracking & Automation

## 1. Objectif du Projet

Pipeline logiciel haute performance en Python capable d'analyser un flux vidéo en temps réel à l'écran, de détecter des cibles (ennemis / alliés) grâce à un modèle YOLO multiclasse, et de simuler des mouvements de souris pour suivre ces cibles.

Le système vise une latence minimale et tire parti de l'accélération matérielle (NVIDIA RTX 4070, TensorRT).

---

## 2. Stack Technique

| Composant | Technologie |
|---|---|
| Langage | Python 3.10+ |
| Hardware cible | CPU multicœur, GPU NVIDIA RTX 4070 |
| Capture d'écran | `dxcam` (API DXGI Desktop Duplication) |
| Inférence IA | `ultralytics` (YOLO), export TensorRT (`.engine`, FP16) |
| Traitement matriciel | `numpy`, `opencv-python` |
| Simulation d'input | API Windows via `ctypes` (`user32.SendInput`) |
| Annotation | LabelImg (format YOLO) |

---

## 3. Architecture du pipeline

Le projet est découpé en modules indépendants (`core/`), orchestrés par `main.py` :

| Module | Fichier | Rôle |
|---|---|---|
| Configuration | `core/config.py` | Constantes runtime (FOV, seuils, classes, hyperparams train) |
| Chemins dataset | `core/dataset_paths.py` | Découverte dynamique `v*` / `data_mining_{NNN}` |
| Chemins modèles | `core/model_paths.py` | Découverte dynamique `models/apex_{NNN}/` |
| Acquisition | `core/capture.py` | Capture du FOV centré via `dxcam`, sortie BGR |
| Inférence | `core/detector.py` | Détection YOLO multiclasse (`ennemi` / `allie`), rendu debug |
| Ciblage | `core/targeting.py` | Sélection de l'ennemi le plus proche du réticule |
| Souris | `core/mouse.py` | Injection via `SendInput` — mode `lock` ou `assist` |
| Data mining | `core/collector.py` | Collecte asynchrone FP/FN suspects → `data_mining_{NNN}/` |
| Orchestration | `core/pipeline.py` | 3 threads (capture / detect / mouse) + queues size=1 |

### Flux d'exécution

```
Thread capture  → frame_queue (size=1)
Thread detect   → détection + target_queue + debug_queue + DataCollector
Thread mouse    → mouvement (si AIM_ASSIST=True)
Thread main     → fenêtre OpenCV (si DEBUG=True)
```

### Lancement

```bash
python main.py                              # Dernier models/apex_* (.engine prioritaire)
python main.py --model path/to/best.pt      # Override modèle

python scripts/extract_frames.py            # Derush → prochain images_extraites/vN/
python scripts/auto_label.py                # Pré-anno des data_mining_*
python scripts/auto_label.py --latest       # Dernière session seulement
python scripts/auto_label.py -d chemin/     # Dossier explicite
python scripts/split_dataset.py             # Fusionne v* + data_mining_* → train/val
python scripts/split_dataset.py -d chemin/  # Sources explicites
python scripts/train.py                     # Crée models/apex_{NNN}/ (base = dernier best.pt)
python scripts/train.py --list
python scripts/train.py --base path/best.pt
python scripts/export_engine.py             # Export TensorRT du dernier apex_*
python scripts/export_engine.py -m models/apex_004
```

Flags dans `core/config.py` : `DEBUG`, `AIM_ASSIST`, `AIM_ASSIST_REQUIRE_LMB`, `ENABLE_DATA_MINING`, `AIM_MODE`.

---

## 4. Entraînement

Les poids vivent sous `models/apex_{NNN}/weights/best.pt` (+ `best.engine` après export).  
Chaque `python scripts/train.py` crée le prochain index et fine-tune depuis le dernier `best.pt` disponible (sinon `yolov8n.pt`).

### Workflow

1. **Data mining** — pipeline avec `ENABLE_DATA_MINING=True` → `data/images_extraites/data_mining_{NNN}/`
2. **Auto-label** — `python scripts/auto_label.py` puis correction LabelImg
3. **Split** — `python scripts/split_dataset.py`
4. **Train** — `python scripts/train.py` → `models/apex_{NNN}/`
5. **Export** — `python scripts/export_engine.py`

---

## 5. Data mining (Active Learning)

`DataCollector` écrit en async dans une nouvelle session `data_mining_{NNN}/` à chaque lancement de pipeline.

| Raison | Condition |
|---|---|
| `fp_suspect` | Ennemi conf. dans la zone incertaine |
| `ally_fp_suspect` | Allié conf. dans la zone incertaine |
| `enemy_as_ally_suspect` | Tir (LMB+RMB), pas d'ennemi confiant, allié détecté |
| `fn_suspect` | Tir (LMB+RMB), ennemi conf. trop basse |

---

## 6. Directives

- Performance first — queues bornées à 1, pas d'I/O dans la boucle detect
- Config = seuils / flags ; chemins versionnés = découverte dynamique
- Valider chaque module isolément (`python -m core.<module>`) avant assemblage
