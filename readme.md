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
| Configuration | `core/config.py` | Constantes centralisées (FOV, seuils, classes, profils d'entraînement, chemins modèles) |
| Acquisition | `core/capture.py` | Capture du FOV centré via `dxcam`, sortie BGR |
| Inférence | `core/detector.py` | Détection YOLO multiclasse (`ennemi` / `allie`), filtrage par seuil de confiance, rendu debug |
| Ciblage | `core/targeting.py` | Sélection de l'ennemi le plus proche du réticule (ignore les alliés) |
| Souris | `core/mouse.py` | Injection de mouvement via `SendInput` — mode `lock` (snap direct) ou `assist` (friction magnétique) |
| Data mining | `core/collector.py` | Collecte asynchrone d'images utiles à l'entraînement (détections incertaines, faux positifs/négatifs suspects) |
| Orchestration | `core/pipeline.py` | `AimPipeline` : 3 threads découplés (capture / détection / souris) reliés par des queues "dernière valeur uniquement" |

### Flux d'exécution (multithread, Producer-Consumer)

```
Thread capture  → frame_queue (size=1)
Thread detect   → lit frame_queue, détecte, alimente target_queue + debug_queue + DataCollector
Thread mouse    → lit target_queue, applique le mouvement (si AIM_ASSIST=True)
Thread main     → lit debug_queue, affiche la fenêtre OpenCV (si DEBUG=True)
```

Chaque queue est bornée à 1 élément (`put_latest`) : aucun thread n'attend sur une valeur périmée, la latence ne s'accumule jamais.

### Paramètres clés (`core/config.py`)

- **FOV :** 416×416 pixels (multiple de 32, aligné sur `imgsz` d'entraînement)
- **Confiance YOLO :** `CONF_THRESHOLD` (0.65 par défaut)
- **Classes :** `CLASS_NAMES = ("ennemi", "allie")`, cible d'aim = `TARGET_CLASS_ID` (0 = ennemi)
- **Aim :** `AIM_MODE = "lock"` (snap direct, banc de test pipeline) ou `"assist"` (friction magnétique classique)
- **Data mining :** `ENABLE_DATA_MINING`, seuils d'incertitude et cooldowns par raison de capture

### Lancement

```bash
python main.py                        # Pipeline complet (capture + detect + [mouse] + [data mining])
python -m core.capture                # Test isolé de la capture
python -m core.detector               # Test isolé capture + détection

python scripts/extract_frames.py      # Extraction de frames depuis des vidéos (derush/)
python scripts/auto_label.py          # Pré-annotation YOLO des images collectées
python scripts/split_dataset.py       # Split train/val (fusionne toutes les versions listées)
python scripts/train.py               # Entraînement (version par défaut : TRAIN_TARGET_VERSION)
python scripts/train.py --version v3  # Entraînement d'une version spécifique
python scripts/train.py --list        # Liste les profils d'entraînement disponibles
python scripts/export_engine.py       # Export TensorRT (.engine, FP16) de la version entraînée
```

Flags dans `core/config.py` :
- **`DEBUG`** — `True` : fenêtre OpenCV avec annotations (quitte avec `q`) ; `False` : mode production sans rendu, `Ctrl+C` pour quitter.
- **`AIM_ASSIST`** — `True` : le thread souris applique les mouvements ; `False` : détection seule, aucun mouvement de souris.
- **`AIM_ASSIST_REQUIRE_LMB`** — si activé, l'aim n'agit que lorsque le clic gauche est maintenu.
- **`ENABLE_DATA_MINING`** — active la collecte asynchrone d'images (voir section 5).

---

## 4. Entraînement du modèle IA

Le modèle est ré-entraîné de manière itérative, chaque version affinant la précédente (transfer learning).

| Version | Base | Dataset | Notes |
|---|---|---|---|
| V1 | `yolov8n.pt` (COCO) | Dataset communautaire Roboflow | Proof of concept, mono-classe |
| V2 | V1 | `apex.yaml` (dataset custom) | Fine-tuning sur images du jeu |
| V3 | V2 | `apex.yaml` | Modèle de prod historique |
| V4 | V3 | `apex.yaml`, multiclasse (`ennemi`/`allie`) | Version en cours d'entraînement |

Les profils (base, dataset, epochs, batch, chemin de sortie) sont définis dans `TRAIN_PROFILES` (`core/config.py`) — voir `python scripts/train.py --list`.

`scripts/train.py` utilise `exist_ok=True` par défaut : relancer un entraînement sur une version écrase le run existant (`runs/detect/apex_model_vX/`) au lieu de créer `-2`, `-3`, etc., ce qui garantit que les chemins fixes (`V4_MODEL`, ...) dans `config.py` restent à jour. Utiliser `--fresh` pour revenir à l'ancien comportement (nouveau dossier à chaque run).

### Workflow d'entraînement itératif

1. **Collecte manuelle** — `scripts/extract_frames.py` extrait des frames depuis des vidéos (`data/derush/vX/`)
2. **Collecte active (data mining)** — en jeu, `ENABLE_DATA_MINING=True` sauvegarde automatiquement les frames à confiance incertaine ou en cas de tir sans détection (voir section 5), dans `data/auto_collected/`
3. **Annotation** — `scripts/auto_label.py` pré-annote via le meilleur modèle disponible, puis correction manuelle (LabelImg, `classes.txt`)
4. **Split** — `scripts/split_dataset.py` fusionne les versions de `data/images_extraites/` et split 80/20 train/val
5. **Entraînement** — `scripts/train.py --version vX`
6. **Export production** — `scripts/export_engine.py --version vX` (TensorRT FP16)

---

## 5. Data mining (Active Learning)

Objectif : collecter en jeu, de façon asynchrone (sans jamais ralentir la pipeline), les images les plus utiles pour corriger le modèle — pas toutes les frames.

`core/collector.py` (`DataCollector`) écrit sur un thread dédié (`queue.Queue` non bornée + `cv2.imwrite`), déclenché par `AimPipeline._detect_loop` via `consider(frame, detections, clicking=...)`. Chaque raison a son propre cooldown pour éviter de saturer le disque :

| Raison | Condition | But |
|---|---|---|
| `fp_suspect` | Ennemi détecté avec confiance dans la zone incertaine | Confirmer/corriger un potentiel faux positif |
| `ally_fp_suspect` | Allié détecté avec confiance dans la zone incertaine | Idem, côté allié (décor/ennemi mal classé) |
| `enemy_as_ally_suspect` | Tir actif (LMB+RMB), pas d'ennemi confiant, mais un allié détecté | L'ennemi a peut-être été classé "allie" par erreur |
| `fn_suspect` | Tir actif (LMB+RMB), aucun ennemi détecté avec assez de confiance | Faux négatif potentiel (l'IA a manqué la cible) |

---

## 6. Directives de développement

- **Performance first** — éviter les copies mémoire inutiles CPU↔GPU ; queues bornées à 1 élément partout dans le pipeline temps réel
- **Modularité** — chaque étape (capture/detect/mouse/collecte) tourne sur son propre thread, découplée par `queue.Queue`
- **Configuration centralisée** — toute constante partagée vit dans `core/config.py` (rien de codé en dur dans les modules)
- **Développement itératif** — valider chaque module isolément (`python -m core.<module>`) avant l'assemblage dans `main.py`

---

## 7. Roadmap

- **Aim-assist avancé** — Nearest Edge (viser le bord de la bbox le plus proche du réticule plutôt que le centre), easing progressif en mode `assist`
- **Recoil Assist** — lissage dissocié par axe (`smooth_x`/`smooth_y`) pour compenser le recul vertical de l'arme
- **Ciblage stratégique (upper-body)** — offset vertical vers le haut de la bounding box, à terme via annotation dédiée tête/torse
