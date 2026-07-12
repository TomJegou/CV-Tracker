# CV-Tracker : Computer Vision Real-Time Tracking & Automation

## 1. Objectif du Projet

Développer un pipeline logiciel haute performance en Python capable d'analyser un flux vidéo en temps réel à l'écran, de détecter des cibles spécifiques grâce à un modèle YOLO, et de simuler des mouvements de souris fluides pour suivre ces cibles.

Le système vise une latence quasi nulle, un rafraîchissement supérieur à 100 FPS, et tire parti de l'accélération matérielle (NVIDIA RTX 4070).

---

## 2. Stack Technique

| Composant | Technologie |
|---|---|
| Langage | Python 3.10+ |
| Hardware cible | CPU multicœur, GPU NVIDIA RTX 4070 |
| Capture d'écran | `dxcam` (API DXGI Desktop Duplication) |
| Inférence IA | `ultralytics` (YOLOv8) |
| Traitement matriciel | `numpy`, `opencv-python` |
| Simulation d'input | API Windows via `ctypes` (`user32.mouse_event`) |
| Annotation | CVAT ou Roboflow (export YOLO) |

---

## 3. Phase 1 : Pipeline Temps Réel

Cette phase construit l'architecture logicielle de base et la valide en conditions réelles avant d'investir dans un modèle personnalisé perfectionné.

Le pipeline est découpé en modules indépendants (`/core/`), orchestrés par `main.py` :

| Module | Fichier | Rôle |
|---|---|---|
| Configuration | `config.py` | Constante partagée `FOV_SIZE` (416 px) |
| Acquisition | `capture.py` | Capture du FOV centré via `dxcam`, sortie BGR |
| Inférence | `detector.py` | Détection YOLO, filtrage par seuil de confiance |
| Ciblage | `targeting.py` | Sélection de la cible la plus proche du réticule |
| Souris | `mouse.py` | Assistance magnétique avec friction dynamique |

### Flux d'exécution

```
Capture → Détection → Ciblage → Souris
```

Chaque module expose une API non bloquante (`get_latest_frame`, `detect`, `get_best_target`, `move`) afin de préparer une architecture *Producer-Consumer* multithreadée.

### Paramètres clés

- **FOV :** 416×416 pixels (multiple de 32, aligné sur `imgsz` d'entraînement), défini dans `core/config.py`
- **Confiance YOLO :** 60 % par défaut
- **Assistance souris :** nulle au-delà de 50 px du centre, progressive à l'approche de la cible

### Lancement

```bash
python main.py          # Pipeline complet
python -m core.capture  # Test isolé de la capture
python -m core.detector # Test isolé capture + détection
```

Le flag `DEBUG` dans `core/config.py` contrôle le rendu visuel :
- `DEBUG = True` — fenêtre OpenCV avec annotations, quitte avec `q`
- `DEBUG = False` — mode production sans `cv2.imshow`, quitte avec `Ctrl+C`

---

## 4. Phase 2 : Entraînement du Modèle IA

Une fois la Phase 1 validée (performances et mouvements fluides), la Phase 2 remplace le modèle générique COCO par un modèle spécialisé sur *Apex Legends*.

### Stratégie en deux temps

#### Modèle V1 — Proof of Concept (en cours)

Plutôt que d'annoter manuellement des centaines d'images dès le départ, on s'appuie sur un **dataset communautaire Roboflow** (846 images déjà annotées, classe `Player-Models`).

- **Objectif :** entraîner rapidement une version V1 sur la RTX 4070
- **Avantage :** tester immédiatement la logique de ciblage en conditions réelles, sans attendre un dataset parfait

L'entraînement est lancé via `train.py` :

```bash
python train.py
```

| Paramètre | Valeur | Justification |
|---|---|---|
| `imgsz` | `FOV_SIZE` (416) | Cohérence entraînement / inférence |
| `batch` | 32 | Optimisé pour la VRAM de la 4070 |
| `patience` | 25 | Early stopping anti-overfitting |
| Sortie | `runs/detect/apex_model_v1/weights/best.pt` | Modèle à intégrer dans `detector.py` |

#### Modèle V2 — Auto-Labeling (vision long terme)

Le modèle V1 sert de fondation pour construire un dataset sur-mesure sans annotation manuelle exhaustive :

1. **Acquisition ciblée** — enregistrement de sessions en FOV 416×416
2. **Pré-annotation IA** — le modèle V1 génère les bounding boxes automatiquement
3. **Supervision humaine** — correction rapide des annotations (quelques secondes par image)
4. **Ré-entraînement** — fine-tuning sur le dataset combiné pour un modèle V2 ultra-spécialisé

### Workflow général d'entraînement

1. **Collecte** — 500 à 2000 images avec variance maximale (angles, éclairage, distances)
2. **Annotation** — bounding boxes exportées au format YOLO (`train/`, `val/`, labels `.txt`)
3. **Transfer learning** — `yolov8n.pt` comme base, entraînement GPU FP16
4. **Optimisation** — export TensorRT (`best.engine`) pour maximiser l'inférence en production

---

## 5. Directives de Développement

- **Performance first** — éviter les copies mémoire inutiles CPU↔GPU ; privilégier `numpy` aux boucles Python natives
- **Modularité** — découpler chaque étape via le pattern *Producer-Consumer* (`queue.Queue` ou `multiprocessing`) pour que l'inférence ne bloque pas la capture
- **Développement itératif** — valider chaque module isolément (`python -m core.<module>`) avant l'assemblage dans `main.py`
- **Configuration centralisée** — toute constante partagée (FOV, seuils, chemins) vit dans `core/config.py`

---

## 6. Roadmap

Fonctionnalités prévues pour passer d'un prototype fonctionnel à un moteur d'assistance abouti.

### Mode Production vs Debug

Implémenté via `DEBUG` dans `core/config.py`. En production (`DEBUG = False`), le rendu OpenCV est désactivé pour maximiser les performances. Les évolutions ci-dessous concernent les prochaines itérations.

### Activation contextuelle (Trigger & ADS)

L'assistance ne s'activera que lors d'un engagement réel. Vérification d'état via `win32api.GetAsyncKeyState` : le module souris n'appliquera la friction que si le joueur maintient le clic gauche (tir) ou le clic droit (visée / ADS).

### Aim-assist avancé

- **Nearest Edge** — viser le pixel de la cible le plus proche du réticule plutôt que le centre de la bounding box, pour un mouvement plus naturel
- **Easing** — courbe d'interpolation inverse : la force magnétique diminue à l'approche de la cible, évitant le jitter et redonnant le contrôle fin au joueur

### Recoil Assist

Dissociation des axes de lissage : `smooth_x` (horizontal, léger) et `smooth_y` (vertical, plus prononcé). Si la cible est sous le réticule (recul de l'arme), compensation verticale renforcée pour stabiliser le spray.

### Ciblage stratégique (Upper-Body)

- **Court terme** — offset mathématique sur l'axe Y vers les 20 % supérieurs de la bounding box
- **Long terme (V2)** — annotations centrées sur la tête et le haut du torse lors de l'auto-labeling, pour que l'IA ignore nativement les membres inférieurs
