# CV-Tracker : Computer Vision Real-Time Tracking & Automation

## 1. Objectif du Projet
Développer un pipeline logiciel haute performance en Python capable d'analyser un flux vidéo en temps réel à l'écran, de détecter des cibles spécifiques grâce à un modèle d'Intelligence Artificielle (YOLO), et de simuler des mouvements de souris fluides pour suivre ces cibles. 

Le système doit fonctionner avec une latence quasi nulle, cibler un rafraîchissement supérieur à 100 FPS, et tirer pleinement parti de l'accélération matérielle (NVIDIA RTX 4070).

---

## 2. Stack Technique Globale
*   **Langage :** Python 3.10+ (Choisi pour son écosystème IA natif et sa modularité).
*   **Hardware Cible :** Processeur multicœur, GPU NVIDIA RTX 4070.
*   **Capture d'écran :** `dxcam` (API DXGI Desktop Duplication pour bypasser le bus mémoire CPU).
*   **Inférence IA :** `ultralytics` (YOLOv8 / YOLOv11).
*   **Traitement matriciel :** `numpy`, `opencv-python` (cv2).
*   **Simulation d'Input (Software) :** API Windows natives via `ctypes` (`user32.SendInput` ou `mouse_event`).
*   **Outil d'Annotation :** CVAT (Computer Vision Annotation Tool - Local/Open Source).

---

## 3. Phase 1 : Pipeline Temps Réel (La Tuyauterie)
Cette phase consiste à construire l'architecture logicielle de base et à la valider avec un modèle YOLO générique (pré-entraîné sur COCO, ex: détection de personnes) pour s'assurer que le pipeline tient les 100+ FPS avant d'investir du temps dans l'entraînement personnalisé.

Le pipeline repose sur une boucle asynchrone ou multithreadée, divisée en 4 modules distincts (`/core/`) :

### Module 1 : Acquisition (Screen Grabber) - `capture.py`
*   Ne capture pas tout l'écran. Définit une zone d'intérêt (ROI / FOV) stricte au centre de l'écran (ex: 400x400 pixels).
*   Utilise `dxcam` pour récupérer l'image directement depuis la VRAM avec une latence < 2ms.
*   Convertit le buffer en tableau `numpy` exploitable par OpenCV/YOLO.

### Module 2 : Inférence (Detector) - `detector.py`
*   Charge le modèle YOLO optimisé.
*   Reçoit l'image du Module 1 et retourne les *bounding boxes* (coordonnées relatives au FOV) et le score de confiance.
*   Filtre les faux positifs (seuil de confiance paramétrable, ex: > 60%).

### Module 3 : Logique Spatiale (Targeting) - `maths.py` ou intégré
*   Convertit les coordonnées relatives de l'IA en coordonnées absolues (ou en vecteurs directionnels) par rapport au centre absolu de l'écran.
*   Applique une logique de priorité si plusieurs cibles sont détectées (ex: cibler l'entité la plus proche du centre du réticule).

### Module 4 : Contrôleur Souris (Mouse Controller) - `mouse.py`
*   Reçoit le vecteur cible (Distance X, Distance Y).
*   Applique un algorithme de lissage (Contrôleur PID ou interpolation de Bézier) pour générer une courbe de mouvement fluide et "humaine".
*   Envoie les instructions de déplacement via l'API Windows.

---

## 4. Phase 2 : Entraînement du Modèle IA (Custom Dataset)
Une fois la Phase 1 validée (performances et mouvements fluides), la Phase 2 consiste à remplacer le modèle générique par un modèle ultra-spécialisé.

### Étape 1 : Collecte des Données
*   Extraire des images brutes in-game (500 à 2000 images).
*   Maximiser la variance (angles, éclairage, distances, arrière-plans).

### Étape 2 : Annotation (Ground Truth) avec CVAT
*   Importer les images brutes dans une instance locale de **CVAT**.
*   Dessiner les boîtes englobantes (*bounding boxes*) autour des cibles spécifiques (ex: `Head`, `Body`).
*   Exporter le dataset final au format **YOLOv8 PyTorch** (dossiers divisés en `train`, `val` avec fichiers texte `.txt` normalisés).

### Étape 3 : Entraînement (Transfer Learning)
*   Charger un modèle pré-entraîné léger (`yolov8n.pt`) via l'API `ultralytics`.
*   Lancer l'entraînement sur les cœurs Tensor de la RTX 4070.

### Étape 4 : Optimisation (TensorRT)
*   Compiler les poids optimaux (`best.pt`) au format **NVIDIA TensorRT** (`best.engine`) via l'exportation en précision FP16 (`half=True`).
*   Intégrer le fichier `.engine` dans le module `detector.py` de la Phase 1.

---

## 5. Directives pour le Développement (Cursor AI)
*   **Performance First :** Éviter à tout prix les copies de mémoire inutiles entre le CPU et le GPU. Utiliser des opérations vectorisées (`numpy`) plutôt que des boucles `for` en Python natif.
*   **Modularité :** Chaque étape du pipeline doit être découplée (utiliser le pattern *Producer-Consumer* avec `queue.Queue` ou `multiprocessing`) pour éviter que le temps d'inférence ne bloque la boucle de capture.
*   **Développement Itératif :** Valider chaque module de la Phase 1 de manière isolée avant de les assembler.

## 🚀 Évolutions Futures Prévues (Roadmap)

Afin d'affiner le comportement du système et de le rendre le plus humain et transparent possible pour le joueur, les fonctionnalités suivantes seront implémentées dans les itérations futures :

### 1. Trigger & ADS Activation (Assistance Contextuelle)
*   **Objectif :** L'assistance ne doit s'activer que lors d'un engagement réel, laissant le joueur totalement libre de ses mouvements lors de la navigation, du loot ou de l'observation.
*   **Implémentation technique :** Intégration d'une vérification d'état (Hardware Polling) via l'API Windows (`win32api.GetAsyncKeyState`). Le module de mouvement n'appliquera les calculs de "Friction" que si le joueur maintient le clic gauche (Tir) ou le clic droit (Visée / ADS). 

### 2. Recoil Assist (Compensation du Recul)
*   **Objectif :** Aider le joueur à contrôler le recul vertical (recoil pattern) des armes sans verrouiller le viseur de manière artificielle.
*   **Implémentation technique :** Dissociation des multiplicateurs de lissage sur les axes. La formule de calcul remplacera le paramètre global `max_smoothing` par un `smooth_x` (mouvement horizontal léger) et un `smooth_y` (mouvement vertical plus prononcé). Si la cible détectée est située sous le réticule (car l'arme monte avec le recul), le système appliquera une force de compensation légèrement supérieure vers le bas pour stabiliser le *spray*.