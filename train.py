from ultralytics import YOLO

from core.config import FOV_SIZE


def main() -> None:
    model = YOLO("yolov8n.pt")

    print("Démarrage de l'entraînement sur la RTX 4070...")

    model.train(
        data="datasets/apex-dataset/data.yaml",
        epochs=100,
        imgsz=FOV_SIZE,
        device=0,
        batch=32,
        workers=4,
        patience=25,
        name="apex_model_v1",
    )

    print(
        "Entraînement terminé ! Le modèle final est disponible dans "
        "runs/detect/apex_model_v1/weights/best.pt"
    )


if __name__ == "__main__":
    main()
