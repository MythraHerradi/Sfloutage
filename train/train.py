# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ultralytics",
#     "pyyaml",
# ]
# ///

import os

import yaml
from ultralytics import YOLO


def create_yaml_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "dataset")
    yaml_path = os.path.join(current_dir, "dataset.yaml")

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"❌ Le dossier de données n'existe pas : {dataset_dir}\nAs-tu bien lancé 'prepareDataset.py' ?"
        )

    yaml_content = {
        "path": dataset_dir,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"📄 Fichier '{yaml_path}' généré avec succès.")
    return yaml_path


def main():
    print("🚀 Démarrage de l'entraînement HAUTE DÉFINITION...")
    yaml_file = create_yaml_file()

    # 1. On utilise le modèle MEDIUM (beaucoup plus précis sur les contours)
    model = YOLO("yolo26s-seg.pt")

    # 2. Entraînement en 1024x1024 pour des masques parfaits
    results = model.train(
        data=yaml_file,
        epochs=1000,
        imgsz=1024,  # 🔥 La clé pour des détouages de haute qualité
        batch=8,  # 🔥 Baissé à 4 pour ne pas saturer ta RTX 5070 (8Go VRAM) avec du 1024p
        device=0,
        name="modele_mads_hq",
        patience=100,
        verbose=True,
    )

    print("\n✅ Entraînement terminé !")

    best_model_path = "runs/segment/modele_mads_hq/weights/best.pt"
    print(f"📦 Exportation du meilleur modèle ({best_model_path}) en ONNX 1024p...")

    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        # On force l'export ONNX à garder cette haute résolution
        best_model.export(format="onnx", simplify=True, imgsz=1024)
        print(
            "\n🎉 Tout est prêt ! Ton modèle HAUTE QUALITÉ en .onnx est dans runs/segment/modele_mads_hq/weights/"
        )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
