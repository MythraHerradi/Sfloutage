# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ultralytics",
# ]
# ///

from ultralytics import YOLO


def main():
    # Met le bon chemin avec le "-2"
    chemin_best_pt = "runs/segment/modele_flouteur-2/weights/best.pt"

    print(f"📦 Chargement du modèle : {chemin_best_pt}")
    model = YOLO(chemin_best_pt)

    print("🔄 Exportation en cours (ONNX)...")
    # L'export le simplifiera pour la vitesse d'exécution
    model.export(format="onnx", simplify=True)

    print("✅ Exportation terminée ! Ton fichier est prêt dans le dossier 'weights'.")


if __name__ == "__main__":
    main()
