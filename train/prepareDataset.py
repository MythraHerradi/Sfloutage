# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python",
#     "numpy",
#     "tqdm",
# ]
# ///

import os
import shutil
import random
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm


def convert_masks_to_yolo(input_dir: str, output_dir: str, val_split: float = 0.2):
    images_dir = Path(input_dir) / "images"
    masks_dir = Path(input_dir) / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        print(f"❌ Dossiers 'images' ou 'masks' introuvables dans {input_dir}")
        print("Vérifie que tu as bien pointé vers le sous-dossier du dataset.")
        return

    # Création des dossiers YOLO
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    mask_files = list(masks_dir.glob("*.png")) + list(masks_dir.glob("*.jpg"))
    if not mask_files:
        print(f"❌ Aucun masque trouvé dans {masks_dir}.")
        return

    print(f"📦 {len(mask_files)} masques trouvés. Conversion en cours...")

    random.seed(42)
    random.shuffle(mask_files)

    split_idx = int(len(mask_files) * (1 - val_split))
    train_files = mask_files[:split_idx]
    val_files = mask_files[split_idx:]

    def process_files(files, split_name):
        for mask_path in tqdm(files, desc=f"Génération {split_name}", colour="green"):
            # Chercher l'image correspondante
            img_path = images_dir / mask_path.name
            if not img_path.exists():
                alt_name = mask_path.name.replace("mask", "image").replace(
                    "Mask", "Image"
                )
                img_path = images_dir / alt_name
                if not img_path.exists():
                    # Tente n'importe quelle image qui a le même nom de base
                    possible = list(images_dir.glob(f"{mask_path.stem}*"))
                    if not possible:
                        continue
                    img_path = possible[0]

            # 1. Lire le masque en niveaux de gris
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            height, width = mask.shape

            # 2. Binariser pour avoir un contour propre
            _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

            # 3. Trouver les contours (polygones) avec OpenCV
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            yolo_polygons = []
            for contour in contours:
                # On ignore les minuscules artefacts (bruit de fond)
                if cv2.contourArea(contour) < 100:
                    continue

                # On simplifie le tracé pour alléger le modèle
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Conversion en coordonnées YOLO (entre 0.0 et 1.0)
                normalized_points = []
                for point in approx:
                    x, y = point[0]
                    nx = max(0.0, min(1.0, x / width))
                    ny = max(0.0, min(1.0, y / height))
                    normalized_points.extend([f"{nx:.6f}", f"{ny:.6f}"])

                if (
                    len(normalized_points) >= 6
                ):  # Minimum 3 points pour faire un polygone
                    yolo_polygons.append("0 " + " ".join(normalized_points))

            if yolo_polygons:
                # 4. Sauvegarder le .txt et l'image
                label_out = (
                    Path(output_dir) / "labels" / split_name / f"{mask_path.stem}.txt"
                )
                with open(label_out, "w") as f:
                    f.write("\n".join(yolo_polygons))

                # Force le même nom entre l'image et le label texte (Requis par YOLO)
                target_img_name = f"{mask_path.stem}{img_path.suffix}"
                shutil.copy(
                    img_path, Path(output_dir) / "images" / split_name / target_img_name
                )

    process_files(train_files, "train")
    process_files(val_files, "val")
    print(f"\n✅ Terminé ! Ton dataset YOLO est prêt dans le dossier : {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convertir dataset de masques MADS en YOLO."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Dossier contenant images/ et masks/"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dataset",
        help="Dossier de sortie (par défaut: dataset)",
    )
    args = parser.parse_args()

    # Gère correctement le symbole "~"
    input_dir = os.path.expanduser(args.input)
    convert_masks_to_yolo(input_dir, args.output)
