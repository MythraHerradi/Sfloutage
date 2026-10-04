# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python",
#     "ultralytics",
#     "onnx",
#     "onnxruntime",
#     "tqdm",
# ]
# ///

import os
import argparse
import cv2
import numpy as np
import subprocess
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def pixelate_image(image, block_size=35):
    h, w = image.shape[:2]
    small = cv2.resize(
        image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def process_video(
    input_path: str, output_path: str, model_path: str, use_mosaic: bool = False
):
    print(f"📦 Chargement du modèle : {model_path}")
    model = YOLO(model_path, task="segment")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ Impossible d'ouvrir la vidéo : {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_video = "temp_sans_son.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    style_txt = "MOSAÏQUE" if use_mosaic else "FLOU GAUSSIEN"
    print(f"\n🎬 Traitement vidéo de : {input_path} (Style : {style_txt})")

    kernel_dilate = np.ones((25, 25), np.uint8)

    with tqdm(
        total=total_frames, desc="Anonymisation", unit="img", colour="green"
    ) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                classes=[0],
                retina_masks=True,
                imgsz=1024,
                conf=0.45,
                verbose=False,
            )

            if use_mosaic:
                anonymized_bg = pixelate_image(frame, block_size=35)
            else:
                anonymized_bg = cv2.GaussianBlur(frame, (121, 121), 0)

            final_frame = frame.copy()

            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                combined_mask = np.zeros((height, width), dtype=np.uint8)

                for mask in masks:
                    mask_resized = cv2.resize(
                        mask, (width, height), interpolation=cv2.INTER_LINEAR
                    )
                    combined_mask = cv2.bitwise_or(
                        combined_mask, (mask_resized > 0.5).astype(np.uint8) * 255
                    )

                combined_mask = cv2.dilate(combined_mask, kernel_dilate, iterations=1)
                combined_mask = cv2.GaussianBlur(combined_mask, (15, 15), 0)

                alpha = (combined_mask / 255.0)[:, :, np.newaxis]
                final_frame = (alpha * anonymized_bg + (1.0 - alpha) * frame).astype(
                    np.uint8
                )

            out.write(final_frame)
            pbar.update(1)

    cap.release()
    out.release()

    print("\n🎵 Ajout de la piste audio originale...")
    try:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            temp_video,
            "-i",
            input_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            output_path,
        ]
        subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        print(f"✅ Vidéo finale avec son terminée : {output_path}\n")
    except FileNotFoundError:
        print("⚠️  'ffmpeg' introuvable. La vidéo n'aura pas de son.")
        os.rename(temp_video, output_path)
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flouter proprement des personnes.")
    parser.add_argument("-i", "--input", required=True, help="Vidéo d'entrée")
    parser.add_argument("-o", "--output", help="Vidéo de sortie")
    parser.add_argument("-m", "--model", default="best.onnx", help="Modèle ONNX")
    parser.add_argument(
        "--mosaic", action="store_true", help="Activer l'effet mosaïque"
    )
    args = parser.parse_args()

    input_file = os.path.expanduser(args.input)
    if args.output:
        output_file = os.path.expanduser(args.output)
    else:
        input_path_obj = Path(input_file)
        suffix = "_mosaique" if args.mosaic else "_floutee"
        output_filename = f"{input_path_obj.stem}{suffix}{input_path_obj.suffix}"
        output_file = str(input_path_obj.parent / output_filename)

    process_video(input_file, output_file, args.model, args.mosaic)
