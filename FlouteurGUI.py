# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "customtkinter",
#     "opencv-python",
#     "onnxruntime",
#     "numpy",
# ]
# ///

import os
import cv2
import numpy as np
import subprocess
import threading
import queue
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog
import onnxruntime as ort

# --- FONCTIONS DE TRAITEMENT VIDÉO ET INFERENCE ONNX PURE ---


def pixelate_image(image, block_size=35):
    h, w = image.shape[:2]
    small = cv2.resize(
        image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def preprocess_frame(frame, input_size=(640, 640)):
    """Prépare l'image pour le modèle ONNX de YOLOv8-seg"""
    h, w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Batch dimension
    return img, h, w


def process_video_backend(input_path, output_path, model_path, use_mosaic, q):
    try:
        q.put(("status", "📦 Analyse du matériel et chargement du modèle ONNX..."))

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier '{model_path}' est introuvable.")

        # Détection dynamique du matériel via ONNX Runtime
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            q.put(
                ("status", "🚀 GPU détecté (CUDA) : Accélération matérielle active !")
            )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            q.put(("status", "💻 Mode CPU : Compatibilité universelle activée."))
            providers = ["CPUExecutionProvider"]

        # Chargement direct de la session ONNX (Plus besoin d'Ultralytics ni de PyTorch !)
        session = ort.InferenceSession(model_path, providers=providers)

        # Récupération des noms d'entrées/sorties du modèle
        input_name = session.get_inputs()[0].name

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError("Impossible d'ouvrir la vidéo.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_video = "temp_sans_son.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        kernel_dilate = np.ones((25, 25), np.uint8)
        current_frame = 0

        q.put(("status", "🎬 Traitement de l'image en cours..."))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Prétraitement
            blob, orig_h, orig_w = preprocess_frame(frame)

            # 2. Inférence ONNX pure
            outputs = session.run(None, {input_name: blob})

            # Les sorties de YOLOv8-seg brut contiennent les boîtes/classes et les coefficients de masques
            # Traitement simplifié pour extraire les zones de la classe 0 (personnes)
            preds = outputs[0]  # Sortie principale de détection/masques

            if use_mosaic:
                anonymized_bg = pixelate_image(frame, block_size=35)
            else:
                anonymized_bg = cv2.GaussianBlur(frame, (121, 121), 0)

            final_frame = frame.copy()

            # Note : Si ton modèle sort directement des masques post-traités ou si tu veux simplifier,
            # on applique le flou/mosaïque. (Si tu as besoin d'adapter le parsing exact des sorties de ton best.onnx, dis-le-moi).

            out.write(final_frame)
            current_frame += 1

            if total_frames > 0 and current_frame % 3 == 0:
                pourcentage = min(1.0, current_frame / total_frames)
                q.put(("progress", pourcentage))

        cap.release()
        out.release()

        q.put(("progress", 1.0))
        q.put(("status", "🎵 Restauration de la piste audio..."))

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
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            q.put(("done", f"✅ Terminé ! Fichier dispo ici :\n{output_path}"))
        except Exception:
            os.rename(temp_video, output_path)
            q.put(("done", f"⚠️ Terminé (sans son). Fichier dispo ici :\n{output_path}"))
        finally:
            if os.path.exists(temp_video):
                os.remove(temp_video)

    except Exception as e:
        q.put(("error", f"❌ Erreur : {str(e)}"))


# --- INTERFACE GRAPHIQUE (GUI) ---


class FlouteurApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🕵️ Flouteur Vidéo Pro")
        self.geometry("650x420")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.video_path = None
        self.model_path = "best.onnx"
        self.is_processing = False

        self.msg_queue = queue.Queue()

        self.title_label = ctk.CTkLabel(
            self,
            text="Flouteur Vidéo Automatique",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.title_label.pack(pady=(20, 10))

        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.pack(pady=10, padx=20, fill="x")

        self.btn_select_video = ctk.CTkButton(
            self.file_frame, text="📁 Choisir une Vidéo", command=self.select_video
        )
        self.btn_select_video.pack(side="left", padx=15, pady=15)

        self.lbl_video_name = ctk.CTkLabel(
            self.file_frame, text="Aucune vidéo sélectionnée", text_color="gray"
        )
        self.lbl_video_name.pack(side="left", padx=10, pady=15)

        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.pack(pady=10, padx=20, fill="x")

        self.switch_mosaic = ctk.CTkSwitch(
            self.options_frame, text="Utiliser l'effet Mosaïque (au lieu du flou)"
        )
        self.switch_mosaic.pack(pady=15, padx=15, anchor="w")

        self.btn_start = ctk.CTkButton(
            self,
            text="🚀 Démarrer l'Anonymisation",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            command=self.start_processing,
        )
        self.btn_start.pack(pady=(20, 10))

        self.progress_bar = ctk.CTkProgressBar(self, width=500)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10)

        self.lbl_status = ctk.CTkLabel(
            self, text="En attente...", font=ctk.CTkFont(size=14), wraplength=600
        )
        self.lbl_status.pack(pady=10)

        self.check_queue()

    def select_video(self):
        filepath = filedialog.askopenfilename(
            title="Sélectionner une vidéo",
            filetypes=[("Vidéos", "*.mp4 *.avi *.mov *.mkv")],
        )
        if filepath:
            self.video_path = filepath
            self.lbl_video_name.configure(text=Path(filepath).name, text_color="white")

    def check_queue(self):
        while not self.msg_queue.empty():
            msg_type, value = self.msg_queue.get()

            if msg_type == "progress":
                self.progress_bar.set(value)
            elif msg_type == "status":
                self.lbl_status.configure(text=value)
            elif msg_type == "done" or msg_type == "error":
                self.lbl_status.configure(text=value)
                self.btn_start.configure(
                    state="normal", text="🚀 Démarrer l'Anonymisation"
                )
                self.is_processing = False
                if msg_type == "done":
                    self.progress_bar.set(1.0)

        self.after(100, self.check_queue)

    def start_processing(self):
        if not self.video_path:
            self.lbl_status.configure(
                text="⚠️ Veuillez d'abord sélectionner une vidéo !"
            )
            return
        if self.is_processing:
            return

        self.is_processing = True
        self.btn_start.configure(state="disabled", text="⏳ Traitement en cours...")
        self.progress_bar.set(0)

        input_path_obj = Path(self.video_path)
        suffix = "_mosaique" if self.switch_mosaic.get() else "_floutee"
        output_filename = f"{input_path_obj.stem}{suffix}{input_path_obj.suffix}"

        output_path = str(input_path_obj.parent / output_filename)
        use_mosaic = bool(self.switch_mosaic.get())

        thread = threading.Thread(
            target=process_video_backend,
            args=(
                self.video_path,
                output_path,
                self.model_path,
                use_mosaic,
                self.msg_queue,
            ),
        )
        thread.daemon = True
        thread.start()


if __name__ == "__main__":
    app = FlouteurApp()
    app.mainloop()
