# Flouteur Vidéo (Sfloutage)

Un outil d'anonymisation vidéo automatique propulsé par l'IA (YOLOv8 + ONNX Runtime), doté d'une interface graphique légère et capable de conserver la piste audio originale.

Conçu pour être rapide, portable et fonctionner sans configuration complexe sur le processeur (CPU) ou la carte graphique (GPU).

## Utilisation rapide (Windows & Linux)

Si vous cherchez simplement à utiliser le logiciel sans toucher au code, téléchargez l'exécutable prêt à l'emploi dans l'onglet **Releases**.

1. Extrayez le dossier.
2. Glissez votre fichier modèle (`best.onnx`) à l'intérieur du dossier extrait.
3. Lancez l'application en double-cliquant sur l'exécutable (`FlouteurGUI`).

---

## 🛠️ Compiler ou exécuter depuis le code source

Cette méthode est recommandée pour les utilisateurs de macOS, BSD, ou les développeurs souhaitant modifier l'outil ou le compiler nativement pour leur architecture.

### 1. Prérequis système

* **Python 3.10** ou supérieur.
* **FFmpeg** (indispensable pour recoller le son sur la vidéo anonymisée) :
  * **Arch Linux** : `sudo pacman -S ffmpeg`
  * **Debian / Ubuntu** : `sudo apt install ffmpeg`
  * **macOS** : `brew install ffmpeg`
  * **Windows** : Téléchargez l'exécutable depuis [ffmpeg.org](https://ffmpeg.org/) et placez-le dans le dossier de l'application (ou ajoutez-le au PATH).

### 2. Installation de l'environnement Python

Clonez le dépôt et installez les dépendances. L'utilisation de `uv` est fortement recommandée pour sa rapidité.

```bash
git clone [https://github.com/VOTRE_PSEUDO/Sfloutage.git](https://github.com/VOTRE_PSEUDO/Sfloutage.git)
cd Sfloutage

# 1. Créer et activer l'environnement virtuel
uv venv
source .venv/bin/activate  # Sous Windows : .venv\Scripts\activate.bat

# 2. Installer les dépendances et les outils de compilation
uv pip install customtkinter opencv-python ultralytics onnx onnxruntime pyinstaller tqdm

```

### 3. Ajout du modèle d'Intelligence Artificielle

Le dépôt ne contient pas le poids du modèle par défaut pour des raisons de taille.
Placez votre modèle entraîné au format ONNX (par défaut nommé `best.onnx`) à la racine du projet, juste à côté du fichier `FlouteurGUI.py`.

### 4. Lancement direct (Mode Développeur)

Une fois l'environnement activé, lancez simplement l'interface graphique :

```bash
python FlouteurGUI.py

```

> 💡 **Version CLI incluse** : Si vous préférez le terminal ou souhaitez automatiser le floutage, utilisez `python Sfloutage.py -i input.mp4 -o output.mp4 --mosaic`.

### 5. Compiler son propre exécutable (Build)

Pour générer un dossier logiciel autonome (portable) adapté à votre propre système d'exploitation, utilisez PyInstaller :

```bash
pyinstaller --noconsole --collect-all customtkinter FlouteurGUI.py

```

Une fois l'opération terminée :

1. Allez dans le dossier généré : `dist/FlouteurGUI/`
2. Copiez manuellement votre fichier `best.onnx` dans ce dossier.
3. Votre logiciel est prêt à être distribué ! Vous pouvez supprimer les dossiers `build/` temporaires.
