# Game Monster & Boss Detection with YOLO

Train a custom YOLO model (YOLO11 / YOLOv8) to detect and isolate monsters and bosses from gameplay videos.

This repository is designed with a **distributed workflow**:
1. **Local Device (Mac):** Extract video frames, prepare scripts, configure Git.
2. **Labeling:** Annotate monsters and bosses (using [Roboflow](https://roboflow.com) or CVAT).
3. **Remote GPU Device (PC/Cloud/Colab):** Pull Git repository, download dataset, train using Nvidia CUDA GPU.
4. **Local Device:** Run inference on gameplay video and crop monsters.

---

## 📁 Project Structure

```text
.
├── extract_frames.py      # Extracts diverse frames from gameplay videos (1-2 FPS)
├── split_dataset.py       # Helper to split local images/labels into train/val
├── train.py               # Main training script (supports GPU auto-detection)
├── predict_video.py       # Runs model on gameplay video & crops monsters
├── data.example.yaml      # Dataset template configuration for YOLO
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignores videos, datasets, and heavy weights
├── videos/                # Put your gameplay recording videos here
└── extracted_frames/      # Output directory for extracted images
```

---

## 🚀 Complete Step-by-Step Guide

### Step 1: Extract Frames from Gameplay Video (On this Mac)

Do not train directly on raw 60 FPS video because adjacent frames are nearly identical. Instead, sample 1 frame per second:

```bash
# Install local dependencies
pip install -r requirements.txt

# Option A: Process a specific video
python extract_frames.py --video videos/my_gameplay.mp4 --fps 1.0

# Option B: Process all videos inside the videos/ folder
python extract_frames.py --input_dir videos --output_dir extracted_frames --fps 1.0
```
Extracted frames will be stored in `extracted_frames/`.

---

### Step 2: Annotate Monsters and Bosses

To teach YOLO what a "monster" or "boss" is, you need bounding boxes:

#### Recommended Method: Roboflow (Easiest & Fastest for Remote Training)
1. Go to [Roboflow.com](https://roboflow.com) (free account).
2. Create a new project: **Object Detection** (or **Instance Segmentation** if you want exact polygon outlines).
3. Upload images from `extracted_frames/`.
4. Draw bounding boxes around:
   - Label `monster` (for regular mobs)
   - Label `boss` (for bosses)
   *(Or just use a single label `monster` for all of them if you don't need to distinguish).*
5. Click **Generate Version** -> **Export Dataset** -> select **YOLOv8** / **YOLO11** format -> choose **download zip to computer** or copy the **terminal curl/download code**.

---

### Step 3: Push Code to Git (On this Mac)

Since you don't want to train on this Mac, push the codebase to GitHub/GitLab:

```bash
git init
git add .
git commit -m "Initial YOLO game monster detection pipeline"

# Replace with your repository URL:
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

> **Note:** `.gitignore` is already configured to exclude large video files, raw frames, and `.pt` weights, keeping your repository lightweight and fast.

---

### Step 4: Setup & Train on Remote Device (GPU Machine)

On your remote device (e.g., Windows PC with Nvidia GPU, Linux cloud server, or Google Colab):

#### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

#### 2. Create virtual environment & install PyTorch with CUDA
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch with Nvidia CUDA support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other requirements
pip install -r requirements.txt
```

#### 3. Place your dataset
Ensure your dataset is placed at `dataset/` with `dataset/data.yaml`:
```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

#### 4. Start Training
Run `train.py`:
```bash
# Fast training with nano model (recommended for first test)
python train.py --data dataset/data.yaml --model yolo11n.pt --epochs 100 --batch 16

# For higher accuracy (larger model)
python train.py --data dataset/data.yaml --model yolo11s.pt --epochs 150 --batch 16
```

The script will automatically detect CUDA GPU and save the best weights to:
`runs/train/game_monster_detector/weights/best.pt`

---

### Step 5: Test on Gameplay Video (Local or Remote)

Once training finishes, copy `best.pt` to your test machine and run:

```bash
# Detect monsters on gameplay video and save annotated video:
python predict_video.py --weights best.pt --source videos/test_gameplay.mp4 --conf 0.3

# Separate & crop every detected monster into individual image files:
python predict_video.py --weights best.pt --source videos/test_gameplay.mp4 --crop
```
All cropped monsters will be saved inside `runs/predict/gameplay_test/crops/`.
