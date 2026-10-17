"""
Dataset Splitting Utility
-------------------------
Automatically splits images and label files into train and validation sets:
  dataset/
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    └── labels/val/

Supports:
1. Separate folders: --images images/ --labels labels/
2. Parent folder:    --source raw_data/ (which contains images/ and labels/)
3. Single mixed folder: --source all_data/ (images and .txt files together)
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def split_data(
    source_dir: str = None,
    images_dir: str = None,
    labels_dir: str = None,
    target_dir: str = "dataset",
    train_ratio: float = 0.8,
    seed: int = 42
):
    random.seed(seed)
    target_path = Path(target_dir)

    img_path = None
    lbl_path = None

    # Determine image and label directory locations
    if images_dir and labels_dir:
        img_path = Path(images_dir)
        lbl_path = Path(labels_dir)
    elif source_dir:
        src = Path(source_dir)
        if (src / "images").exists() and (src / "labels").exists():
            img_path = src / "images"
            lbl_path = src / "labels"
        elif (src / "images").exists():
            img_path = src / "images"
            lbl_path = src
        else:
            img_path = src
            lbl_path = src
    else:
        # Default fallback check in current working directory
        if Path("images").exists() and Path("labels").exists():
            img_path = Path("images")
            lbl_path = Path("labels")
        elif Path("raw_data").exists():
            img_path = Path("raw_data/images") if Path("raw_data/images").exists() else Path("raw_data")
            lbl_path = Path("raw_data/labels") if Path("raw_data/labels").exists() else Path("raw_data")

    if not img_path or not img_path.exists():
        print(f"[Error] Images directory not found: {img_path}")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in img_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not images:
        print(f"[Error] No images found in '{img_path}'.")
        return

    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    # Ensure at least 1 image in validation set if possible
    if split_idx == len(images) and len(images) > 1:
        split_idx = len(images) - 1

    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for split, split_imgs in [("train", train_images), ("val", val_images)]:
        img_out = target_path / "images" / split
        lbl_out = target_path / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_file in split_imgs:
            # Copy image
            shutil.copy2(img_file, img_out / img_file.name)

            # Locate matching label file
            txt_name = f"{img_file.stem}.txt"
            label_file = (lbl_path / txt_name) if lbl_path else (img_file.parent / txt_name)

            if label_file.exists():
                shutil.copy2(label_file, lbl_out / txt_name)
            else:
                # Background frame with no objects
                (lbl_out / txt_name).touch()

    # Create dataset/data.yaml if not present
    data_yaml_path = target_path / "data.yaml"
    if not data_yaml_path.exists():
        yaml_content = f"""# Auto-generated YOLO dataset configuration
path: {target_path.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: monster
  # 1: boss  # Uncomment if you labeled bosses as class 1
"""
        with open(data_yaml_path, "w") as f:
            f.write(yaml_content)
        print(f"Created dataset config at: {data_yaml_path}")

    print("=" * 60)
    print(f"Dataset split complete! Total images: {len(images)}")
    print(f" -> Train: {len(train_images)} images saved to '{target_path / 'images/train'}'")
    print(f" -> Val:   {len(val_images)} images saved to '{target_path / 'images/val'}'")
    print(f" -> Labels saved to '{target_path / 'labels'}'")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Split raw images and labels into train and val folders for YOLO."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Directory containing images and labels (or containing 'images/' and 'labels/' subfolders)."
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Path to the images folder."
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to the labels folder."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset",
        help="Output directory (default: 'dataset')."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Train ratio between 0.0 and 1.0 (default: 0.8 = 80%% train, 20%% val)."
    )
    args = parser.parse_args()

    split_data(
        source_dir=args.source,
        images_dir=args.images,
        labels_dir=args.labels,
        target_dir=args.output,
        train_ratio=args.ratio
    )


if __name__ == "__main__":
    main()
