"""
Dataset Splitting Utility
-------------------------
Splits a folder of labeled images and .txt label files into:
  dataset/
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    └── labels/val/
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def split_data(source_dir: str, target_dir: str, train_ratio: float = 0.8, seed: int = 42):
    random.seed(seed)
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    # Find all images
    images = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    if not images:
        print(f"[Error] No images found in '{source_dir}'.")
        return

    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
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

            # Copy corresponding .txt label if it exists
            txt_file = img_file.with_suffix(".txt")
            if txt_file.exists():
                shutil.copy2(txt_file, lbl_out / txt_file.name)
            else:
                # If no label file, create an empty one (background image without objects)
                (lbl_out / f"{img_file.stem}.txt").touch()

    print(f"Successfully split {len(images)} images:")
    print(f" - Train: {len(train_images)} images -> {target_path / 'images/train'}")
    print(f" - Val:   {len(val_images)} images   -> {target_path / 'images/val'}")


def main():
    parser = argparse.ArgumentParser(description="Split labeled images and labels into train and val folders.")
    parser.add_argument("--source", type=str, required=True, help="Directory containing images and .txt files.")
    parser.add_argument("--output", type=str, default="dataset", help="Output dataset directory (default: 'dataset').")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train ratio between 0.0 and 1.0 (default: 0.8).")
    args = parser.parse_args()

    split_data(args.source, args.output, train_ratio=args.ratio)


if __name__ == "__main__":
    main()
