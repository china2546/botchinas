"""
YOLO Training Script
--------------------
Designed to be configured locally and executed on a remote GPU machine.
Supports YOLO11 and YOLOv8 models from Ultralytics.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def check_environment():
    """Check and display available compute hardware."""
    print("=" * 60)
    print("Environment Diagnostic:")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] CUDA is available with {gpu_count} GPU(s).")
        print(f"[Device] Default GPU: {gpu_name}")
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Device] Total VRAM: {total_memory_gb:.2f} GB")
        default_device = "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device] Apple Silicon MPS (Metal Performance Shaders) is available.")
        default_device = "mps"
    else:
        print("[Device] No GPU detected. Running on CPU (training will be slow).")
        default_device = "cpu"
    print("=" * 60)
    return default_device


def train(args):
    detected_device = check_environment()
    chosen_device = args.device if args.device is not None else detected_device

    # Check data yaml existence
    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"[Error] Dataset configuration not found: {data_yaml}")
        print("Please ensure your dataset and data.yaml are prepared.")
        sys.exit(1)

    model_name = args.model
    if model_name.endswith(".py") and not Path(model_name).exists():
        suggested = model_name[:-3] + ".pt"
        print(f"[Notice] Model '{model_name}' ends with '.py'. Auto-correcting to '{suggested}'...")
        model_name = suggested

    print(f"Loading base model: {model_name}")
    model = YOLO(model_name)

    print("\nStarting YOLO Training...")
    print(f" - Data YAML: {data_yaml}")
    print(f" - Epochs:    {args.epochs}")
    print(f" - Img Size:  {args.imgsz}")
    print(f" - Batch:     {args.batch}")
    print(f" - Device:    {chosen_device}")
    print(f" - Workers:   {args.workers}")
    print(f" - Project:   {args.project}")
    print(f" - Run Name:  {args.name}\n")

    train_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": chosen_device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "save": True,
        "verbose": True,
    }

    results = model.train(**train_args)

    print("\nTraining completed successfully!")
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(args.project) / args.name
    best_weights = save_dir / "weights" / "best.pt"

    if best_weights.exists():
        print(f"-> Best model weights saved at: {best_weights.resolve()}")
    else:
        print(f"-> Check run directory for weights: {save_dir.resolve()}")

    # Run validation
    print("\nRunning validation on validation set...")
    val_results = model.val()

    # Optional ONNX export
    if args.export_onnx:
        print("\nExporting model to ONNX format...")
        model.export(format="onnx", imgsz=args.imgsz)
        print("-> Export finished.")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model on custom game dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Path to data.yaml dataset config file (default: 'dataset/data.yaml')."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base model to fine-tune (e.g., 'yolo11n.pt', 'yolo11s.pt', 'yolov8n.pt', 'yolo11n-seg.pt')."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (default: 640)."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16). Use -1 for automatic batch size calculation."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('0', '0,1', 'cpu', 'mps'). If not set, automatically detects GPU."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of dataloader worker threads (default: 2 for Windows stability)."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience in epochs (default: 30)."
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Directory to save training runs (default: 'runs/train')."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="game_monster_detector",
        help="Experiment name (default: 'game_monster_detector')."
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        help="Export the trained model to ONNX format after training."
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
