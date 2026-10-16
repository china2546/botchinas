"""
YOLO Video Inference and Object Separation Utility
--------------------------------------------------
Runs the trained YOLO model on gameplay videos, draws bounding boxes around
monsters/bosses, and optionally crops detected monsters into separate images.
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def run_inference(args):
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[Error] Weights file not found: {weights_path}")
        print("Please train a model first or provide the path to your trained .pt file via --weights.")
        return

    print(f"Loading trained model: {weights_path}")
    model = YOLO(str(weights_path))

    source_path = Path(args.source)
    if not source_path.exists() and args.source != "0":
        print(f"[Error] Source video or image not found: {args.source}")
        return

    print(f"Running inference on: {args.source}")
    print(f" - Confidence Threshold: {args.conf}")
    print(f" - Output Directory:     {args.output_dir}")
    print(f" - Crop Objects:         {args.crop}")

    # Create crops folder if crop enabled
    crops_dir = Path(args.output_dir) / "crops"
    if args.crop:
        crops_dir.mkdir(parents=True, exist_ok=True)

    # Run prediction stream
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        project=args.output_dir,
        name=args.name,
        stream=True,
        show=args.show,
        save_crop=args.crop,
    )

    frame_count = 0
    total_detections = 0

    for result in results:
        frame_count += 1
        num_boxes = len(result.boxes)
        total_detections += num_boxes

        if frame_count % 60 == 0:
            print(f"Processed frame {frame_count} | Detections in frame: {num_boxes}")

    print("\nInference finished!")
    print(f" - Total frames processed: {frame_count}")
    print(f" - Total detections:       {total_detections}")
    print(f" - Annotated video saved in: {Path(args.output_dir) / args.name}")
    if args.crop:
        print(f" - Cropped monsters saved in: {Path(args.output_dir) / args.name / 'crops'}")


def main():
    parser = argparse.ArgumentParser(description="Run trained YOLO model on gameplay video.")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/game_monster_detector/weights/best.pt",
        help="Path to trained model weights (.pt)."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to gameplay video or image (e.g., 'videos/boss_fight.mp4')."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)."
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/predict",
        help="Directory to save output results (default: 'runs/predict')."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="gameplay_test",
        help="Run name for output folder."
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop and save each detected monster into separate image files."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save annotated video to disk (default: True)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live video window while processing (requires desktop display)."
    )

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
