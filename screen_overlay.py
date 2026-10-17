"""
Real-time Game Screen Overlay & Monster Detector
------------------------------------------------
Captures your screen/monitor in real-time and draws bright green bounding
boxes around detected monsters, bosses, items, and objectives.

Features:
- Corner/Region selection (full screen, center, top-left, top-right, bottom-left, bottom-right)
- Multi-monitor support (--monitor 1, 2)
- Always-on-top HUD window positioned in any corner of your choice
- Real-time FPS counter and detection statistics
- Uses GTX 1050 Ti GPU acceleration via CUDA
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Try importing mss for high-performance screen capture; fallback to PIL.ImageGrab
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    from PIL import ImageGrab


def get_screen_dimensions(monitor_idx=1):
    """Get dimensions of chosen monitor."""
    if HAS_MSS:
        with mss.mss() as sct:
            monitors = sct.monitors
            idx = monitor_idx if monitor_idx < len(monitors) else 1
            return monitors[idx]["width"], monitors[idx]["height"], monitors[idx]
    else:
        # Fallback using PIL
        img = ImageGrab.grab()
        return img.width, img.height, {"left": 0, "top": 0, "width": img.width, "height": img.height}


def calculate_region(region_name: str, monitor_idx: int = 1):
    """Calculate the pixel bounding box for the selected corner or region."""
    screen_w, screen_h, mon_meta = get_screen_dimensions(monitor_idx)
    offset_x = mon_meta.get("left", 0)
    offset_y = mon_meta.get("top", 0)

    half_w = screen_w // 2
    half_h = screen_h // 2

    regions = {
        "full": {"top": offset_y, "left": offset_x, "width": screen_w, "height": screen_h},
        "center": {"top": offset_y + screen_h // 4, "left": offset_x + screen_w // 4, "width": half_w, "height": half_h},
        "top-left": {"top": offset_y, "left": offset_x, "width": half_w, "height": half_h},
        "top-right": {"top": offset_y, "left": offset_x + half_w, "width": half_w, "height": half_h},
        "bottom-left": {"top": offset_y + half_h, "left": offset_x, "width": half_w, "height": half_h},
        "bottom-right": {"top": offset_y + half_h, "left": offset_x + half_w, "width": half_w, "height": half_h},
    }

    selected = regions.get(region_name.lower(), regions["full"])
    return selected


def capture_frame(sct, bbox):
    """Grab a screen frame as a numpy BGR image."""
    if sct is not None:
        raw = sct.grab(bbox)
        # Convert raw BGRA to BGR
        img = np.array(raw)[:, :, :3]
        return np.ascontiguousarray(img)
    else:
        # Fallback to PIL ImageGrab
        bbox_tuple = (bbox["left"], bbox["top"], bbox["left"] + bbox["width"], bbox["top"] + bbox["height"])
        img = ImageGrab.grab(bbox=bbox_tuple)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def run_overlay(args):
    # Locate weights
    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Search runs directory if default path isn't found
        candidates = list(Path("runs").rglob("best.pt"))
        if candidates:
            weights_path = candidates[-1]
            print(f"[Notice] Using discovered weights at: {weights_path}")
        else:
            print(f"[Error] Weights file not found: {args.weights}")
            print("Please provide the correct path using --weights.")
            sys.exit(1)

    print(f"Loading YOLO model: {weights_path}")
    model = YOLO(str(weights_path))

    # Device selection
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Determine capture region
    capture_bbox = calculate_region(args.region, monitor_idx=args.monitor)
    print(f"Selected Region: '{args.region}' -> X:{capture_bbox['left']}, Y:{capture_bbox['top']}, W:{capture_bbox['width']}, H:{capture_bbox['height']}")

    # Setup window
    window_name = "Game AI Overlay [Monster / Boss Detector]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    # Position window according to user preference
    display_w = min(capture_bbox["width"], 960)
    display_h = int(display_w * (capture_bbox["height"] / capture_bbox["width"]))
    cv2.resizeWindow(window_name, display_w, display_h)

    # Set window initial position on screen
    if args.position == "top-right":
        cv2.moveWindow(window_name, capture_bbox["left"] + capture_bbox["width"] - display_w - 20, capture_bbox["top"] + 40)
    elif args.position == "top-left":
        cv2.moveWindow(window_name, capture_bbox["left"] + 20, capture_bbox["top"] + 40)
    elif args.position == "bottom-right":
        cv2.moveWindow(window_name, capture_bbox["left"] + capture_bbox["width"] - display_w - 20, capture_bbox["top"] + capture_bbox["height"] - display_h - 60)
    elif args.position == "bottom-left":
        cv2.moveWindow(window_name, capture_bbox["left"] + 20, capture_bbox["top"] + capture_bbox["height"] - display_h - 60)

    sct = mss.mss() if HAS_MSS else None

    # Colors (Green box as requested: (0, 255, 0))
    GREEN_COLOR = (0, 255, 0)
    TEXT_BG_COLOR = (0, 180, 0)
    WHITE_COLOR = (255, 255, 255)

    print("\nOverlay running! Controls:")
    print(" - Press 'q' in the overlay window to exit.")
    print(" - Press 'c' to cycle regions (full -> center -> top-left -> top-right -> bottom-left -> bottom-right).")
    print(" - Press '+' / '-' to increase / decrease confidence threshold.\n")

    region_list = ["full", "center", "top-left", "top-right", "bottom-left", "bottom-right"]
    curr_region_idx = region_list.index(args.region) if args.region in region_list else 0
    current_conf = args.conf

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            # Capture frame
            frame = capture_frame(sct, capture_bbox)

            # Run inference
            results = model.predict(
                source=frame,
                conf=current_conf,
                device=device,
                imgsz=args.imgsz,
                verbose=False
            )[0]

            # Calculate FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # Draw green boxes and labels
            detected_counts = {}
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"cls_{cls_id}")
                conf_val = float(box.conf[0].item())

                # Filter classes if user specified
                if args.classes and cls_name not in args.classes:
                    continue

                detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Draw green bounding box (thickness = 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN_COLOR, 2)

                # Label text
                label = f"{cls_name.upper()} {conf_val:.2f}"
                (txt_w, txt_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Draw solid label tag above box
                tag_y1 = max(0, y1 - txt_h - 6)
                cv2.rectangle(frame, (x1, tag_y1), (x1 + txt_w + 6, y1), GREEN_COLOR, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw HUD stats banner at the top
            hud_text = f"FPS: {fps:.1f} | Conf: {current_conf:.2f} | Region: {region_list[curr_region_idx]}"
            if detected_counts:
                counts_str = ", ".join([f"{k}: {v}" for k, v in detected_counts.items()])
                hud_text += f" | Detected: [{counts_str}]"
            else:
                hud_text += " | No objects detected"

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(frame, hud_text, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

            # Display window
            cv2.imshow(window_name, frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or Esc
                break
            elif key == ord('c'):  # Cycle region
                curr_region_idx = (curr_region_idx + 1) % len(region_list)
                args.region = region_list[curr_region_idx]
                capture_bbox = calculate_region(args.region, monitor_idx=args.monitor)
                print(f"[Switched Region] -> {args.region}")
            elif key == ord('+') or key == ord('='):
                current_conf = min(0.95, current_conf + 0.05)
                print(f"[Confidence] -> {current_conf:.2f}")
            elif key == ord('-') or key == ord('_'):
                current_conf = max(0.05, current_conf - 0.05)
                print(f"[Confidence] -> {current_conf:.2f}")

    finally:
        if sct is not None:
            sct.close()
        cv2.destroyAllWindows()
        print("Overlay closed.")


def main():
    parser = argparse.ArgumentParser(description="Real-time screen overlay with green bounding boxes for game objects.")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/runs/train/game_monster_detector-3/weights/best.pt",
        help="Path to trained YOLO weights file (default: best.pt from last run)."
    )
    parser.add_argument(
        "--region",
        type=str,
        default="full",
        choices=["full", "center", "top-left", "top-right", "bottom-left", "bottom-right"],
        help="Screen region to monitor (default: 'full')."
    )
    parser.add_argument(
        "--position",
        type=str,
        default="top-right",
        choices=["top-right", "top-left", "bottom-right", "bottom-left"],
        help="Where on your screen to pin the overlay HUD window (default: 'top-right')."
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Monitor index to capture (1 for primary, 2 for secondary). Default: 1."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640)."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Specific classes to detect (e.g. --classes monster boss item objective)."
    )

    args = parser.parse_args()
    run_overlay(args)


if __name__ == "__main__":
    main()
