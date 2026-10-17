"""
Transparent Click-Through Game Screen Overlay
---------------------------------------------
Draws live green bounding boxes DIRECTLY on top of your game without blocking
your mouse clicks or keyboard inputs (uses Windows Transparent Layered Window).

Features:
- 100% transparent & click-through (mouse clicks pass directly to the game)
- Corner selection: monitor the full screen or a specific corner (top-left, top-right, etc.)
- Green bounding boxes around monsters, bosses, items, and objectives
- Ultra-low latency GPU inference on GTX 1050 Ti
"""

import argparse
import ctypes
from pathlib import Path
import threading
import time
import tkinter as tk
import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    from PIL import ImageGrab


def get_region_rect(region_name: str, monitor_idx: int = 1):
    """Calculate screen coordinates and size for the chosen corner/region."""
    if HAS_MSS:
        with mss.mss() as sct:
            mon = sct.monitors[monitor_idx if monitor_idx < len(sct.monitors) else 1]
            sw, sh = mon["width"], mon["height"]
            sx, sy = mon["left"], mon["top"]
    else:
        sw, sh = 1920, 1080
        sx, sy = 0, 0

    hw, hh = sw // 2, sh // 2

    regions = {
        "full": (sx, sy, sw, sh),
        "center": (sx + sw // 4, sy + sh // 4, hw, hh),
        "top-left": (sx, sy, hw, hh),
        "top-right": (sx + hw, sy, hw, hh),
        "bottom-left": (sx, sy + hh, hw, hh),
        "bottom-right": (sx + hw, sy + hh, hw, hh),
    }
    return regions.get(region_name.lower(), regions["full"])


class TransparentGameOverlay:
    def __init__(self, args):
        self.args = args
        self.running = True

        # Load YOLO model
        weights_path = Path(args.weights)
        if not weights_path.exists():
            candidates = list(Path("runs").rglob("best.pt"))
            weights_path = candidates[-1] if candidates else weights_path

        print(f"Loading weights: {weights_path}")
        self.model = YOLO(str(weights_path))
        self.device = "0" if torch.cuda.is_available() else "cpu"
        print(f"CUDA Device: {self.device}")

        # Region bounds
        self.rx, self.ry, self.rw, self.rh = get_region_rect(args.region, args.monitor)
        print(f"Overlay Region: {args.region} -> X:{self.rx}, Y:{self.ry}, W:{self.rw}, H:{self.rh}")

        # Shared detections list for GUI thread
        self.boxes_lock = threading.Lock()
        self.current_boxes = []

        # Setup Tkinter Transparent Window
        self.root = tk.Tk()
        self.root.title("YOLO_Game_Overlay")
        self.root.geometry(f"{self.rw}x{self.rh}+{self.rx}+{self.ry}")
        self.root.overrideredirect(True)  # Borderless
        self.root.attributes("-topmost", True)

        # Transparent background color
        self.TRANS_COLOR = "#000001"
        self.root.attributes("-transparentcolor", self.TRANS_COLOR)
        self.root.config(bg=self.TRANS_COLOR)

        self.canvas = tk.Canvas(self.root, bg=self.TRANS_COLOR, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Make window click-through on Windows
        self._enable_click_through()

        # Key bindings
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.bind("<q>", lambda e: self.close())

        # Start background detection worker thread
        self.worker_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.worker_thread.start()

        # Start UI render loop
        self._render_loop()

    def _enable_click_through(self):
        """Use Win32 API to make window transparent to mouse clicks."""
        try:
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED = 0x00080000
            GWL_EXSTYLE = -20
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_LAYERED)
            print("[Success] Click-through mode enabled! Mouse clicks pass directly to your game.")
        except Exception as e:
            print(f"[Notice] Click-through setup info: {e}")

    def _detection_loop(self):
        """Continuously captures the screen and detects objects via GPU."""
        bbox = {"top": self.ry, "left": self.rx, "width": self.rw, "height": self.rh}
        sct = mss.mss() if HAS_MSS else None

        while self.running:
            try:
                # Capture frame
                if sct is not None:
                    raw = sct.grab(bbox)
                    frame = np.array(raw)[:, :, :3]
                else:
                    frame = np.array(ImageGrab.grab(bbox=(self.rx, self.ry, self.rx + self.rw, self.ry + self.rh)))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Inference
                results = self.model.predict(
                    source=frame,
                    conf=self.args.conf,
                    device=self.device,
                    imgsz=self.args.imgsz,
                    verbose=False
                )[0]

                new_boxes = []
                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                    if self.args.classes and cls_name not in self.args.classes:
                        continue

                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    new_boxes.append((x1, y1, x2, y2, cls_name, conf))

                with self.boxes_lock:
                    self.current_boxes = new_boxes

            except Exception:
                pass

        if sct is not None:
            sct.close()

    def _render_loop(self):
        """Redraws the green bounding boxes on the transparent canvas."""
        if not self.running:
            return

        self.canvas.delete("all")

        with self.boxes_lock:
            boxes_to_draw = list(self.current_boxes)

        for x1, y1, x2, y2, cls_name, conf in boxes_to_draw:
            # Green bounding box
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00FF00", width=3)

            # Label text with green background
            label = f" {cls_name.upper()} {conf:.2f} "
            tag_y = max(15, y1 - 10)
            self.canvas.create_rectangle(x1, tag_y - 12, x1 + len(label) * 8, tag_y + 4, fill="#00AA00", outline="#00FF00")
            self.canvas.create_text(x1 + 2, tag_y - 4, text=label, fill="#FFFFFF", font=("Arial", 9, "bold"), anchor="w")

        # Refresh every ~20ms (~50 FPS)
        self.root.after(20, self._render_loop)

    def close(self):
        self.running = False
        self.root.destroy()
        print("Transparent overlay closed.")

    def start(self):
        print("\nTransparent Overlay Active!")
        print(" - Green boxes are drawn directly on your game.")
        print(" - You can play normally (mouse clicks pass through).")
        print(" - Press Ctrl+C in terminal or close terminal to stop.\n")
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Transparent click-through green box overlay for games.")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/runs/train/game_monster_detector-3/weights/best.pt",
        help="Path to trained YOLO weights file."
    )
    parser.add_argument(
        "--region",
        type=str,
        default="full",
        choices=["full", "center", "top-left", "top-right", "bottom-left", "bottom-right"],
        help="Screen corner or region to overlay (default: 'full')."
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Monitor index to capture (default: 1)."
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
        help="Filter classes (e.g., --classes monster boss item objective)."
    )

    args = parser.parse_args()
    overlay = TransparentGameOverlay(args)
    overlay.start()


if __name__ == "__main__":
    main()
