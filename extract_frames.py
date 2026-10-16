"""
Video Frame Extraction Utility
------------------------------
Extracts diverse frames from gameplay videos for dataset labeling.
Extracting at a low frame rate (e.g., 1-2 FPS) avoids duplicate frames.
"""

import argparse
import os
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    fps_sample: float = 1.0,
    prefix: str = "frame"
) -> int:
    """Extract frames from a single video at the specified sampling rate."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0 or total_frames <= 0:
        print(f"[Warning] Invalid FPS or frame count for {video_path}. Falling back to every 30 frames.")
        frame_interval = 30
    else:
        frame_interval = max(1, int(round(video_fps / fps_sample)))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    frame_idx = 0

    video_stem = Path(video_path).stem
    pbar = tqdm(total=total_frames, desc=f"Processing {video_stem}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_filename = f"{video_stem}_{prefix}_{saved_count:06d}.jpg"
            save_path = output_path / frame_filename
            cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} -> {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from gameplay video(s) for YOLO labeling.")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a single video file (e.g., videos/gameplay.mp4)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="videos",
        help="Directory containing video files if --video is not specified (default: 'videos')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="extracted_frames",
        help="Directory where extracted image frames will be saved (default: 'extracted_frames')."
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Number of frames to extract per second of video (default: 1.0). Use 0.5 for 1 frame every 2 sec."
    )

    args = parser.parse_args()

    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"}

    if args.video:
        if not os.path.isfile(args.video):
            print(f"[Error] File not found: {args.video}")
            return
        extract_frames_from_video(args.video, args.output_dir, fps_sample=args.fps)
    else:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"[Notice] Directory '{args.input_dir}' not found. Creating it for you...")
            input_path.mkdir(parents=True, exist_ok=True)
            print(f"Please put your gameplay videos into '{args.input_dir}/' and run this script again.")
            return

        videos = [p for p in input_path.iterdir() if p.suffix.lower() in video_extensions]
        if not videos:
            print(f"No video files found in '{args.input_dir}/' (supported: {', '.join(video_extensions)}).")
            return

        print(f"Found {len(videos)} video(s) to process.")
        total_extracted = 0
        for vid in videos:
            total_extracted += extract_frames_from_video(str(vid), args.output_dir, fps_sample=args.fps)

        print(f"\nDone! Total {total_extracted} frames extracted into '{args.output_dir}'.")
        print("Next step: Upload these frames to a labeling tool (e.g., Roboflow or CVAT) to annotate monsters & bosses.")


if __name__ == "__main__":
    main()
