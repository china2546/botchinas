"""
Fix Dataset YAML Classes
------------------------
Scans the dataset labels to detect all class IDs, or copies class names from
an existing Roboflow data.yaml, and updates dataset/data.yaml so that YOLO
does not reject images with 'Label class X exceeds dataset class count'.
"""

import argparse
from pathlib import Path
import yaml


def scan_classes(dataset_dir: str = "dataset", source_yaml: str = None):
    dataset_path = Path(dataset_dir)
    labels_dir = dataset_path / "labels"

    if not labels_dir.exists():
        print(f"[Error] Labels directory not found at: {labels_dir}")
        return

    # Scan all .txt files to find all unique class indices
    class_ids = set()
    txt_files = list(labels_dir.rglob("*.txt"))
    for txt in txt_files:
        try:
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_ids.add(int(parts[0]))
        except Exception:
            continue

    if not class_ids:
        print("[Warning] No class labels found in .txt files.")
        return

    sorted_ids = sorted(list(class_ids))
    max_id = max(sorted_ids)
    print(f"Found {len(sorted_ids)} unique classes in dataset: {sorted_ids} (max class ID: {max_id})")

    # Check if a source yaml with names exists (e.g., from Roboflow)
    known_names = {}
    candidate_yamls = []
    if source_yaml:
        candidate_yamls.append(Path(source_yaml))
    candidate_yamls.extend([
        Path("train/data.yaml"),
        Path("data.yaml"),
        Path("train/README.roboflow.txt"),
        Path("README.roboflow.txt"),
    ])

    for yml in candidate_yamls:
        if yml.exists():
            try:
                with open(yml, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and "names" in data:
                        raw_names = data["names"]
                        if isinstance(raw_names, list):
                            known_names = {i: name for i, name in enumerate(raw_names)}
                        elif isinstance(raw_names, dict):
                            known_names = {int(k): v for k, v in raw_names.items()}
                        print(f"Loaded class names from {yml}: {known_names}")
                        break
            except Exception:
                pass

    # Build names dictionary for all classes up to max_id
    names_dict = {}
    for i in range(max_id + 1):
        if i in known_names:
            names_dict[i] = known_names[i]
        else:
            names_dict[i] = f"class_{i}"

    target_yaml = dataset_path / "data.yaml"
    yaml_data = {
        "path": dataset_path.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names_dict),
        "names": names_dict,
    }

    with open(target_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    # Remove old YOLO label cache files so YOLO rescans the labels with new classes
    cache_files = list(labels_dir.rglob("*.cache"))
    for cache in cache_files:
        try:
            cache.unlink()
            print(f"Cleared old cache: {cache.name}")
        except Exception:
            pass

    print("\n" + "=" * 60)
    print(f"Successfully updated '{target_yaml}' with {len(names_dict)} classes:")
    for k, v in names_dict.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print("You can now run training without class count errors!")


def main():
    parser = argparse.ArgumentParser(description="Fix dataset/data.yaml classes.")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory (default: 'dataset').")
    parser.add_argument("--source_yaml", type=str, default=None, help="Path to Roboflow's original data.yaml if available.")
    args = parser.parse_args()
    scan_classes(dataset_dir=args.dataset, source_yaml=args.source_yaml)


if __name__ == "__main__":
    main()
