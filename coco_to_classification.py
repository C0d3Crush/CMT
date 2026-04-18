"""
coco_to_classification.py
─────────────────────────────────────────────────────────────────────────────
Converts a COCO-annotated dataset (e.g. ARCADE) into a Places365-style
classification folder tree so that Places365's train.py can be used
with minimal changes.

Strategy: dominant class per image
  → For each image, the category that appears most often in the annotations
    is used as the single class label.
  → Ties are broken by the lower category_id (deterministic).

Usage
─────
  # Symlinks only (fast, saves disk space) – default
  python coco_to_classification.py \
      --annotations path/to/annotations.json \
      --images      path/to/images/ \
      --output      data/train

  # Physically copy images instead
  python coco_to_classification.py \
      --annotations path/to/annotations.json \
      --images      path/to/images/ \
      --output      data/train \
      --copy

  # Multiple annotation splits (train + val)
  python coco_to_classification.py \
      --annotations path/to/train.json \
      --images      path/to/images/ \
      --output      data/train

  python coco_to_classification.py \
      --annotations path/to/val.json \
      --images      path/to/images/ \
      --output      data/val

Output
──────
  data/train/
    class_0/   image_001.png  image_042.png  ...
    class_1/   image_007.png  ...
    ...
  data/train/class_map.json   # {class_idx: original_category_name}
  data/train/image_labels.csv # image_filename, class_idx  (for debugging)
"""

import argparse
import json
import os
import shutil
import csv
from collections import Counter
from pathlib import Path


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_coco(annotation_path: str) -> tuple[dict, dict, dict]:
    """
    Returns:
        images      : {image_id: file_name}
        categories  : {category_id: name}
        annotations : {image_id: Counter({category_id: count})}
    """
    with open(annotation_path) as f:
        coco = json.load(f)

    images = {img["id"]: img["file_name"] for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    annotations: dict[int, Counter] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id not in annotations:
            annotations[img_id] = Counter()
        annotations[img_id][cat_id] += 1

    return images, categories, annotations


def dominant_category(counter: Counter) -> int:
    """Most frequent category_id; ties broken by lowest id."""
    max_count = max(counter.values())
    candidates = sorted(k for k, v in counter.items() if v == max_count)
    return candidates[0]


def build_class_index(categories: dict) -> tuple[dict, dict]:
    """
    Maps original category_ids to 0-based class indices.
    Returns:
        cat_to_idx : {category_id: class_idx}
        idx_to_name: {class_idx: category_name}
    """
    sorted_ids = sorted(categories.keys())
    cat_to_idx  = {cat_id: idx for idx, cat_id in enumerate(sorted_ids)}
    idx_to_name = {idx: categories[cat_id] for cat_id, idx in cat_to_idx.items()}
    return cat_to_idx, idx_to_name


# ─── main ─────────────────────────────────────────────────────────────────────

def convert(annotation_path: str,
            images_dir: str,
            output_dir: str,
            copy: bool = False) -> None:

    print(f"Loading annotations from: {annotation_path}")
    images, categories, annotations = load_coco(annotation_path)
    cat_to_idx, idx_to_name = build_class_index(categories)

    num_classes = len(categories)
    print(f"  → {len(images)} images | {num_classes} categories")

    output_path = Path(output_dir)
    images_path = Path(images_dir)

    # Create class subdirectories
    for idx in range(num_classes):
        (output_path / f"class_{idx}").mkdir(parents=True, exist_ok=True)

    skipped = 0
    label_rows = []

    for img_id, file_name in images.items():

        # Images with no annotations → skip (background / unannotated)
        if img_id not in annotations:
            skipped += 1
            continue

        cat_id    = dominant_category(annotations[img_id])
        class_idx = cat_to_idx[cat_id]

        src = images_path / file_name
        if not src.exists():
            print(f"  [WARN] image not found, skipping: {src}")
            skipped += 1
            continue

        dst = output_path / f"class_{class_idx}" / src.name

        if copy:
            shutil.copy2(src, dst)
        else:
            # Symlink with absolute path so the tree is portable
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

        label_rows.append((src.name, class_idx))

    # ── write metadata ────────────────────────────────────────────────────────

    class_map_path = output_path / "class_map.json"
    with open(class_map_path, "w") as f:
        json.dump(idx_to_name, f, indent=2, ensure_ascii=False)

    labels_csv_path = output_path / "image_labels.csv"
    with open(labels_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class_idx"])
        writer.writerows(sorted(label_rows))

    mode = "copied" if copy else "symlinked"
    print(f"\nDone.")
    print(f"  {len(label_rows)} images {mode} → {output_dir}")
    print(f"  {skipped} images skipped (no annotations or missing file)")
    print(f"  class map  → {class_map_path}")
    print(f"  label CSV  → {labels_csv_path}")
    print(f"\nClass overview:")
    for idx, name in sorted(idx_to_name.items()):
        folder = output_path / f"class_{idx}"
        n = sum(1 for _ in folder.iterdir()) if folder.exists() else 0
        print(f"  class_{idx:>3}  ({name:<30})  {n:>5} images")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COCO → Places365-style classification tree"
    )
    parser.add_argument("--annotations", required=True,
                        help="Path to COCO annotations JSON")
    parser.add_argument("--images",      required=True,
                        help="Directory containing the raw images")
    parser.add_argument("--output",      required=True,
                        help="Output directory (e.g. data/train)")
    parser.add_argument("--copy",        action="store_true",
                        help="Copy images instead of symlinking (uses more disk)")
    args = parser.parse_args()

    convert(
        annotation_path=args.annotations,
        images_dir=args.images,
        output_dir=args.output,
        copy=args.copy,
    )
