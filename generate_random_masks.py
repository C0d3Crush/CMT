# -*- coding: utf-8 -*-
"""
generate_random_masks.py
────────────────────────────────────────────────────────────────────────────
Generates realistic vessel-like masks by extracting real vessel shapes from
COCO annotations and randomly placing/rotating/scaling them onto images.

Usage
─────
  # Generate masks for all training images
  python generate_random_masks.py \
      --annotations arcade/syntax/train/annotations/train.json \
      --images      arcade/syntax/train/images \
      --output      arcade/syntax/train/random_masks

  # Preview first mask
  python generate_random_masks.py \
      --annotations arcade/syntax/train/annotations/train.json \
      --images      arcade/syntax/train/images \
      --output      arcade/syntax/train/random_masks \
      --preview

Output
──────
  One mask per image, same filename, saved in --output directory.
  Masks are binary: 255 = vessel region (to inpaint), 0 = background.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


# ── Load vessel shapes from COCO ─────────────────────────────────────────────

def load_vessel_shapes(ann_path, exclude_category_name='stenosis'):
    """
    Extract all vessel polygon shapes from COCO annotations.
    Returns a list of numpy arrays, each shape (N, 2) normalized to [0, 1].
    """
    with open(ann_path) as f:
        coco = json.load(f)

    exclude_ids = {
        cat['id'] for cat in coco['categories']
        if cat['name'].lower() == exclude_category_name
    }

    shapes = []
    for ann in coco['annotations']:
        if ann['category_id'] in exclude_ids:
            continue
        for poly in ann['segmentation']:
            pts = np.array(list(zip(poly[0::2], poly[1::2])), dtype=np.float32)
            if len(pts) >= 3:
                mn  = pts.min(axis=0)
                mx  = pts.max(axis=0)
                rng = mx - mn
                if rng[0] > 0 and rng[1] > 0:
                    pts = (pts - mn) / rng  # normalize to [0, 1]
                    shapes.append(pts)

    print(f"  Loaded {len(shapes)} vessel shapes from annotations")
    return shapes


# ── Random mask generation ────────────────────────────────────────────────────

def place_shape(draw, shape, W, H, rng):
    """
    Place a single normalized vessel shape with random position,
    rotation and scale.
    """
    scale = rng.uniform(0.05, 0.4) * min(W, H)
    angle = rng.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    pts = (shape - 0.5) @ rot.T * scale
    cx  = rng.uniform(W * 0.1, W * 0.9)
    cy  = rng.uniform(H * 0.1, H * 0.9)
    pts[:, 0] += cx
    pts[:, 1] += cy

    xy = [(float(x), float(y)) for x, y in pts]
    if len(xy) >= 3:
        draw.polygon(xy, fill=255)


def generate_mask(W, H, shapes, n_shapes=5, rng=None):
    """Generate a single binary mask by randomly placing vessel shapes."""
    if rng is None:
        rng = np.random.default_rng()

    mask = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(mask)

    chosen = rng.choice(len(shapes), size=min(n_shapes, len(shapes)), replace=False)
    for idx in chosen:
        place_shape(draw, shapes[idx], W, H, rng)

    return mask


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic vessel masks from COCO annotations"
    )
    parser.add_argument('--annotations', required=True,
                        help='Path to COCO annotations JSON')
    parser.add_argument('--images',      required=True,
                        help='Directory of input images')
    parser.add_argument('--output',      required=True,
                        help='Output directory for generated masks')
    parser.add_argument('--n_shapes',    type=int, default=5,
                        help='Number of vessel shapes per mask (default: 5)')
    parser.add_argument('--seed',        type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--preview',     action='store_true',
                        help='Save only first mask as preview and exit')
    args = parser.parse_args()

    print("Loading vessel shapes...")
    shapes = load_vessel_shapes(args.annotations)

    if not shapes:
        print("ERROR: no vessel shapes found")
        return

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    images_path = Path(args.images)
    image_files = sorted(list(images_path.glob('*.png')) +
                         list(images_path.glob('*.jpg')) +
                         list(images_path.glob('*.jpeg')))

    if not image_files:
        print(f"ERROR: no images found in {args.images}")
        return

    print(f"Generating masks for {len(image_files)} images → {args.output}")
    rng = np.random.default_rng(args.seed)

    for i, img_path in enumerate(image_files):
        img  = Image.open(img_path)
        W, H = img.size
        mask = generate_mask(W, H, shapes, n_shapes=args.n_shapes, rng=rng)

        if args.preview:
            mask.save('mask_preview.png')
            side = Image.new('L', (W * 2, H), 0)
            side.paste(img.convert('L'), (0, 0))
            side.paste(mask, (W, 0))
            side.save('mask_preview_sidebyside.png')
            print("Saved: mask_preview.png + mask_preview_sidebyside.png")
            return

        mask.save(output_path / img_path.name)

        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            print(f"  {i + 1}/{len(image_files)} done")

    print(f"\nDone. {len(image_files)} masks saved to {args.output}")


if __name__ == '__main__':
    main()
