#!/usr/bin/env python3
"""Visualize model predictions against ground truth annotations.

This script overlays predicted boxes/keypoints and ground-truth labels on the
original images to make it easy to inspect qualitative performance.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402


def _load_predictions(prediction_path: Path, score_thr: float):
    with prediction_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    pred_by_image = defaultdict(list)
    for pred in predictions:
        if pred.get("score", 1.0) < score_thr:
            continue
        pred_by_image[pred["image_id"]].append(pred)
    return pred_by_image


def _get_skeleton(coco: COCO):
    cat_ids = coco.getCatIds()
    if not cat_ids:
        return None
    categories = coco.loadCats(cat_ids)
    for cat in categories:
        if "skeleton" in cat:
            return np.array(cat["skeleton"]) - 1
    return None


def _draw_keypoints(ax, keypoints, skeleton, color: str):
    kp = np.array(keypoints).reshape(-1, 3)
    xs, ys, vs = kp[:, 0], kp[:, 1], kp[:, 2]
    visible = vs > 0
    ax.scatter(xs[visible], ys[visible], s=10, color=color, edgecolors="black", linewidths=0.5)
    if skeleton is None:
        return
    for sk in skeleton:
        sk = np.array(sk)
        if (vs[sk] > 0).all():
            ax.plot(xs[sk], ys[sk], color=color, linewidth=2)


def _draw_instances(ax, anns, skeleton, color: str, label_prefix: str):
    for ann in anns:
        bbox = ann.get("bbox")
        if bbox is not None:
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, fill=False, color=color, linewidth=2)
            ax.add_patch(rect)
            label = label_prefix
            if "score" in ann:
                label = f"{label_prefix}{ann['score']:.2f}"
            ax.text(x, y - 2, label, color="white", bbox={"facecolor": color, "alpha": 0.6, "pad": 2})
        if "keypoints" in ann and ann["keypoints"] is not None:
            _draw_keypoints(ax, ann["keypoints"], skeleton, color)


def _iter_image_ids(coco: COCO, limit: Optional[int], requested: Optional[Set[int]]):
    all_ids = coco.getImgIds()
    if requested:
        all_ids = [img_id for img_id in all_ids if img_id in requested]
    if limit is not None:
        all_ids = all_ids[:limit]
    return all_ids


def visualize(args):
    coco = COCO(args.annotations)
    skeleton = _get_skeleton(coco)
    prediction_path = Path(args.predictions)
    pred_by_image = _load_predictions(prediction_path, args.score_thr)
    requested_ids = set(args.image_ids) if args.image_ids else None
    image_ids = _iter_image_ids(coco, args.max_images, requested_ids)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        info = coco.loadImgs([image_id])[0]
        img_path = Path(args.image_root) / info["file_name"]
        if not img_path.exists():
            print(f"[warn] missing image file: {img_path}")
            continue
        image = plt.imread(str(img_path))
        fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
        ax.imshow(image)
        ax.axis("off")

        ann_ids = coco.getAnnIds(imgIds=[image_id])
        gt_anns = coco.loadAnns(ann_ids)
        _draw_instances(ax, gt_anns, skeleton, color="lime", label_prefix="gt ")

        pred_anns = pred_by_image.get(image_id, [])
        _draw_instances(ax, pred_anns, skeleton, color="red", label_prefix="pred ")

        ax.set_title(f"image_id: {image_id}")
        output_path = output_dir / f"{image_id}.png"
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Overlay predictions and ground truth on images.")
    parser.add_argument("predictions", help="Path to COCO-style prediction JSON file.")
    parser.add_argument("annotations", help="Path to COCO annotations JSON file for ground truth.")
    parser.add_argument("image_root", help="Directory containing the dataset images.")
    parser.add_argument("--output-dir", default="vis_compare", help="Directory to save rendered images.")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Minimum prediction score to visualize.")
    parser.add_argument(
        "--max-images", type=int, default=50, help="Maximum number of images to render (default: 50). Use -1 for all."
    )
    parser.add_argument("--image-ids", type=int, nargs="*", help="Optional list of image ids to visualize.")

    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 0:
        args.max_images = None
    visualize(args)


if __name__ == "__main__":
    main()
