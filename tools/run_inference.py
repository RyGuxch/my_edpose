#!/usr/bin/env python3
"""Run inference on a dataset split and export COCO-style predictions.

This script mirrors the project's evaluation pipeline to generate a JSON file
compatible with ``tools/visualize_predictions.py``.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, SequentialSampler

from datasets.coco_eval import convert_to_xywh
from datasets import build_dataset
from main import build_model_main
from util.config import Config, DictAction
import util.misc as utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export predictions for visualization", add_help=True)
    parser.add_argument("--config_file", "-c", required=True, help="Path to the model config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path to load model weights")
    parser.add_argument("--output", default="predictions.json", help="Path to save COCO-style predictions")
    parser.add_argument("--image_set", default="val", help="Dataset split to run (e.g., val/test)")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--score_thr", type=float, default=0.0, help="Optional score threshold for saving predictions")
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="Override settings in the config, e.g. key=value",
    )
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg_dict: Dict[str, Any] = cfg._cfg_dict.to_dict()
    for key, value in cfg_dict.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    defaults: Dict[str, Any] = {
        "use_ema": False,
        "debug": False,
        "dataset_file": "coco",
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "find_unused_params": False,
        "seed": 42,
        "resume": "",
        "pretrain_model_path": None,
        "finetune_ignore": None,
        "start_epoch": 0,
        "eval": False,
        "test": False,
        "save_results": False,
        "save_log": False,
        "dist_url": "env://",
        "amp": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    if args.dataset_file == "coco":
        args.coco_path = os.environ.get("EDPOSE_COCO_PATH")
    elif args.dataset_file == "crowdpose":
        args.crowdpose_path = os.environ.get("EDPOSE_CrowdPose_PATH")
    elif args.dataset_file == "humanart":
        args.humanart_path = os.environ.get("EDPOSE_HumanArt_PATH")

    return args


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    device = torch.device(args.device)
    model, _, postprocessors = build_model_main(args)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, postprocessors


def build_data_loader(args: argparse.Namespace):
    dataset = build_dataset(image_set=args.image_set, args=args)
    sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )


def export_predictions(args: argparse.Namespace):
    args = merge_config(args)
    device = torch.device(args.device)
    model, postprocessors = load_model(args)
    data_loader = build_data_loader(args)

    results = []
    with torch.no_grad():
        for samples, targets in utils.MetricLogger(delimiter="  ").log_every(data_loader, 10, "Inference:"):
            samples = samples.to(device)
            outputs = model(samples)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            processed = postprocessors["bbox"](outputs, orig_sizes)

            for target, pred in zip(targets, processed):
                image_id = int(target["image_id"])
                boxes_xywh = convert_to_xywh(pred["boxes"]).tolist()
                scores = pred["scores"].tolist()
                labels = pred["labels"].tolist()
                keypoints = pred.get("keypoints")
                keypoints_list = None
                if keypoints is not None:
                    keypoints_list = keypoints.flatten(start_dim=1).tolist()

                for idx, (box, score, label) in enumerate(zip(boxes_xywh, scores, labels)):
                    if score < args.score_thr:
                        continue
                    entry: Dict[str, Any] = {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box,
                        "score": float(score),
                    }
                    if keypoints_list is not None:
                        entry["keypoints"] = keypoints_list[idx]
                    results.append(entry)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} predictions to {output_path}")


def main():
    args = parse_args()
    utils.init_distributed_mode(args)
    export_predictions(args)


if __name__ == "__main__":
    main()
