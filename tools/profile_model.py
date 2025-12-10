#!/usr/bin/env python
import argparse
import importlib.util
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Tuple

import torch
from main import build_model_main
from util.config import Config, DictAction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Profile model parameters, flops, and latency", add_help=True
    )
    parser.add_argument("--config_file", "-c", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="Override settings in the config, e.g. key=value",
    )
    parser.add_argument("--device", default="cpu", help="Device used for profiling")
    parser.add_argument("--batch_size", type=int, default=1, help="Synthetic batch size")
    parser.add_argument("--height", type=int, default=640, help="Input height for the synthetic batch")
    parser.add_argument("--width", type=int, default=640, help="Input width for the synthetic batch")
    parser.add_argument("--channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--warmup_iters", type=int, default=5, help="Warmup iterations for latency")
    parser.add_argument("--measure_iters", type=int, default=20, help="Measured iterations for latency")
    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision for latency test"
    )
    parser.add_argument(
        "--skip_flops", action="store_true", help="Skip flop computation even if fvcore is available"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional path to save the profiling summary as JSON"
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
    defaults = {
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
        "num_workers": 0,
        "test": False,
        "save_results": False,
        "save_log": False,
        "dist_url": "env://",
        "amp": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    model, _, _ = build_model_main(args)
    model.to(torch.device(args.device))
    model.eval()
    return model


def get_synthetic_input(args: argparse.Namespace) -> torch.Tensor:
    shape = (args.batch_size, args.channels, args.height, args.width)
    return torch.randn(shape, device=args.device)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def parameter_size_mb(model: torch.nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


def compute_flops(model: torch.nn.Module, inputs: torch.Tensor) -> Tuple[int, str]:
    if not importlib.util.find_spec("fvcore.nn"):
        return 0, "fvcore not installed; skipping flop computation"
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    analysis = FlopCountAnalysis(model, inputs)
    total_flops = analysis.total()
    table = flop_count_table(analysis)
    return total_flops, table


def measure_latency(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    warmup_iters: int,
    measure_iters: int,
    use_amp: bool,
    batch_size: int,
) -> Dict[str, float]:
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    timings = []
    autocast_enabled = use_amp and inputs.device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup_iters):
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                _ = model(inputs)
        for _ in range(measure_iters):
            start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                _ = model(inputs)
            if inputs.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)
    timings_ms = [t * 1000 for t in timings]
    sorted_timings = sorted(timings_ms)
    median_ms = sorted_timings[len(sorted_timings) // 2] if sorted_timings else 0.0
    percentile_90 = sorted_timings[int(0.9 * len(sorted_timings))] if sorted_timings else 0.0
    percentile_99 = sorted_timings[int(0.99 * len(sorted_timings))] if sorted_timings else 0.0
    avg = mean(timings_ms) if timings_ms else 0.0
    throughput = 0.0
    if avg > 0:
        throughput = batch_size / (avg / 1000.0)
    return {
        "mean_ms": avg,
        "median_ms": median_ms,
        "p90_ms": percentile_90,
        "p99_ms": percentile_99,
        "throughput_fps": throughput,
    }


def summarize_results(
    args: argparse.Namespace,
    trainable_params: int,
    total_params: int,
    param_size: float,
    flops: int,
    flop_table: str,
    latency: Dict[str, float],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "config_file": args.config_file,
        "device": args.device,
        "input_shape": [args.batch_size, args.channels, args.height, args.width],
        "parameters": {
            "trainable": trainable_params,
            "total": total_params,
            "size_mb": round(param_size, 2),
        },
        "latency_ms": latency,
    }
    if flops > 0:
        summary["flops"] = flops
    if flop_table:
        summary["flop_table"] = flop_table
    return summary


def main():
    args = parse_args()
    args = merge_config(args)
    model = build_model(args)
    inputs = get_synthetic_input(args)
    trainable_params, total_params = count_parameters(model)
    param_size = parameter_size_mb(model)
    flop_table = ""
    flops = 0
    if not args.skip_flops:
        flops, flop_table = compute_flops(model, inputs)
    latency = measure_latency(
        model, inputs, args.warmup_iters, args.measure_iters, args.use_amp, args.batch_size
    )
    summary = summarize_results(
        args, trainable_params, total_params, param_size, flops, flop_table, latency
    )
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Profiling summary saved to {output_path}")


if __name__ == "__main__":
    main()
