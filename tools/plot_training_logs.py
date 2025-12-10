#!/usr/bin/env python3
"""CLI helper to visualize training curves from log files."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from util.plot_utils import plot_logs  # noqa: E402


def _parse_fields(fields: str):
    return [f.strip() for f in fields.split(",") if f.strip()]


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from JSON log files.")
    parser.add_argument("logs", nargs="+", help="One or more log directories containing log.txt.")
    parser.add_argument(
        "--fields",
        default="class_error,loss_bbox_unscaled,mAP",
        help="Comma-separated list of fields to plot.",
    )
    parser.add_argument(
        "--ewm-col",
        type=float,
        default=0,
        help="Smoothing factor passed to pandas.DataFrame.ewm(com=?).",
    )
    parser.add_argument("--log-name", default="log.txt", help="Log filename inside each directory.")
    parser.add_argument("--output", help="Optional path to save the generated figure.")
    parser.add_argument("--dpi", type=int, default=150, help="Resolution for the saved figure.")

    args = parser.parse_args()

    log_paths = [Path(p) for p in args.logs]
    fields = _parse_fields(args.fields)
    fig, _ = plot_logs(log_paths, fields=fields, ewm_col=args.ewm_col, log_name=args.log_name)
    fig.tight_layout()
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=args.dpi)
        print(f"saved plot to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
