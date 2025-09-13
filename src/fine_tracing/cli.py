"""Command line interface for Fine Tracing.

Example usage:
    fine-tracing run --frames-dir path/to/frames --masks-dir path/to/masks --output-dir results/
    fine-tracing metrics --ground-truth drawn-cracks --generated results/.../pure-lines-out
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

from . import config as cfg
from . import main as legacy_main
from . import metrics as metrics_mod


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def cmd_run(args: argparse.Namespace) -> int:
    # mutate config dynamically (temporary approach until refactor)
    if args.frames_dir:
        cfg.frames_dir = args.frames_dir
    if args.masks_dir:
        cfg.masks_dir = args.masks_dir
    if args.output_dir:
        cfg.frames_output_dir = args.output_dir
    # recompute derived paths
    cfg.crack_length_file = f"{cfg.frames_output_dir}/crack_length.csv"
    cfg.gen_frames_dir = f"{cfg.frames_output_dir}/pure-lines-out"
    cfg.metrics_output_dir = f"{cfg.frames_output_dir}/metrics"
    cfg.output_json_file = f"{cfg.metrics_output_dir}/_metrics.json"
    cfg.output_csv_file = f"{cfg.metrics_output_dir}/_metrics.csv"
    cfg.output_overlay_dir = f"{cfg.metrics_output_dir}/overlay"

    legacy_main.main()
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    if args.ground_truth:
        cfg.ground_frames_dir = args.ground_truth
    if args.generated:
        cfg.gen_frames_dir = args.generated
    if args.output_dir:
        cfg.metrics_output_dir = args.output_dir
        cfg.output_json_file = f"{cfg.metrics_output_dir}/_metrics.json"
        cfg.output_csv_file = f"{cfg.metrics_output_dir}/_metrics.csv"
        cfg.output_overlay_dir = f"{cfg.metrics_output_dir}/overlay"
    metrics_mod.main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fine-tracing", description="Fine crack tracing toolkit")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run crack tracing on frames + masks")
    run_p.add_argument("--frames-dir", required=True, help="Directory containing raw image frames")
    run_p.add_argument("--masks-dir", required=True, help="Directory containing segmentation masks")
    run_p.add_argument("--output-dir", required=True, help="Directory for algorithm outputs")
    run_p.set_defaults(func=cmd_run)

    metrics_p = sub.add_parser("metrics", help="Compute metrics comparing generated vs ground truth")
    metrics_p.add_argument("--ground-truth", required=True, help="Directory of ground truth crack drawings")
    metrics_p.add_argument("--generated", required=True, help="Directory of generated crack lines (pure-lines-out)")
    metrics_p.add_argument("--output-dir", required=False, help="Metrics output directory (optional)")
    metrics_p.set_defaults(func=cmd_metrics)

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
