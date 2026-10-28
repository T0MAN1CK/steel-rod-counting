"""Command-line experiment controls and argument validation."""

import argparse
import random

import numpy as np
from pathlib import Path

import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared/training_v1"))
    parser.add_argument(
        "--heatmaps",
        dest="heatmap_dir",
        type=Path,
        help="Alternate target root containing train/, validation/ and summary.json",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/resnet50_baseline"))
    parser.add_argument("--validation-scenes", type=Path,
                        help="Optional reconstructed-data root; select checkpoints on complete validation scenes")
    parser.add_argument(
        "--encoder", default="resnet50", choices=["resnet18", "resnet34", "resnet50"]
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Offline wiring checks only for baseline",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="One batch: CPU forward, CUDA forward/backward; no optimizer step",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Start at 0 on Windows; increase after checking",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-alpha", type=float, default=10)
    parser.add_argument("--peak-threshold", type=float, default=0.3)
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--match-tolerance", type=float, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--stretch",
        type=float,
        default=0,
        help="Optional paired axis compression, e.g. 0.15; 0 preserves the baseline",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--zoom", type=float, default=0,
                        help="Uniform zoom +/- amount on 50%% of training patches; try 0.2")
    parser.add_argument(
        "--preview-count",
        type=int,
        default=4,
        help="Fixed validation examples saved each epoch",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.stretch <= 0.25:
        parser.error("Stretch must be between 0 and 0.25.")
    if args.no_augmentation and args.stretch:
        parser.error("Stretch requires augmentation; remove --no-augmentation.")
    if not 0 <= args.zoom <= 0.25:
        parser.error("Zoom must be between 0 and 0.25.")
    if args.zoom and (args.no_augmentation or args.stretch):
        parser.error("Zoom requires augmentation and --stretch 0 for this experiment.")
    if args.heatmap_dir is None:
        args.heatmap_dir = args.data / "heatmaps"
    if args.preview_count < 0:
        parser.error("Preview count must be nonnegative.")
    if min(args.batch_size, args.epochs, args.patience) < 1 or args.workers < 0:
        parser.error(
            "Batch size, epochs and patience must be positive; workers nonnegative."
        )
    if args.lr <= 0 or min(args.loss_alpha, args.weight_decay, args.min_delta) < 0:
        parser.error(
            "Learning rate must be positive; alpha, weight decay and min delta nonnegative."
        )
    if (
        not 0 < args.peak_threshold <= 1
        or args.min_distance < 1
        or args.match_tolerance <= 0
    ):
        parser.error("Invalid peak threshold, distance or matching tolerance.")
    if not args.check_only and args.device != "cuda":
        parser.error(
            "Training is reserved for the GPU PC; CPU is allowed only with --check-only."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA unavailable. For CPU wiring checks use --device cpu --check-only."
        )
    return args


def seed_everything(seed):
    """Keep the original RNG initialization and CUDA determinism settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
