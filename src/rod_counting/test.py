"""Evaluate saved weights on validation patches or reconstructed full scenes."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch
from .dataloader import validation_loader
from .model import load_model
from .reporting import select_previews
from .validation import validate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--data", type=Path, help="Override the saved prepared-data root"
    )
    parser.add_argument("--heatmaps", type=Path, help="Override the saved target root")
    parser.add_argument("--validation-scenes", type=Path,
                        help="Evaluate only complete scenes from this prepared root")
    parser.add_argument(
        "--peak-threshold", type=float, help="Default: checkpoint setting"
    )
    parser.add_argument("--batch-size", type=int, help="Default: checkpoint setting")
    parser.add_argument(
        "--output", type=Path, help="New directory for metrics and preview PNGs"
    )
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    settings = SimpleNamespace(**checkpoint["config"])
    scene_root = args.validation_scenes or getattr(settings, "validation_scenes", None)
    data = args.data or Path(scene_root or settings.data)
    heatmaps = args.heatmaps or (data / "heatmaps" if scene_root else Path(settings.heatmap_dir))
    if args.peak_threshold is not None:
        settings.peak_threshold = args.peak_threshold
    batch_size = args.batch_size if args.batch_size is not None else (1 if scene_root else settings.batch_size)
    if batch_size < 1 or not 0 < settings.peak_threshold <= 1:
        parser.error("Batch size must be positive and threshold in (0,1].")

    loader = validation_loader(data, heatmaps, batch_size, settings.workers,
                               scenes_only=bool(scene_root))
    dataset = loader.dataset
    preview_indices = (
        select_previews(dataset, settings.preview_count) if args.output else ()
    )
    metrics, previews = validate(model, loader, device, settings, preview_indices)
    result = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint["epoch"],
        "peak_threshold": settings.peak_threshold,
        "heatmaps": str(heatmaps),
        "split": "validation",
        **metrics,
    }
    if scene_root:
        result.update(validation_mode="full_scenes", samples=len(dataset), data=str(data))
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        for index, preview in enumerate(previews, 1):
            Image.fromarray(preview["image"]).save(
                args.output / f"sample_{index:02d}.png"
            )


if __name__ == "__main__":
    main()
