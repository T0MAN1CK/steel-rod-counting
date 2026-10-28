"""Compare each run's best validation-count epoch; reads no images or weights."""

import argparse
import csv
import json
import sys
from pathlib import Path

CONTROL_SETTINGS = (
    "encoder",
    "no_pretrained",
    "batch_size",
    "workers",
    "epochs",
    "patience",
    "min_delta",
    "lr",
    "weight_decay",
    "loss_alpha",
    "peak_threshold",
    "min_distance",
    "match_tolerance",
    "seed",
    "no_augmentation",
    "stretch",
    "zoom",
    "no_amp",
    "split_sha256",
    "validation_scenes",
    "validation_scene_split_sha256",
)


def summarize_run(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config.setdefault("stretch", 0)  # Runs predating optional stretching used none.
    config.setdefault("zoom", 0)
    metrics = path / "metrics.jsonl"
    rows = (
        [
            json.loads(line)
            for line in metrics.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if metrics.exists()
        else []
    )
    best = min(rows, key=lambda row: row["val/count_mae"]) if rows else {}
    return {
        "run": path.as_posix(),
        "sigma": config["heatmaps"]["sigma"],
        "threshold": config["peak_threshold"],
        "distance": config["min_distance"],
        "seed": config["seed"],
        "epochs_logged": len(rows),
        "best_epoch": best.get("epoch", ""),
        "count_mae": best.get("val/count_mae", ""),
        "count_rmse": best.get("val/count_rmse", ""),
        "count_bias": best.get("val/count_bias", ""),
        "precision": best.get("val/precision", ""),
        "recall": best.get("val/recall", ""),
        "f1": best.get("val/f1", ""),
    }, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", type=Path, nargs="+", help="Run directories with config.json"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional new CSV under runs/ or artifacts/"
    )
    args = parser.parse_args()
    summaries, configs = zip(*(summarize_run(path) for path in args.runs))
    different = [
        key
        for key in CONTROL_SETTINGS
        if any(config.get(key) != configs[0].get(key) for config in configs[1:])
    ]
    if different:
        print(
            "Other settings differ; this is not a sigma-only comparison: "
            + ", ".join(different),
            file=sys.stderr,
        )
    else:
        print(
            "Controlled settings and split checksums match. Compare count/localization metrics; "
            "heatmap loss differs with target width.",
            file=sys.stderr,
        )
    writer = csv.DictWriter(
        sys.stdout, fieldnames=list(summaries[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(summaries)
    if args.output:
        if args.output.suffix.lower() != ".csv":
            parser.error("Comparison output must be a CSV file.")
        target = args.output.resolve()
        if not any(
            target.is_relative_to(Path(folder).resolve())
            for folder in ("runs", "artifacts")
        ):
            parser.error("Save comparison output under ignored runs/ or artifacts/.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == "__main__":
    main()
