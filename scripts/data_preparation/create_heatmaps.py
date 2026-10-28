"""Generate float32 Gaussian centre targets and one preview per split.

Requires the installed project and Pillow. Images and point labels remain unchanged.
"""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from rod_counting.heatmaps import centre_heatmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sigma", type=float, default=2.0, help="Gaussian width in image pixels"
    )
    args = parser.parse_args()
    if not math.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("Sigma must be finite and positive.")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory.")
    args.output.mkdir(parents=True)
    summary = {
        "sigma": args.sigma,
        "combination": "maximum",
        "format": "float32 TIFF",
        "support": "square neighbourhood extending approximately 3 sigma",
        "splits": {},
    }
    for split, label_file in [
        ("train", "train_labels.json"),
        ("validation", "validation_labels.json"),
    ]:
        labels = json.loads((args.splits / label_file).read_text(encoding="utf-8"))
        target_dir = args.output / split
        target_dir.mkdir()
        # Choose one nonempty example with median point count, for two previews total.
        nonempty = sorted(
            (len(r["regions"]), r["filename"]) for r in labels.values() if r["regions"]
        )
        preview_name = nonempty[len(nonempty) // 2][1] if nonempty else None
        point_count = 0
        for record in labels.values():
            filename = record["filename"]
            if Path(filename).name != filename or "/" in filename or "\\" in filename:
                raise ValueError(f"Expected a plain filename: {filename}")
            points = [
                (r["shape_attributes"]["cx"], r["shape_attributes"]["cy"])
                for r in record["regions"].values()
            ]
            with Image.open(args.train_dir / filename) as source:
                heatmap = centre_heatmap(source.size, points, args.sigma)
                heatmap.save(
                    target_dir / (filename + ".tiff"), compression="tiff_adobe_deflate"
                )
                if filename == preview_name:
                    rgb = source.convert("RGB")
                    gray = heatmap.point(lambda value: value * 255).convert("L")
                    red = Image.new("RGB", source.size, "red")
                    overlay = Image.composite(
                        red, rgb, gray.point(lambda v: round(v * 0.65))
                    )
                    draw = ImageDraw.Draw(overlay)
                    for x, y in points:
                        draw.ellipse((x - 1, y - 1, x + 1, y + 1), outline="lime")
                    preview = Image.new("RGB", (rgb.width * 3, rgb.height))
                    preview.paste(rgb, (0, 0))
                    preview.paste(gray.convert("RGB"), (rgb.width, 0))
                    preview.paste(overlay, (rgb.width * 2, 0))
                    preview.save(args.output / f"{split}_preview.png")
            point_count += len(points)
        summary["splits"][split] = {
            "images": len(labels),
            "points": point_count,
            "preview": preview_name,
        }
        print(f"{split}: {len(labels)} heatmaps from {point_count} annotated points")
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
