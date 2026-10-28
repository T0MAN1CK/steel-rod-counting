"""Count rods in complete test photos using a saved checkpoint; no training."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .metrics import find_peaks
from .model import load_model


def predict_heatmap(model, image, device):
    """One full-photo forward pass, padded to a multiple of 32 then unpadded."""
    pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255
    height, width = pixels.shape[:2]
    inputs = torch.from_numpy(pixels).permute(2, 0, 1)[None].to(device)
    mean = inputs.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = inputs.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    inputs = torch.nn.functional.pad(
        (inputs - mean) / std, (0, -width % 32, 0, -height % 32), mode="replicate"
    )
    model.eval()
    with torch.inference_mode():
        heatmap = torch.sigmoid(model(inputs))[0, 0, :height, :width].cpu().numpy()
    if not np.isfinite(heatmap).all():
        raise ValueError("Nonfinite predicted heatmap.")
    return heatmap


def prediction_preview(image, heatmap, points, record, original_size):
    """RGB, fixed-scale heatmap and centre overlay; green predictions, yellow labels."""
    width = min(image.width, 1024)
    size = (width, max(1, round(image.height * width / image.width)))
    photo = image.resize(size)
    scores = Image.fromarray((heatmap.clip(0, 1) * 255).round().astype(np.uint8))
    scores = scores.resize(size).convert("RGB")
    overlay = photo.copy()
    draw = ImageDraw.Draw(overlay)
    for region in (record["regions"].values() if record is not None else ()):
        point = region["shape_attributes"]
        x = (point["cx"] + 0.5) * width / original_size[0] - 0.5
        y = (point["cy"] + 0.5) * size[1] / original_size[1] - 0.5
        draw.line((x-2, y, x+2, y), fill="yellow")
        draw.line((x, y-2, x, y+2), fill="yellow")
    for x, y in points:
        x = (x + 0.5) * width / image.width - 0.5
        y = (y + 0.5) * size[1] / image.height - 0.5
        draw.ellipse((x-3, y-3, x+3, y+3), outline="lime")
    sheet = Image.new("RGB", (width * 3, size[1] + 24))
    caption = (f"Pred {len(points)} (green) / annotated {len(record['regions'])} (yellow)"
               if record is not None else f"Pred {len(points)} (green); no reference labels")
    for column, (title, panel) in enumerate([
        ("RGB", photo), ("Predicted heatmap", scores),
        (caption, overlay),
    ]):
        sheet.paste(panel, (column * width, 24))
        ImageDraw.Draw(sheet).text((column * width + 3, 5), title, fill="white")
    return sheet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/raw/segmentation_dataset/test_data"))
    parser.add_argument("--labels", type=Path, help="Optional annotations; defaults to DATA/test_labels.json if present")
    parser.add_argument("--size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Optional whole-photo resize; omit for native resolution")
    parser.add_argument("--peak-threshold", type=float, default=0.65)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--save-previews", action="store_true",
                        help="Save RGB/heatmap/centre-overlay PNGs")
    parser.add_argument("--output", type=Path, required=True, help="New directory for CSV/JSON")
    args = parser.parse_args(argv)
    if not 0 < args.peak_threshold <= 1 or (args.size and min(args.size) < 1):
        parser.error("Threshold must be in (0,1]; resize dimensions must be positive.")
    label_path = args.labels or args.data / "test_labels.json"
    if args.labels and not label_path.is_file():
        parser.error(f"Label file does not exist: {label_path}")
    records = json.loads(label_path.read_text(encoding="utf-8")) if label_path.is_file() else {}
    labels = {record["filename"]: record for record in records.values()}
    if len(labels) != len(records):
        parser.error("Duplicate annotation filenames.")
    files = sorted(p for p in args.data.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    if not files:
        parser.error("No supported images found in the input directory.")
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    min_distance = checkpoint["config"]["min_distance"]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, path in enumerate(files, 1):
        with Image.open(path) as source:
            image = source.convert("RGB")
        original_size = image.size
        if args.size:
            image = image.resize(tuple(args.size), Image.Resampling.BILINEAR)
        heatmap = predict_heatmap(model, image, device)
        points = find_peaks(heatmap, args.peak_threshold, min_distance)
        predicted = len(points)
        record = labels.get(path.name)
        annotated = len(record["regions"]) if record is not None else None
        if args.save_previews:
            prediction_preview(image, heatmap, points, record, original_size).save(
                args.output / (path.name + "_prediction.png")
            )
        rows.append({"filename": path.name, "width": image.width, "height": image.height,
                     "predicted_count": predicted, "annotated_count": annotated,
                     "count_error": predicted - annotated if annotated is not None else None})
        reference = (f"annotated {annotated}, error {predicted - annotated:+d}"
                     if annotated is not None else "no reference labels")
        print(f"{index}/{len(files)} {path.name}: predicted {predicted}, "
              f"{reference}", flush=True)
    labeled = [row for row in rows if row["annotated_count"] is not None]
    errors = np.array([row["count_error"] for row in labeled], dtype=float)
    result = {"checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"],
              "data": str(args.data), "labels": str(label_path) if label_path.is_file() else None,
              "resize": args.size,
              "peak_threshold": args.peak_threshold, "min_distance": min_distance,
              "save_previews": args.save_previews,
              "image_count": len(rows), "count_error_unit": "rods per full photo",
              "labeled_image_count": len(labeled), "unlabeled_image_count": len(rows) - len(labeled),
              "total_predicted": sum(row["predicted_count"] for row in rows),
              "total_predicted_labeled": sum(row["predicted_count"] for row in labeled) if labeled else None,
              "total_annotated": sum(row["annotated_count"] for row in labeled) if labeled else None,
              "count_mae": float(np.abs(errors).mean()) if labeled else None,
              "count_rmse": float(np.sqrt(np.mean(errors**2))) if labeled else None,
              "count_bias": float(errors.mean()) if labeled else None}
    with (args.output / "counts.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
