"""Split prepared patch labels by source photo; never move or modify images.

Output labels map edge coordinate 256 to 255 only on a 256-pixel dimension.
Source labels are unchanged. Requires Pillow; use --train-dir to locate images.
"""

import argparse
import json
from pathlib import Path
import random

from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--train-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True, help="New split directory")
parser.add_argument("--validation-fraction", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
if not 0 < args.validation_fraction < 1:
    parser.error("Validation fraction must be between 0 and 1.")
if args.output.exists():
    parser.error("Output already exists; choose a new directory.")

labels = json.loads((args.train_dir / "train_labels.json").read_text(encoding="utf-8"))
photo_for_key = {}
adjustments = []
for key, record in labels.items():
    filename = record["filename"]
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError(f"Expected a plain image filename: {filename}")
    photo, separator, patch = Path(filename).stem.rpartition("_")
    if not separator or not photo or not patch.isdigit():
        raise ValueError(f"Cannot identify source photo and patch: {filename}")
    if not (args.train_dir / filename).is_file():
        raise ValueError(f"Missing image: {filename}")
    with Image.open(args.train_dir / filename) as image:
        width, height = image.size
    for region_id, region in record.get("regions", {}).items():
        shape = region["shape_attributes"]
        for axis, size in [("cx", width), ("cy", height)]:
            value = shape[axis]
            if size == 256 and value == 256:
                shape[axis] = 255
                adjustments.append(
                    {
                        "filename": filename,
                        "region_id": region_id,
                        "axis": axis,
                        "before": 256,
                        "after": 255,
                    }
                )
            if not 0 <= shape[axis] < size:
                raise ValueError(
                    f"Invalid {axis}={shape[axis]} in {filename} ({width}x{height})"
                )
    photo_for_key[key] = photo

photos = sorted(set(photo_for_key.values()))
if len(photos) < 2:
    parser.error("At least two source photos are required.")
random.Random(args.seed).shuffle(photos)
validation_count = max(
    1, min(len(photos) - 1, round(len(photos) * args.validation_fraction))
)
validation_photos = set(photos[:validation_count])

training = {
    key: record
    for key, record in labels.items()
    if photo_for_key[key] not in validation_photos
}
validation = {
    key: record
    for key, record in labels.items()
    if photo_for_key[key] in validation_photos
}
summary = {
    "seed": args.seed,
    "validation_fraction_requested": args.validation_fraction,
    "grouping": "filename stem with final underscore and patch index removed",
    "boundary_adjustments": adjustments,
    "adjusted_coordinates": len(adjustments),
}
for name, records in [("training", training), ("validation", validation)]:
    summary[name] = {
        "photo_ids": sorted({photo_for_key[key] for key in records}),
        "images": len(records),
        "points": sum(len(r.get("regions", {})) for r in records.values()),
        "empty_images": sum(not r.get("regions", {}) for r in records.values()),
    }

args.output.mkdir(parents=True)
for filename, content in [
    ("train_labels.json", training),
    ("validation_labels.json", validation),
    ("split_summary.json", summary),
]:
    (args.output / filename).write_text(json.dumps(content, indent=2), encoding="utf-8")
for name in ("training", "validation"):
    row = summary[name]
    print(
        f'{name}: {len(row["photo_ids"])} photos, {row["images"]} patches, '
        f'{row["points"]} points, {row["empty_images"]} empty patches'
    )

print(f"Adjusted {len(adjustments)} boundary coordinates in output labels only.")
