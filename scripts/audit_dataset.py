"""Read-only local archive audit. Requires Pillow; no torch or training.

Run: python3 scripts/audit_dataset.py --data-root data/raw/segmentation_dataset
Outputs go to ignored artifacts/dataset_audit by default.
"""

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from PIL import Image, ImageDraw


def audit(root, output):
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    examples = []
    for split in ("train", "test"):
        folder = root / f"{split}_data"
        labels = json.loads((folder / f"{split}_labels.json").read_text())
        images = {
            p.name: p
            for p in folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        }
        names = [r["filename"] for r in labels.values()]
        stats = {
            "records": len(labels),
            "images": len(images),
            "missing_images": sorted(set(names) - images.keys()),
            "unlabelled_images": sorted(images.keys() - set(names)),
            "duplicate_label_names": [
                n for n, c in collections.Counter(names).items() if c > 1
            ],
            "dimensions": collections.Counter(),
            "invalid_points": [],
            "decode_errors": [],
            "duplicate_point_count": 0,
            "point_count": 0,
            "empty_records": 0,
        }
        hashes = collections.defaultdict(list)
        candidates = []
        for record in labels.values():
            name = record["filename"]
            if name not in images:
                continue
            path = images[name]
            try:
                with Image.open(path) as im:
                    im.load()
                    w, h = im.size
                stats["dimensions"][f"{w}x{h}"] += 1
            except Exception as exc:
                stats["decode_errors"].append([name, str(exc)])
                continue
            hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(name)
            regions = record.get("regions", {})
            regions = list(regions.values()) if isinstance(regions, dict) else regions
            points = []
            for region in regions:
                a = region.get("shape_attributes", {})
                x, y = a.get("cx"), a.get("cy")
                valid = (
                    a.get("name") == "point"
                    and isinstance(x, (int, float))
                    and isinstance(y, (int, float))
                    and math.isfinite(x)
                    and math.isfinite(y)
                    and 0 <= x < w
                    and 0 <= y < h
                )
                if not valid:
                    stats["invalid_points"].append([name, a])
                else:
                    points.append((x, y))
            stats["point_count"] += len(regions)
            stats["empty_records"] += not regions
            stats["duplicate_point_count"] += len(points) - len(set(points))
            candidates.append((len(points), name, points))
        stats["dimensions"] = dict(stats["dimensions"])
        stats["exact_duplicate_image_groups"] = [
            v for v in hashes.values() if len(v) > 1
        ]
        report[split] = stats
        nonempty = sorted(c for c in candidates if c[0] > 0)
        if nonempty:
            examples.append((split, folder, nonempty[len(nonempty) // 2]))
    rows = list(csv.DictReader((root / "train_folds.csv").open()))
    groups = collections.defaultdict(set)
    for row in rows:
        groups[re.sub(r"_\d+$", "", Path(row["filename"]).stem)].add(row["fold"])
    train_names = [
        r["filename"]
        for r in json.loads(
            (root / "train_data/train_labels.json").read_text()
        ).values()
    ]
    expected_masks = {Path(n).stem + ".pt" for n in train_names}
    actual_masks = {p.name for p in (root / "precomputed_masks").glob("*.pt")}
    report["mask_files"] = {
        "missing": sorted(expected_masks - actual_masks),
        "extra": sorted(actual_masks - expected_masks),
        "contents_checked": False,
    }
    report["saved_folds"] = {
        "rows": len(rows),
        "filename_groups": len(groups),
        "groups_in_multiple_folds": sum(len(v) > 1 for v in groups.values()),
        "csv_names_match_labels": set(r["filename"] for r in rows) == set(train_names),
    }
    for split, folder, (count, name, points) in examples:
        with Image.open(folder / name) as im:
            im = im.convert("RGB")
            scale = min(1, 1000 / max(im.size))
            im = im.resize((round(im.width * scale), round(im.height * scale)))
            draw = ImageDraw.Draw(im)
            for x, y in points:
                x, y = x * scale, y * scale
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="lime", width=1)
                draw.line((x - 2, y, x + 2, y), fill="red", width=1)
            im.save(output / f"{split}_overlay.png")
        report[split]["overlay"] = {
            "filename": name,
            "points": count,
            "display_scale": scale,
        }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print("Audit saved to", output / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset_audit"))
    args = parser.parse_args()
    audit(args.data_root, args.output)
