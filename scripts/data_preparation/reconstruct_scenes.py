"""Reassemble complete 4x3 grids; retain incomplete-photo patches in their splits."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image


PATCH_SIZE = 256
COLUMNS, ROWS = 4, 3


def reconstruct_photo(records, image_dir):
    """Paste row-major patches and translate every annotated centre by its offset."""
    if set(records) != set(range(COLUMNS * ROWS)):
        raise ValueError("A scene requires every patch from 0 to 11; no gap filling.")
    image = Image.new("RGB", (COLUMNS * PATCH_SIZE, ROWS * PATCH_SIZE))
    regions = {}
    for index in range(COLUMNS * ROWS):
        record = records[index]
        name = record["filename"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError(f"Expected a plain filename: {name}")
        dx, dy = (index % COLUMNS) * PATCH_SIZE, (index // COLUMNS) * PATCH_SIZE
        with Image.open(image_dir / name) as patch:
            if patch.size != (PATCH_SIZE, PATCH_SIZE):
                raise ValueError(f"Expected a 256x256 patch: {name}")
            image.paste(patch.convert("RGB"), (dx, dy))
        for key, region in record["regions"].items():
            translated = copy.deepcopy(region)
            shape = translated["shape_attributes"]
            if not (0 <= shape["cx"] < PATCH_SIZE and 0 <= shape["cy"] < PATCH_SIZE):
                raise ValueError(f"Invalid prepared centre in {name}")
            shape["cx"] += dx
            shape["cy"] += dy
            regions[f"{index}_{key}"] = translated
    return image, regions


def prepare_scenes(data, output):
    grouped = {}
    source_hashes = {}
    owners = {}
    incomplete = {}
    for split in ("train", "validation"):
        path = data / "splits" / f"{split}_labels.json"
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = json.loads(path.read_text(encoding="utf-8"))
        photos = {}
        for record in records.values():
            photo, index = Path(record["filename"]).stem.rsplit("_", 1)
            if photo in owners and owners[photo] != split:
                raise ValueError(f"Source photo crosses splits: {photo}")
            owners[photo] = split
            patches = photos.setdefault(photo, {})
            index = int(index)
            if index in patches or not 0 <= index < COLUMNS * ROWS:
                raise ValueError(f"Unexpected/duplicate patch index: {record['filename']}")
            patches[index] = record
        grouped[split] = photos
        incomplete[split] = {}
        for photo, patches in sorted(photos.items()):
            missing = sorted(set(range(COLUMNS * ROWS)) - set(patches))
            if missing:
                incomplete[split][photo] = missing
        if not grouped[split]:
            raise ValueError(f"No source photos in {split}.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "train_data").mkdir()
    (output / "splits").mkdir()
    summary = {"source_data": str(data), "source_split_sha256": source_hashes,
               "columns": COLUMNS, "rows": ROWS, "patch_size": PATCH_SIZE,
               "order": "row-major", "scene_size": [1024, 768],
               "incomplete_policy": "retain available individual patches, never fill gaps",
               "incomplete_photo_missing_indices": incomplete, "splits": {}}
    for split, photos in grouped.items():
        labels = {}
        scenes = patches_retained = 0
        for photo, patches in photos.items():
            if photo in incomplete[split]:
                for record in patches.values():
                    name = record["filename"]
                    if Path(name).name != name or "/" in name or "\\" in name:
                        raise ValueError(f"Expected a plain filename: {name}")
                    shutil.copy2(data / "train_data" / name, output / "train_data" / name)
                    labels[name] = copy.deepcopy(record)
                    patches_retained += 1
                continue
            image, regions = reconstruct_photo(patches, data / "train_data")
            name = photo + "_scene.png"
            image.save(output / "train_data" / name)
            labels[name] = {"filename": name, "regions": regions,
                            "source_patches": [patches[i]["filename"] for i in range(12)]}
            scenes += 1
        (output / "splits" / f"{split}_labels.json").write_text(
            json.dumps(labels, indent=2), encoding="utf-8"
        )
        summary["splits"][split] = {"images": len(labels),
            "scenes": scenes, "individual_patches": patches_retained,
            "points": sum(len(r["regions"]) for r in labels.values()),
            "photo_ids": sorted(photos)}
        print(f"{split}: {scenes} scenes + {patches_retained} individual patches; "
              f"{summary['splits'][split]['points']} annotated points")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared/training_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a separate new experiment directory.")
    prepare_scenes(args.data, args.output)


if __name__ == "__main__":
    main()
