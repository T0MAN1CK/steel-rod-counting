"""Copy raw training data into a prepared folder, excluding black padding.

Requires Pillow. Raw images and labels are never changed.
"""

import argparse
import json
from pathlib import Path
import shutil

from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True, help="Raw train_data folder")
parser.add_argument(
    "--output", type=Path, required=True, help="New prepared train_data folder"
)
args = parser.parse_args()

source = args.source.resolve()
output = args.output.resolve()
if output == source or output.is_relative_to(source) or source.is_relative_to(output):
    parser.error(
        "Raw and prepared folders must be separate and must not contain each other."
    )
if output.exists():
    parser.error("Output already exists. Choose a new prepared folder.")

labels = json.loads((source / "train_labels.json").read_text(encoding="utf-8"))
prepared_labels = {}
removed_count = 0
output.mkdir(parents=True)

for key, record in labels.items():
    filename = record["filename"]
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError(f"Expected a plain image filename: {filename}")
    image_path = source / filename
    is_padding = image_path.stem.endswith(("_12", "_13", "_14", "_15"))
    is_black = False
    if is_padding:
        with Image.open(image_path) as image:
            is_black = all(
                low == high == 0 for low, high in image.convert("RGB").getextrema()
            )

    if is_padding and is_black and not record.get("regions", {}):
        removed_count += 1
        continue  # Exclude from prepared output; leave the raw image untouched.

    shutil.copy2(image_path, output / filename)
    prepared_labels[key] = record

(output / "train_labels.json").write_text(
    json.dumps(prepared_labels, indent=2), encoding="utf-8"
)
print(
    f"Copied {len(prepared_labels)} images and matching label records; "
    f"excluded {removed_count} black padding patches. Raw data unchanged."
)
