"""Check reconstruction geometry, label offsets and the frozen photo split."""

import copy
import json
from pathlib import Path
import random
import runpy
import tempfile
import unittest

import numpy as np
from PIL import Image

from rod_counting.augmentation import augment_pair


SCRIPT = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    "scripts/data_preparation/reconstruct_scenes.py"))


class ScenePreparationTests(unittest.TestCase):
    def test_pixels_offsets_incomplete_retention_and_split_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            (source / "train_data").mkdir(parents=True)
            (source / "splits").mkdir()
            for split, photo in [("train", "Photo (1)"), ("validation", "Photo (2)")]:
                labels = {}
                for i in range(12):
                    name = f"{photo}_{i}.png"
                    Image.new("RGB", (256, 256), (i, 30, 50)).save(source / "train_data" / name)
                    labels[name] = {"filename": name, "regions": {
                        "point": {"shape_attributes": {"name": "point", "cx": 255, "cy": 2}}
                    }}
                if split == "train":
                    labels["incomplete"] = {"filename": "Photo (3)_0.png", "regions": {}}
                    Image.new("RGB", (256, 256)).save(source / "train_data/Photo (3)_0.png")
                (source / "splits" / f"{split}_labels.json").write_text(json.dumps(labels))
            patches = {int(Path(r["filename"]).stem.rsplit("_", 1)[1]): r for r in
                       json.loads((source / "splits/validation_labels.json").read_text()).values()}
            original = copy.deepcopy(patches)
            image, regions = SCRIPT["reconstruct_photo"](patches, source / "train_data")
            self.assertEqual(image.size, (1024, 768))
            self.assertEqual(patches, original)
            for i in range(12):
                dx, dy = i % 4 * 256, i // 4 * 256
                self.assertEqual(image.getpixel((dx, dy)), (i, 30, 50))
                self.assertEqual(regions[f"{i}_point"]["shape_attributes"]["cx"], dx + 255)
                self.assertEqual(regions[f"{i}_point"]["shape_attributes"]["cy"], dy + 2)
            with self.assertRaisesRegex(ValueError, "every patch"):
                SCRIPT["reconstruct_photo"]({0: patches[0]}, source / "train_data")
            output = Path(temp) / "scenes"
            summary = SCRIPT["prepare_scenes"](source, output)
            self.assertEqual(summary["splits"]["train"]["photo_ids"], ["Photo (1)", "Photo (3)"])
            self.assertEqual(summary["splits"]["train"]["individual_patches"], 1)
            self.assertTrue((output / "train_data/Photo (3)_0.png").is_file())
            self.assertEqual(summary["splits"]["validation"]["points"], 12)
            self.assertEqual(summary["incomplete_photo_missing_indices"]["train"]["Photo (3)"], list(range(1, 12)))
            with self.assertRaises(FileExistsError):
                SCRIPT["prepare_scenes"](source, output)

    def test_rectangular_rotation_keeps_rgb_heatmap_alignment(self):
        values = np.zeros((64, 96), dtype=np.float32)
        values[12, 20] = 1
        pixels = np.zeros((64, 96, 3), dtype=np.uint8)
        pixels[12, 20] = 255
        for seed in range(8):
            image, target = augment_pair(Image.fromarray(pixels), Image.fromarray(values), random.Random(seed))
            self.assertEqual(image.size, target.size)
            self.assertIn(image.size, [(96, 64), (64, 96)])
            y, x = np.argwhere(np.asarray(target) == 1)[0]
            self.assertGreater(np.asarray(image)[y, x].min(), 200)


if __name__ == "__main__":
    unittest.main()
