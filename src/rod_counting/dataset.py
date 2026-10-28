"""Load one RGB patch, its continuous heatmap and its annotation index."""

import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .augmentation import augment_pair, load_pair, zoom_crop
from .heatmaps import centre_heatmap


class RodDataset(Dataset):
    def __init__(self, image_dir, label_file, heatmap_dir, training=False, stretch=0, zoom=0):
        self.image_dir = Path(image_dir)
        self.heatmap_dir = Path(heatmap_dir)
        self.training = training
        self.stretch = stretch
        self.zoom = zoom
        if not 0 <= zoom <= 0.25:
            raise ValueError("Zoom must be between 0 and 0.25.")
        if training and zoom:
            self.sigma = float(json.loads(
                (self.heatmap_dir.parent / "summary.json").read_text(encoding="utf-8")
            )["sigma"])
            if not math.isfinite(self.sigma) or self.sigma <= 0:
                raise ValueError("Target summary must contain a finite positive sigma.")
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

        labels = json.loads(Path(label_file).read_text(encoding="utf-8"))
        self.records = sorted(labels.values(), key=lambda record: record["filename"])
        if not self.records:
            raise ValueError("Split has no records.")
        for record in self.records:
            name = record["filename"]
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError(f"Expected plain filename: {name}")
            image_path = self.image_dir / name
            target_path = self.heatmap_dir / (name + ".tiff")
            if not image_path.is_file() or not target_path.is_file():
                raise ValueError(f"Missing image or target: {name}")
            if training and zoom:
                with Image.open(image_path) as image:
                    if image.size != (256, 256):
                        raise ValueError("Zoom experiment requires original 256x256 patches.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        name = self.records[index]["filename"]
        image, target = load_pair(
            self.image_dir / name,
            self.heatmap_dir / (name + ".tiff"),
            training=self.training and not self.zoom,
            stretch=self.stretch,
        )
        if self.training and self.zoom:
            if random.random() < 0.5:
                points = [(r["shape_attributes"]["cx"], r["shape_attributes"]["cy"])
                          for r in self.records[index]["regions"].values()]
                image, points = zoom_crop(image, points, self.zoom, random)
                target = centre_heatmap(image.size, points, self.sigma)
            image, target = augment_pair(image, target, stretch=self.stretch)
        image = (
            torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255
        )
        target = torch.from_numpy(np.array(target, dtype=np.float32))[None]
        if not torch.isfinite(target).all() or target.min() < 0 or target.max() > 1:
            raise ValueError(f"Target outside [0,1]: {name}")
        image = (image - self.mean) / self.std
        return image, target, index

    def points(self, index):
        """Original annotated centres, usable only with unaugmented samples."""
        if self.training:
            raise ValueError(
                "Original points cannot be used with augmented training images."
            )
        return [
            (region["shape_attributes"]["cx"], region["shape_attributes"]["cy"])
            for region in self.records[index]["regions"].values()
        ]
