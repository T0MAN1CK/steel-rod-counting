"""Build train/validation loaders from the existing source-grouped split."""

import random
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import RodDataset


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def validation_loader(data, heatmaps, batch_size, workers, *, scenes_only=False, generator=None):
    dataset = RodDataset(data / "train_data", data / "splits/validation_labels.json",
                         heatmaps / "validation")
    if scenes_only:
        dataset.records = [r for r in dataset.records if r.get("source_patches")]
        if not dataset.records:
            raise ValueError("No complete reconstructed validation scenes found.")
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                      worker_init_fn=seed_worker, generator=generator)


def prepare_loaders(args):
    train_set = RodDataset(
        args.data / "train_data",
        args.data / "splits/train_labels.json",
        args.heatmap_dir / "train",
        training=not args.no_augmentation,
        stretch=args.stretch,
        zoom=args.zoom,
    )
    val_loader = validation_loader(args.data, args.heatmap_dir, args.batch_size, args.workers)
    val_set = val_loader.dataset
    if args.batch_size != 1 and any(r.get("source_patches") for r in train_set.records):
        raise ValueError("Mixed rectangular scenes/patches require --batch-size 1.")
    train_photos = {
        Path(record["filename"]).stem.rsplit("_", 1)[0] for record in train_set.records
    }
    val_photos = {
        Path(record["filename"]).stem.rsplit("_", 1)[0] for record in val_set.records
    }
    if train_photos & val_photos:
        raise ValueError("Source photo overlap in train/validation splits.")
    if args.validation_scenes:
        scenes = validation_loader(args.validation_scenes, args.validation_scenes / "heatmaps",
                                   1, args.workers, scenes_only=True)
        scene_photos = {Path(r["filename"]).stem.rsplit("_", 1)[0] for r in scenes.dataset.records}
        original_names = {r["filename"] for r in val_set.records}
        if not scene_photos <= val_photos or any(
            not set(r["source_patches"]) <= original_names for r in scenes.dataset.records
        ):
            raise ValueError("Full scenes must come only from the original validation split.")
        train_summary = json.loads((args.heatmap_dir / "summary.json").read_text(encoding="utf-8"))
        scene_summary = json.loads((args.validation_scenes / "heatmaps/summary.json").read_text(encoding="utf-8"))
        if train_summary["sigma"] != scene_summary["sigma"]:
            raise ValueError("Training and full-scene validation targets must use the same sigma.")
        val_loader = scenes

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_loader, val_loader
