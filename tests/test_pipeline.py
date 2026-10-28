"""Small CPU checks; no optimizer steps or model training."""

import json
import csv
import runpy
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from rod_counting.dataset import RodDataset
from rod_counting.metrics import find_peaks, match_points, summarize_counts
from rod_counting.model import build_model
from rod_counting.losses import heatmap_loss
from rod_counting.heatmaps import centre_heatmap


class PipelineTests(unittest.TestCase):
    def test_zoom_regenerates_fixed_sigma_target_and_never_augments_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "targets/train").mkdir(parents=True)
            (root / "targets/summary.json").write_text(json.dumps({"sigma": 3}))
            name = "Photo (1)_0.png"
            pixels = np.zeros((256, 256, 3), dtype=np.uint8)
            pixels[126:131, 126:131] = 255
            Image.fromarray(pixels).save(root / name)
            labels = {"sample": {"filename": name, "regions": {
                str(i): {"shape_attributes": {"cx": x, "cy": y}}
                for i, (x, y) in enumerate([(0, 0), (128, 128)])
            }}}
            (root / "labels.json").write_text(json.dumps(labels))
            saved_target = centre_heatmap((256, 256), [(0, 0), (128, 128)], 3)
            target_path = root / "targets/train" / (name + ".tiff")
            saved_target.save(target_path)
            original = target_path.read_bytes()
            dataset = RodDataset(root, root / "labels.json", root / "targets/train",
                                 training=True, zoom=0.25)
            with patch("rod_counting.dataset.random.random", return_value=0), \
                    patch("rod_counting.dataset.random.uniform", return_value=1.25), \
                    patch("rod_counting.dataset.random.randint", side_effect=[20, 15]), \
                    patch("rod_counting.dataset.augment_pair", side_effect=lambda im, hm, **kw: (im, hm)):
                image, target, _ = dataset[0]
            expected = np.asarray(centre_heatmap((256, 256), [(140.125, 145.125)], 3))
            np.testing.assert_array_equal(target[0].numpy(), expected)
            self.assertGreater(image[:, 145, 140].min().item(), 1)
            self.assertEqual(target_path.read_bytes(), original)
            with patch("rod_counting.dataset.random.random", return_value=0.75), \
                    patch("rod_counting.dataset.augment_pair", side_effect=lambda im, hm, **kw: (im, hm)):
                _, skipped, _ = dataset[0]
            np.testing.assert_array_equal(skipped[0].numpy(), np.asarray(saved_target))
            validation = RodDataset(root, root / "labels.json", root / "targets/train", zoom=0.25)
            with patch("rod_counting.dataset.random.random", side_effect=AssertionError):
                _, untouched, _ = validation[0]
            np.testing.assert_array_equal(untouched[0].numpy(), np.asarray(saved_target))

    def test_comparison_uses_best_count_epoch_and_handles_pending_run(self):
        summarize = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "scripts/compare_runs.py")
        )["summarize_run"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {
                "heatmaps": {"sigma": 3},
                "peak_threshold": 0.3,
                "min_distance": 3,
                "seed": 42,
            }
            (root / "config.json").write_text(json.dumps(config))
            pending, _ = summarize(root)
            self.assertEqual(pending["epochs_logged"], 0)
            self.assertEqual(pending["best_epoch"], "")
            self.assertEqual(summarize(root)[1]["zoom"], 0)
            rows = [
                {"epoch": 1, "val/count_mae": 1.0, "val/f1": 0.9},
                {"epoch": 2, "val/count_mae": 2.0, "val/f1": 0.8},
            ]
            (root / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            summary, _ = summarize(root)
            self.assertEqual(summary["sigma"], 3)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(summary["count_mae"], 1.0)
            self.assertEqual(summary["f1"], 0.9)

    def test_wider_gaussians_preserve_centres_and_empty_targets(self):
        generate = runpy.run_path(
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/data_preparation/create_heatmaps.py"
            )
        )["centre_heatmap"]
        narrow = np.asarray(generate((16, 16), [(8, 8)], 2))
        wide = np.asarray(generate((16, 16), [(8, 8)], 3))
        self.assertEqual(wide[8, 8], 1.0)
        self.assertGreater(wide[8, 11], narrow[8, 11])
        self.assertEqual(find_peaks(wide), [(8, 8)])
        self.assertFalse(np.asarray(generate((16, 16), [], 3)).any())

    def test_local_progress_saves_metrics_previews_and_history(self):
        from rod_counting.reporting import save_progress

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            preview = {
                "image": np.full((8, 8, 3), 51, dtype=np.uint8),
                "caption": "photo <test>; predicted 1, annotated 2",
            }
            row = {
                "epoch": 1,
                "train/loss": 0.4,
                "val/loss": 0.3,
                "val/count_mae": 2.0,
                "val/scene_count_mae": 4.0,
                "val/patch_count_mae": 1.0,
                "val/f1": 0.5,
                "best/count_mae": 2.0,
                "best/epoch": 1,
            }
            save_progress(root, row, [preview])
            row = {
                **row,
                "epoch": 2,
                "val/count_mae": 1.0,
                "best/count_mae": 1.0,
                "best/epoch": 2,
            }
            save_progress(root, row, [preview])
            self.assertEqual(len((root / "metrics.jsonl").read_text().splitlines()), 2)
            with (root / "metrics.csv").open(newline="") as log:
                self.assertEqual([r["epoch"] for r in csv.DictReader(log)], ["1", "2"])
            with Image.open(root / "previews/epoch_002_sample_01.png") as image:
                np.testing.assert_array_equal(np.asarray(image), preview["image"])
            page = (root / "progress.html").read_text()
            self.assertIn("epoch_001.html", page)
            self.assertIn("photo &lt;test&gt;", page)
            self.assertIn("<svg", page)
            self.assertIn("val/scene_count_mae", page)
            self.assertIn("val/patch_count_mae", page)
            self.assertNotIn("https://", page)

    def test_previews_span_density_and_allow_zero(self):
        from rod_counting.reporting import select_previews

        class Samples:
            records = [{"regions": dict.fromkeys(range(n))} for n in [20, 0, 3, 50, 8]]

            def __len__(self):
                return len(self.records)

        self.assertEqual(select_previews(Samples(), 2), {1, 3})
        self.assertEqual(select_previews(Samples(), 0), set())
        self.assertEqual(len(select_previews(Samples(), 20)), 5)

    def test_preview_keeps_fixed_heatmap_scale_and_denormalizes_rgb(self):
        from rod_counting.reporting import validation_preview

        image = torch.tensor([0.485, 0.456, 0.406])[:, None, None].expand(3, 16, 16)
        image = (
            image - torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        ) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        target = np.zeros((16, 16), dtype=np.float32)
        target[8, 8] = 1
        prediction = np.full((16, 16), 0.2, dtype=np.float32)
        preview = validation_preview(
            image, target, prediction, [(8, 8)], [(8, 8)], "sample.png"
        )
        pixels = preview["image"]
        self.assertEqual(pixels.shape, (40, 80, 3))
        np.testing.assert_array_equal(pixels[24, 32], [51, 51, 51])
        np.testing.assert_array_equal(pixels[32, 24], [255, 255, 255])
        np.testing.assert_allclose(pixels[24, 0], [124, 116, 104], atol=1)
        self.assertIn("predicted 1, annotated 1", preview["caption"])

    def test_peak_borders_plateaus_and_suppression(self):
        heatmap = np.zeros((16, 16), dtype=np.float32)
        heatmap[0, 0] = 1
        heatmap[15, 15] = 0.8
        heatmap[7, 7] = 0.9
        heatmap[7, 9] = 0.7
        heatmap[2:4, 10:12] = 0.6
        self.assertEqual(set(find_peaks(heatmap)), {(0, 0), (15, 15), (7, 7), (10, 2)})
        self.assertEqual(find_peaks(np.zeros((16, 16))), [])

    def test_matching_is_one_to_one_and_maximizes_matches(self):
        self.assertEqual(match_points([(0, 0), (0, 1)], [(0, 0)], 2), (1, 1, 0))
        # Greedy nearest assignment would consume the only match of prediction 2.
        self.assertEqual(
            match_points([(1, 0), (0, 0)], [(0, 0), (2, 0)], 1.1), (2, 0, 0)
        )
        self.assertEqual(match_points([], [(2, 2)]), (0, 0, 1))
        self.assertEqual(match_points([(2, 2)], []), (0, 1, 0))

    def test_count_metrics_do_not_hide_cancelled_errors(self):
        metrics = summarize_counts([-2, 2], 2, 1, 1)
        self.assertEqual(metrics["count_mae"], 2)
        self.assertEqual(metrics["count_bias"], 0)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_loader_preserves_float_targets_and_reference_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGB", (32, 32), (255, 0, 0)).save(root / "Photo (2)_0.png")
            target = Image.new("F", (32, 32))
            target.putpixel((0, 0), 1)
            target.putpixel((1, 0), 0.125)
            target.save(root / "Photo (2)_0.png.tiff")
            labels = {
                "a": {
                    "filename": "Photo (2)_0.png",
                    "regions": {"0": {"shape_attributes": {"cx": 0, "cy": 0}}},
                }
            }
            (root / "labels.json").write_text(json.dumps(labels))
            ds = RodDataset(root, root / "labels.json", root)
            image, heatmap, index = ds[0]
            self.assertEqual(tuple(image.shape), (3, 32, 32))
            self.assertEqual(tuple(heatmap.shape), (1, 32, 32))
            self.assertEqual(heatmap[0, 0, 1].item(), 0.125)
            self.assertAlmostEqual(image[0, 0, 0].item(), (1 - 0.485) / 0.229, places=5)
            self.assertEqual(ds.points(index), [(0, 0)])

    def test_validation_counts_annotations_not_target_components(self):
        from types import SimpleNamespace
        from rod_counting.validation import validate

        class Samples(torch.utils.data.Dataset):
            records = [{"filename": "example_scene.png", "source_patches": ["patch.png"]},
                       {"filename": "other_0.png"}]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                image = torch.zeros(3, 16, 16)
                x, y = [(0, 0), (7, 7)][index]
                image[0, y, x] = 1
                return image, torch.zeros(1, 16, 16), index

            def points(self, index):
                return [[(0, 0)], [(7, 7), (12, 12)]][index]

        class Predictions(torch.nn.Module):
            def forward(self, images):
                return images[:, 0:1] * 20 - 10

        loader = torch.utils.data.DataLoader(Samples(), batch_size=2)
        args = SimpleNamespace(
            loss_alpha=10, peak_threshold=0.3, min_distance=3, match_tolerance=2
        )
        result, _ = validate(Predictions(), loader, torch.device("cpu"), args)
        self.assertEqual(result["count_mae"], 0.5)
        self.assertEqual(result["count_bias"], -0.5)
        self.assertEqual(result["precision"], 1)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertEqual(result["scene_count_mae"], 0)
        self.assertEqual(result["patch_count_mae"], 1)

    def test_weighting_emphasizes_centre_errors(self):
        target = torch.tensor([[[[1.0, 0.0]]]])
        logits = torch.zeros_like(target, requires_grad=True)
        self.assertAlmostEqual(heatmap_loss(logits, target, 0).item(), 0.25)
        self.assertGreater(heatmap_loss(logits, target, 10).item(), 0.25)
        heatmap_loss(logits, target).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0, 0, 0, 0], 0)  # Raise centre prediction.
        self.assertGreater(logits.grad[0, 0, 0, 1], 0)  # Lower background prediction.

    def test_resnet50_unet_forward_and_gradients_without_weight_updates(self):
        torch.set_num_threads(2)
        model = build_model(pretrained=False)
        model.eval()
        model.requires_grad_(False)
        # Check differentiability through the network without optimizer/weight updates.
        image = torch.zeros(1, 3, 64, 64, requires_grad=True)
        logits = model(image)
        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))
        heatmap_loss(logits, torch.zeros_like(logits)).backward()
        self.assertTrue(torch.isfinite(image.grad).all())


if __name__ == "__main__":
    unittest.main()
