"""Check split isolation and checkpoint decisions without training a network."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

from rod_counting.config import parse_args, seed_everything
from rod_counting.dataloader import prepare_loaders
from rod_counting.train import run_training
from rod_counting.reporting import save_run_config
from rod_counting.test import main as evaluate_checkpoint


class TrainingControlsTests(unittest.TestCase):
    def make_scene_fixture(self, root):
        data, scenes = root / "patches", root / "scenes"
        for folder in (data, scenes):
            (folder / "train_data").mkdir(parents=True)
            (folder / "splits").mkdir()
            (folder / "heatmaps/validation").mkdir(parents=True)
            (folder / "heatmaps/summary.json").write_text(json.dumps({"sigma": 3}))
        (data / "heatmaps/train").mkdir()
        for split, names in [("train", ["Photo (1)_0.png"]),
                             ("validation", ["Photo (2)_0.png", "Photo (2)_1.png", "Photo (3)_0.png"])]:
            labels = {}
            for name in names:
                Image.new("RGB", (32, 32)).save(data / "train_data" / name)
                Image.new("F", (32, 32)).save(data / "heatmaps" / split / (name + ".tiff"))
                labels[name] = {"filename": name, "regions": {}}
            (data / "splits" / f"{split}_labels.json").write_text(json.dumps(labels))
        labels = {
            "scene": {"filename": "Photo (2)_scene.png",
                      "source_patches": ["Photo (2)_0.png", "Photo (2)_1.png"],
                      "regions": {"point": {"shape_attributes": {"cx": 20, "cy": 8}}}},
            "leftover": {"filename": "Photo (3)_0.png", "regions": {}},
        }
        for record in labels.values():
            name = record["filename"]
            Image.new("RGB", (64, 32)).save(scenes / "train_data" / name)
            Image.new("F", (64, 32)).save(scenes / "heatmaps/validation" / (name + ".tiff"))
        (scenes / "splits/validation_labels.json").write_text(json.dumps(labels))
        args = parse_args(["--device", "cpu", "--check-only", "--data", str(data),
                           "--validation-scenes", str(scenes), "--batch-size", "8"])
        return data, scenes, args

    def test_scene_validation_excludes_leftovers_and_preserves_training(self):
        with tempfile.TemporaryDirectory() as temp:
            data, scenes, args = self.make_scene_fixture(Path(temp))
            training, validation = prepare_loaders(args)
            self.assertEqual(training.batch_size, 8)
            self.assertEqual(validation.batch_size, 1)
            self.assertEqual(len(validation.dataset), 1)
            self.assertFalse(validation.dataset.training)
            self.assertEqual(validation.dataset.points(0), [(20, 8)])
            self.assertEqual(validation.dataset.records[0]["filename"], "Photo (2)_scene.png")
            path = scenes / "splits/validation_labels.json"
            original = json.loads(path.read_text())
            bad = json.loads(path.read_text())
            bad["scene"]["source_patches"] = ["Photo (1)_0.png"]
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "original validation split"):
                prepare_loaders(args)
            path.write_text(json.dumps(original))
            (scenes / "heatmaps/summary.json").write_text(json.dumps({"sigma": 2}))
            with self.assertRaisesRegex(ValueError, "same sigma"):
                prepare_loaders(args)
            path.write_text(json.dumps({"leftover": original["leftover"]}))
            with self.assertRaisesRegex(ValueError, "No complete"):
                prepare_loaders(args)

    def test_saved_scene_config_evaluates_scenes_automatically(self):
        class Predictions(torch.nn.Module):
            def forward(self, images):
                logits = images.new_full((len(images), 1, *images.shape[-2:]), -20)
                logits[:, 0, 8, 20] = 20
                return logits

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, scenes, args = self.make_scene_fixture(root)
            args.run_dir = root / "run"
            responses = [SimpleNamespace(returncode=0, stdout=value)
                         for value in ["test-revision", "", "torch==test\n"]]
            with patch("rod_counting.reporting.torch.cuda.get_device_name", return_value="test GPU"), \
                    patch("rod_counting.reporting.subprocess.run", side_effect=responses):
                config = save_run_config(args, args.heatmap_dir)
            self.assertEqual(config["validation_mode"], "full_scenes")
            self.assertEqual(config["validation_scene_files"], ["Photo (2)_scene.png"])
            self.assertEqual(config["validation_scene_batch_size"], 1)
            output = root / "evaluation"
            with patch("rod_counting.test.load_model", return_value=(Predictions(), {"config": config, "epoch": 3})), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluate_checkpoint(["unused.pt", "--device", "cpu", "--output", str(output)])
            result = json.loads((output / "metrics.json").read_text())
            self.assertEqual(result["validation_mode"], "full_scenes")
            self.assertEqual(result["samples"], 1)
            self.assertEqual(result["count_mae"], 0)

    def test_scene_checkpoint_selection_ignores_improving_patch_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(run_dir=Path(temp), heatmap_dir=Path(temp), data=Path(temp),
                validation_scenes=Path(temp), batch_size=8, workers=0, seed=43,
                preview_count=0, lr=1e-4, weight_decay=1e-4, no_amp=True,
                epochs=10, patience=2, min_delta=0.001)
            loader = Mock(dataset=SimpleNamespace(records=[{"regions": {}}]))
            optimizer = Mock(param_groups=[{"lr": args.lr}])
            results = []
            for scene, individual in zip([5, 4, 6, 6], [1, 0.2, 0.1, 0.05]):
                results.extend([({"count_mae": scene, "scene_count_mae": scene, "loss": 0.1, "f1": 0.9}, []),
                                ({"count_mae": individual, "loss": 0.1, "f1": 0.9}, [])])
            with patch("rod_counting.train.save_run_config", return_value={}), \
                    patch("rod_counting.train.validation_loader", return_value=loader), \
                    patch("rod_counting.train.select_previews", return_value=set()), \
                    patch("rod_counting.train.torch.optim.AdamW", return_value=optimizer), \
                    patch("rod_counting.train.torch.amp.GradScaler"), \
                    patch("rod_counting.train.torch.save") as save, \
                    patch("rod_counting.train.train_one_epoch", return_value=0.2) as train, \
                    patch("rod_counting.train.validate", side_effect=results), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_training(Mock(), loader, loader, torch.device("cpu"), args)
            self.assertEqual(train.call_count, 4)
            self.assertEqual([c.args[0]["epoch"] for c in save.call_args_list if c.args[1].name == "best.pt"], [1, 2])
            rows = [json.loads(s) for s in (args.run_dir / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(rows[-1]["best/count_mae"], 4)
            self.assertEqual(rows[-1]["patch_val/count_mae"], 0.05)
            page = (args.run_dir / "progress.html").read_text()
            self.assertIn("full-scene count MAE", page)
            self.assertIn("patch_val/count_mae", page)
            optimizer.step.assert_not_called()

    def test_zoom_cli_rejects_invalid_or_confounded_settings(self):
        base = ["--device", "cpu", "--check-only"]
        for flags in [["--zoom", "0.3"], ["--zoom", "0.2", "--no-augmentation"],
                      ["--zoom", "0.2", "--stretch", "0.1"]]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base + flags)
        self.assertEqual(parse_args(base + ["--zoom", "0.2"]).zoom, 0.2)

    def test_loaders_keep_validation_unaugmented_and_reject_source_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data / "train_data").mkdir()
            (data / "splits").mkdir()
            for split, name in [
                ("train", "Photo (1)_0.png"),
                ("validation", "Photo (2)_0.png"),
            ]:
                (data / "heatmaps" / split).mkdir(parents=True)
                Image.new("RGB", (32, 32), (70, 90, 110)).save(
                    data / "train_data" / name
                )
                Image.new("F", (32, 32)).save(
                    data / "heatmaps" / split / (name + ".tiff")
                )
                labels = {"sample": {"filename": name, "regions": {}}}
                (data / "splits" / f"{split}_labels.json").write_text(
                    json.dumps(labels)
                )
            args = parse_args(
                [
                    "--device",
                    "cpu",
                    "--check-only",
                    "--data",
                    str(data),
                    "--stretch",
                    "0.15",
                ]
            )
            seed_everything(args.seed)
            training, validation = prepare_loaders(args)
            self.assertTrue(training.dataset.training)
            self.assertEqual(training.dataset.stretch, 0.15)
            self.assertFalse(validation.dataset.training)
            self.assertEqual(validation.dataset.stretch, 0)
            first = validation.dataset[0]
            second = validation.dataset[0]
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertTrue(torch.equal(first[1], second[1]))
            labels_path = data / "splits/train_labels.json"
            labels = json.loads(labels_path.read_text())
            next(iter(labels.values()))["source_patches"] = ["original_0.png"]
            labels_path.write_text(json.dumps(labels))
            args.batch_size = 2
            with self.assertRaisesRegex(ValueError, "require --batch-size 1"):
                prepare_loaders(args)
            args.batch_size = 1
            # A second patch from the same source photo must fail the runtime guard.
            (data / "splits/validation_labels.json").write_text(
                (data / "splits/train_labels.json").read_text()
            )
            Image.new("F", (32, 32)).save(
                data / "heatmaps/validation/Photo (1)_0.png.tiff"
            )
            with self.assertRaisesRegex(ValueError, "Source photo overlap"):
                prepare_loaders(args)

    def test_checkpoint_minimum_is_separate_from_stopping_tolerance(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(
                run_dir=Path(temp),
                heatmap_dir=Path(temp),
                preview_count=0,
                lr=1e-4,
                weight_decay=1e-4,
                no_amp=True,
                epochs=10,
                patience=2,
                min_delta=0.001,
            )
            samples = SimpleNamespace(records=[{"regions": {}}])
            loader = Mock(dataset=samples)
            optimizer = Mock(param_groups=[{"lr": args.lr}])
            results = [
                ({"loss": 0.1, "count_mae": mae, "f1": 0.9}, [])
                for mae in [3.0, 2.9995, 3.1]
            ]
            with (
                patch("rod_counting.train.save_run_config", return_value={}),
                patch("rod_counting.train.select_previews", return_value=set()),
                patch("rod_counting.train.torch.optim.AdamW", return_value=optimizer),
                patch("rod_counting.train.torch.amp.GradScaler"),
                patch("rod_counting.train.torch.save") as save,
                patch("rod_counting.train.train_one_epoch", return_value=0.2) as train,
                patch("rod_counting.train.validate", side_effect=results),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run_training(Mock(), loader, loader, torch.device("cpu"), args)
            self.assertEqual(train.call_count, 3)
            self.assertEqual(
                [
                    (call.args[0]["epoch"], call.args[1].name)
                    for call in save.call_args_list
                ],
                [
                    (1, "last.pt"),
                    (1, "best.pt"),
                    (2, "last.pt"),
                    (2, "best.pt"),
                    (3, "last.pt"),
                ],
            )
            optimizer.step.assert_not_called()
            rows = [
                json.loads(line)
                for line in (args.run_dir / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["best/epoch"] for row in rows], [1, 2, 2])
            self.assertTrue((args.run_dir / "epoch_003.html").is_file())


if __name__ == "__main__":
    unittest.main()
