"""Check full-photo padding and annotation-based count evaluation."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from rod_counting.inference import main, predict_heatmap
from rod_counting.metrics import find_peaks


class RedChannelModel(torch.nn.Module):
    def forward(self, inputs):
        assert inputs.shape[-1] % 32 == inputs.shape[-2] % 32 == 0
        red = (inputs[:, :1] * 0.229 + 0.485).clamp(0.001, 0.999)
        return torch.logit(red)


class InferenceTests(unittest.TestCase):
    def test_padding_preserves_photo_dimensions_and_border_peak(self):
        pixels = np.zeros((279, 371, 3), dtype=np.uint8)
        pixels[278, 370, 0] = 255
        prediction = predict_heatmap(
            RedChannelModel(), Image.fromarray(pixels), torch.device("cpu")
        )
        self.assertEqual(prediction.shape, (279, 371))
        self.assertEqual(find_peaks(prediction, 0.65, 3), [(370, 278)])

    def test_cli_reports_counts_from_annotations_and_saves_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "images"
            data.mkdir()
            pixels = np.zeros((275, 351, 3), dtype=np.uint8)
            pixels[192, 192, 0] = 255
            Image.fromarray(pixels).save(data / "example.png")
            labels = {"sample": {"filename": "example.png", "regions": {
                "0": {"shape_attributes": {"cx": 192, "cy": 192}},
                "1": {"shape_attributes": {"cx": 30, "cy": 30}},
            }}}
            (data / "test_labels.json").write_text(json.dumps(labels))
            checkpoint = {"epoch": 62, "config": {"min_distance": 3}}
            output = Path(temp) / "output"
            with patch("rod_counting.inference.load_model",
                       return_value=(RedChannelModel(), checkpoint)), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(["unused.pt", "--device", "cpu", "--data", str(data),
                      "--output", str(output), "--save-previews"])
            result = json.loads((output / "metrics.json").read_text())
            self.assertEqual(result["count_mae"], 1)
            self.assertEqual(result["count_rmse"], 1)
            self.assertEqual(result["count_bias"], -1)
            self.assertEqual(result["total_annotated"], 2)
            with Image.open(output / "example.png_prediction.png") as preview:
                self.assertEqual(preview.size, (351 * 3, 275 + 24))
            with (output / "counts.csv").open(newline="") as source:
                row = next(csv.DictReader(source))
            self.assertEqual(row["predicted_count"], "1")
            self.assertEqual(row["annotated_count"], "2")

    def test_new_images_without_labels_get_predictions_and_previews(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "images"
            data.mkdir()
            pixels = np.zeros((32, 32, 3), dtype=np.uint8)
            pixels[16, 16, 0] = 255
            Image.fromarray(pixels).save(data / "new.png")
            output = Path(temp) / "output"
            with patch("rod_counting.inference.load_model",
                       return_value=(RedChannelModel(), {"epoch": 83, "config": {"min_distance": 3}})), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(["unused.pt", "--device", "cpu", "--data", str(data),
                      "--output", str(output), "--save-previews"])
            result = json.loads((output / "metrics.json").read_text())
            self.assertEqual(result["total_predicted"], 1)
            self.assertEqual(result["unlabeled_image_count"], 1)
            self.assertIsNone(result["count_mae"])
            self.assertIsNone(result["total_annotated"])
            self.assertTrue((output / "new.png_prediction.png").is_file())
            with (output / "counts.csv").open(newline="") as source:
                row = next(csv.DictReader(source))
            self.assertEqual(row["predicted_count"], "1")
            self.assertEqual(row["annotated_count"], "")
            self.assertEqual(row["count_error"], "")

    def test_mixed_folder_evaluates_only_photos_with_reference_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "images"
            data.mkdir()
            pixels = np.zeros((32, 32, 3), dtype=np.uint8)
            pixels[16, 16, 0] = 255
            for name in ("known.png", "new.png"):
                Image.fromarray(pixels).save(data / name)
            (data / "test_labels.json").write_text(json.dumps({"known": {
                "filename": "known.png", "regions": {
                    "0": {"shape_attributes": {"cx": 16, "cy": 16}},
                    "1": {"shape_attributes": {"cx": 5, "cy": 5}},
                }}}))
            output = Path(temp) / "output"
            with patch("rod_counting.inference.load_model",
                       return_value=(RedChannelModel(), {"epoch": 83, "config": {"min_distance": 3}})), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(["unused.pt", "--device", "cpu", "--data", str(data),
                      "--output", str(output)])
            result = json.loads((output / "metrics.json").read_text())
            self.assertEqual(result["image_count"], 2)
            self.assertEqual(result["labeled_image_count"], 1)
            self.assertEqual(result["unlabeled_image_count"], 1)
            self.assertEqual(result["total_predicted"], 2)
            self.assertEqual(result["total_predicted_labeled"], 1)
            self.assertEqual(result["count_mae"], 1)
            self.assertEqual(result["count_bias"], -1)

    def test_explicit_missing_labels_are_reported_as_an_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["unused.pt", "--labels", "missing_labels.json", "--output", "unused"])


if __name__ == "__main__":
    unittest.main()
