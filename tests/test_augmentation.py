"""Check geometric alignment independently of image appearance changes."""

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
from rod_counting.augmentation import augment_pair, load_pair, stretch_pair, zoom_crop
from rod_counting.heatmaps import centre_heatmap
import numpy as np


class AugmentationTests(unittest.TestCase):
    def test_zoom_in_crops_centres_and_preserves_pixel_alignment(self):
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[15:18, 15:18] = 255
        image = Image.fromarray(pixels)
        rng = Mock()
        rng.uniform.return_value = 1.25
        rng.randint.side_effect = [2, 3]
        photo, points = zoom_crop(image, [(0, 0), (16, 16), (31, 31)], 0.25, rng)
        self.assertEqual(photo.size, image.size)
        self.assertEqual(points, [(18.125, 17.125)])
        self.assertGreater(np.asarray(photo)[17, 18].min(), 200)
        np.testing.assert_array_equal(np.asarray(image), pixels)
        target = np.asarray(centre_heatmap(photo.size, points, 3))
        self.assertEqual(np.unravel_index(target.argmax(), target.shape), (17, 18))

    def test_zoom_out_pads_and_keeps_continuous_centres(self):
        rng = Mock()
        rng.uniform.return_value = 0.75
        rng.randint.side_effect = [3, 5]
        photo, points = zoom_crop(Image.new("RGB", (32, 32), "white"),
                                  [(0, 0), (31, 31)], 0.25, rng)
        self.assertEqual(points, [(2.875, 4.875), (26.125, 28.125)])
        self.assertEqual(photo.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(photo.getpixel((3, 5)), (255, 255, 255))
        self.assertEqual(photo.size, (32, 32))

    def test_zoom_can_make_a_genuinely_empty_crop(self):
        rng = Mock()
        rng.uniform.return_value = 1.25
        rng.randint.side_effect = [8, 8]
        image, points = zoom_crop(Image.new("RGB", (32, 32)), [(0, 0)], 0.25, rng)
        self.assertEqual(points, [])
        self.assertFalse(np.asarray(centre_heatmap(image.size, points, 3)).any())

    def test_all_flip_rotation_combinations_keep_alignment(self):
        image = Image.new("RGB", (8, 8))
        heatmap = Image.new("F", (8, 8))
        # Asymmetric marks, including corners, with a fractional heatmap value.
        for point, value in [((0, 0), 1), ((7, 2), 0.5), ((3, 6), 0.25)]:
            image.putpixel(point, (int(value * 200), 0, 0))
            heatmap.putpixel(point, value)
        original = heatmap.tobytes()
        for horizontal in (False, True):
            for vertical in (False, True):
                for turns in range(4):
                    rng = Mock()
                    rng.random.side_effect = [
                        0 if horizontal else 1,
                        0 if vertical else 1,
                    ]
                    rng.choice.return_value = [
                        None,
                        Image.Transpose.ROTATE_90,
                        Image.Transpose.ROTATE_180,
                        Image.Transpose.ROTATE_270,
                    ][turns]
                    rng.uniform.return_value = 1  # Disable appearance for this check.
                    rgb, target = augment_pair(image, heatmap, rng)
                    for y in range(8):
                        for x in range(8):
                            # Independent inverse mapping for Pillow's CCW rotations.
                            sx, sy = x, y
                            for _ in range(turns):
                                sx, sy = 7 - sy, sx
                            if vertical:
                                sy = 7 - sy
                            if horizontal:
                                sx = 7 - sx
                            self.assertEqual(
                                target.getpixel((x, y)), heatmap.getpixel((sx, sy))
                            )
                            self.assertEqual(
                                rgb.getpixel((x, y)), image.getpixel((sx, sy))
                            )
                    self.assertEqual(
                        sorted(
                            target.getpixel((x, y)) for y in range(8) for x in range(8)
                        ),
                        sorted(
                            heatmap.getpixel((x, y)) for y in range(8) for x in range(8)
                        ),
                    )
        self.assertEqual(heatmap.tobytes(), original)

    def test_disabled_stretch_consumes_no_extra_randomness(self):
        image = Image.new("RGB", (16, 16), (80, 110, 150))
        target = Image.new("F", (16, 16))
        first_rng, second_rng = random.Random(42), random.Random(42)
        first = augment_pair(image, target, first_rng)
        second = augment_pair(image, target, second_rng, stretch=0)
        self.assertEqual(
            [item.tobytes() for item in first], [item.tobytes() for item in second]
        )
        self.assertEqual(first_rng.getstate(), second_rng.getstate())

    def test_stretch_pairs_continuous_heatmaps_and_keeps_border_objects(self):
        values = np.zeros((64, 64), dtype=np.float32)
        for x, y in [(0, 0), (63, 0), (0, 63), (63, 63), (24, 32)]:
            values[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = 0.8
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.rint(values * 255).astype(np.uint8)
        image, target = Image.fromarray(pixels), Image.fromarray(values)
        rng = Mock()
        rng.uniform.side_effect = [0.75, 0.9]
        photo, heatmap = stretch_pair(image, target, 0.25, rng)
        self.assertEqual(photo.size, (64, 64))
        self.assertEqual(heatmap.mode, "F")
        actual = np.asarray(heatmap)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(actual.max(), 0)
        self.assertLessEqual(actual.max(), 1)
        np.testing.assert_allclose(
            np.asarray(photo)[:, :, 0] / 255, actual, atol=2 / 255
        )
        # All four border marks survive the compression and padding.
        from scipy.ndimage import label

        self.assertEqual(label(actual > 0.1)[1], 5)
        self.assertTrue(((actual > 0) & (actual < 0.8)).any())
        np.testing.assert_array_equal(np.asarray(target), values)

    def test_validation_unchanged_and_training_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            image = Image.new("RGB", (8, 8), (80, 110, 150))
            heatmap = Image.new("F", (8, 8))
            heatmap.putpixel((1, 2), 1)
            image.save(folder / "image.png")
            heatmap.save(folder / "target.tiff")
            a, b = load_pair(folder / "image.png", folder / "target.tiff", stretch=0.25)
            self.assertEqual(a.tobytes(), image.tobytes())
            self.assertEqual(b.tobytes(), heatmap.tobytes())
            first = load_pair(
                folder / "image.png",
                folder / "target.tiff",
                training=True,
                rng=random.Random(42),
            )
            second = load_pair(
                folder / "image.png",
                folder / "target.tiff",
                training=True,
                rng=random.Random(42),
            )
            self.assertEqual(
                [p.tobytes() for p in first], [p.tobytes() for p in second]
            )
            self.assertEqual(
                sorted(first[1].getpixel((x, y)) for y in range(8) for x in range(8)),
                sorted(heatmap.getpixel((x, y)) for y in range(8) for x in range(8)),
            )


if __name__ == "__main__":
    unittest.main()
