"""Paired, on-the-fly training augmentation using Pillow only."""

import random

from PIL import Image, ImageEnhance


def zoom_crop(image, points, amount, rng):
    """Uniform zoom, random crop/padding, and matching pixel-centre coordinates."""
    if image.width != image.height or not 0 < amount <= 0.25:
        raise ValueError("Zoom expects square patches and an amount in (0,0.25].")
    width, height = image.size
    if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
        raise ValueError("Zoom requires valid prepared centre coordinates.")
    side = max(1, round(width * rng.uniform(1 - amount, 1 + amount)))
    resized = image.resize((side, side), Image.Resampling.BILINEAR)
    if side >= width:
        dx, dy = -rng.randint(0, side - width), -rng.randint(0, side - height)
        result = resized.crop((-dx, -dy, width - dx, height - dy))
    else:
        dx, dy = rng.randint(0, width - side), rng.randint(0, height - side)
        result = Image.new("RGB", image.size)
        result.paste(resized, (dx, dy))
    transformed = [((x + 0.5) * side / width - 0.5 + dx,
                    (y + 0.5) * side / height - 0.5 + dy) for x, y in points]
    return result, [(x, y) for x, y in transformed if 0 <= x < width and 0 <= y < height]


def stretch_pair(image, heatmap, amount, rng):
    """Compress axes independently and centre-pad; never crop away rods.

    Use bilinear interpolation for the continuous heatmap as well as RGB.
    This changes target widths and is an optional experiment, not the baseline.
    """
    width, height = image.size
    size = (
        max(1, round(width * rng.uniform(1 - amount, 1))),
        max(1, round(height * rng.uniform(1 - amount, 1))),
    )
    offset = ((width - size[0]) // 2, (height - size[1]) // 2)
    photo = Image.new("RGB", image.size)
    target = Image.new("F", heatmap.size)
    photo.paste(image.resize(size, Image.Resampling.BILINEAR), offset)
    target.paste(heatmap.resize(size, Image.Resampling.BILINEAR), offset)
    return photo, target


def augment_pair(image, heatmap, rng=None, *, stretch=0):
    """Return augmented copies; sample geometry once and apply it to both.

    Accept an RGB image and a float32 (Pillow F-mode) centre heatmap.
    Supply random.Random(seed) for repeatable checks. During training, reuse
    a seeded generator rather than resetting its seed for each sample.
    """
    if image.mode != "RGB" or heatmap.mode != "F":
        raise ValueError("Expected an RGB image and an F-mode heatmap.")
    if image.size != heatmap.size:
        raise ValueError("Image and heatmap dimensions must match.")
    # Rectangular scenes can swap width/height under 90-degree rotation.
    # Use batch size 1 for those experiments so samples need not share a shape.
    if not 0 <= stretch <= 0.25:
        raise ValueError("Stretch must be between 0 and 0.25.")
    rng = rng if rng is not None else random
    image, heatmap = image.copy(), heatmap.copy()

    # Exactly the same flips/rotation, with no resizing or interpolation.
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        heatmap = heatmap.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        heatmap = heatmap.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    rotation = rng.choice(
        [
            None,
            Image.Transpose.ROTATE_90,
            Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
        ]
    )
    if rotation is not None:
        image = image.transpose(rotation)
        heatmap = heatmap.transpose(rotation)

    # Appearance changes belong only to the photograph, never the target.
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.9, 1.1))
    # Disabled means no extra random draws: the baseline RNG sequence is unchanged.
    if stretch:
        image, heatmap = stretch_pair(image, heatmap, stretch, rng)
    return image, heatmap


def load_pair(image_path, heatmap_path, *, training=False, rng=None, stretch=0):
    """Read a pair; augment training only. Return PIL images, not tensors.

    Use the training split JSON to select training samples. Validation must
    use its own split JSON and training=False. No files are written.
    """
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    with Image.open(heatmap_path) as target:
        heatmap = target.copy()
    if image.size != heatmap.size or heatmap.mode != "F":
        raise ValueError("Expected matching image/float32 heatmap dimensions.")
    if training:
        return augment_pair(image, heatmap, rng, stretch=stretch)
    return image, heatmap
