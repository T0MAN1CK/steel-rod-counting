"""Continuous Gaussian centre targets shared by preparation and augmentation."""

import math

from PIL import Image


def centre_heatmap(size, points, sigma):
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be finite and positive.")
    width, height = size
    values = [0.0] * (width * height)
    radius = math.ceil(3 * sigma)
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
            raise ValueError(f"Point {(x, y)} outside image {size}")
        for py in range(max(0, math.floor(y) - radius), min(height, math.ceil(y) + radius + 1)):
            for px in range(max(0, math.floor(x) - radius), min(width, math.ceil(x) + radius + 1)):
                value = math.exp(-((px - x) ** 2 + (py - y) ** 2) / (2 * sigma**2))
                index = py * width + px
                values[index] = max(values[index], value)
    heatmap = Image.new("F", size)
    heatmap.putdata(values)
    return heatmap
