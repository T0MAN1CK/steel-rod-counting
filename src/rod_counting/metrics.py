"""Peak detection and one-to-one point matching, independent of torch."""

import numpy as np
from scipy.ndimage import maximum_filter, label, find_objects
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def find_peaks(heatmap, threshold=0.3, min_distance=3):
    """Keep local maxima, collapse plateaus and suppress nearby duplicates.

    Coordinates are returned as (x,y). Border peaks are eligible.
    """
    if not 0 < threshold <= 1 or min_distance < 1:
        raise ValueError("Threshold must be in (0,1], minimum distance >=1.")
    heatmap = np.asarray(heatmap)
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError("Expected a finite 2D heatmap.")
    maxima = (heatmap >= threshold) & (heatmap == maximum_filter(heatmap, size=3))
    plateaus, count = label(maxima, structure=np.ones((3, 3)))
    candidates = []
    for component, bounds in enumerate(find_objects(plateaus), 1):
        ys, xs = np.where(plateaus[bounds] == component)
        ys, xs = ys + bounds[0].start, xs + bounds[1].start
        best = np.argmax(heatmap[ys, xs])
        candidates.append(
            (float(heatmap[ys[best], xs[best]]), int(xs[best]), int(ys[best]))
        )
    kept = []
    for _, x, y in sorted(candidates, key=lambda p: (-p[0], p[2], p[1])):
        if all((x - px) ** 2 + (y - py) ** 2 >= min_distance**2 for px, py in kept):
            kept.append((x, y))
    return kept


def match_points(predicted, reference, tolerance=5.0):
    """Maximum-cardinality one-to-one matching within tolerance, in pixels."""
    if tolerance <= 0:
        raise ValueError("Matching tolerance must be positive.")
    if not predicted or not reference:
        return 0, len(predicted), len(reference)
    edges = csr_matrix(cdist(predicted, reference) <= tolerance)
    matched = maximum_bipartite_matching(edges, perm_type="column")
    tp = int(np.sum(matched >= 0))
    return tp, len(predicted) - tp, len(reference) - tp


def summarize_counts(errors, tp, fp, fn):
    errors = np.asarray(errors, dtype=float)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "count_mae": float(np.abs(errors).mean()),
        "count_rmse": float(np.sqrt(np.mean(errors**2))),
        "count_bias": float(errors.mean()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }
