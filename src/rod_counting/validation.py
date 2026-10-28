"""Evaluate heatmaps against annotated centres without updating weights."""

import torch

from .losses import heatmap_loss
from .metrics import find_peaks, match_points, summarize_counts
from .reporting import validation_preview


def validate(model, loader, device, args, preview_indices=()):
    model.eval()
    loss_sum = 0.0
    count_errors = []
    scene_errors, patch_errors = [], []
    true_positives = false_positives = false_negatives = 0
    previews = []
    with torch.inference_mode():
        for images, targets, indices in loader:
            logits = model(images.to(device))
            loss = heatmap_loss(logits, targets.to(device), args.loss_alpha)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite validation loss.")
            loss_sum += loss.item() * len(images)
            for image, prediction, target, index in zip(
                images, torch.sigmoid(logits).cpu(), targets, indices
            ):
                points = find_peaks(
                    prediction[0].numpy(), args.peak_threshold, args.min_distance
                )
                reference = loader.dataset.points(int(index))
                error = len(points) - len(reference)
                count_errors.append(error)
                record = loader.dataset.records[int(index)]
                (scene_errors if record.get("source_patches") else patch_errors).append(error)
                matched, extra, missed = match_points(
                    points, reference, args.match_tolerance
                )
                true_positives += matched
                false_positives += extra
                false_negatives += missed
                if int(index) in preview_indices:
                    filename = loader.dataset.records[int(index)]["filename"]
                    previews.append(
                        validation_preview(
                            image,
                            target[0].numpy(),
                            prediction[0].numpy(),
                            points,
                            reference,
                            filename,
                        )
                    )
    metrics = summarize_counts(
        count_errors, true_positives, false_positives, false_negatives
    )
    metrics["loss"] = loss_sum / len(loader.dataset)
    if scene_errors:
        metrics["scene_count_mae"] = sum(abs(e) for e in scene_errors) / len(scene_errors)
        metrics["scene_images"] = len(scene_errors)
        if patch_errors:
            metrics["patch_count_mae"] = sum(abs(e) for e in patch_errors) / len(patch_errors)
            metrics["patch_images"] = len(patch_errors)
    return metrics, previews
