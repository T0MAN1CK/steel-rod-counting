"""Local metric logs, validation PNGs and HTML progress pages."""

import csv
import hashlib
import subprocess
import sys

import torch
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def select_previews(dataset, count):
    """Fixed samples spanning empty/sparse through dense validation patches."""
    ordered = sorted(
        range(len(dataset)), key=lambda i: len(dataset.records[i]["regions"])
    )
    return {
        ordered[i]
        for i in np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)
    }


def validation_preview(image, target, prediction, points, reference, filename):
    """Render fixed-scale heatmaps and centres; never binarize the training target."""
    rgb = (
        image.detach().cpu()
        * image.new_tensor([0.229, 0.224, 0.225])[:, None, None].cpu()
    )
    rgb = rgb + rgb.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    rgb = (rgb.permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    photo = Image.fromarray(rgb)
    target_image = Image.fromarray(
        (target.clip(0, 1) * 255).round().astype(np.uint8)
    ).convert("RGB")
    predicted_image = Image.fromarray(
        (prediction.clip(0, 1) * 255).round().astype(np.uint8)
    ).convert("RGB")
    overlay = Image.composite(
        Image.new("RGB", photo.size, "red"),
        photo,
        Image.fromarray((prediction.clip(0, 1) * 160).astype(np.uint8)),
    )
    centres = photo.copy()
    draw = ImageDraw.Draw(centres)
    for x, y in reference:
        draw.line((x - 2, y, x + 2, y), fill="yellow")
        draw.line((x, y - 2, x, y + 2), fill="yellow")
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="lime")
    panels = [
        ("RGB", photo),
        ("Target heatmap", target_image),
        ("Predicted heatmap", predicted_image),
        ("Prediction overlay", overlay),
        ("Pred green / truth yellow", centres),
    ]
    sheet = Image.new("RGB", (photo.width * len(panels), photo.height + 24))
    for column, (title, panel) in enumerate(panels):
        sheet.paste(panel, (column * photo.width, 24))
        ImageDraw.Draw(sheet).text((column * photo.width + 3, 5), title, fill="white")
    return {
        "image": np.asarray(sheet),
        "caption": f"{filename}; predicted {len(points)}, annotated {len(reference)}",
    }


def metric_chart(rows, keys):
    """Small inline SVG chart; no plotting dependency or remote resources."""
    values = [r[key] for r in rows for key in keys]
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)
    lines = []
    colors = ["#2563eb", "#dc2626"]
    for key, color in zip(keys, colors):
        points = [
            (45 + 320 * i / max(len(rows) - 1, 1), 125 - 95 * (r[key] - low) / span)
            for i, r in enumerate(rows)
        ]
        coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        x, y = points[-1]
        lines.append(
            f'<polyline fill="none" stroke="{color}" points="{coordinates}"/>'
            f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>'
        )
    legend = " / ".join(escape(key) for key in keys)
    return (
        f"<figure><figcaption>{legend} (blue / red)</figcaption>"
        f'<svg viewBox="0 0 400 160"><text x="0" y="30">{high:.3g}</text>'
        f'<text x="0" y="130">{low:.3g}</text><path d="M45 20V130H370" fill="none" stroke="#999"/>'
        + "".join(lines)
        + f'<text x="45" y="150">Epoch {rows[0]["epoch"]}</text>'
        f'<text x="285" y="150">Epoch {rows[-1]["epoch"]}</text></svg></figure>'
    )


def save_progress(run_dir, row, previews):
    """Save metrics, epoch PNGs and a self-contained local progress page."""
    run_dir = Path(run_dir)
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(row) + "\n")
    csv_path = run_dir / "metrics.csv"
    new_csv = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as log:
        writer = csv.DictWriter(log, fieldnames=list(row))
        if new_csv:
            writer.writeheader()
        writer.writerow(row)
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    folder = run_dir / "previews"
    folder.mkdir(exist_ok=True)
    images = []
    for index, preview in enumerate(previews, 1):
        filename = f'epoch_{row["epoch"]:03d}_sample_{index:02d}.png'
        Image.fromarray(preview["image"]).save(folder / filename)
        images.append(
            f'<figure><figcaption>{escape(preview["caption"])}</figcaption>'
            f'<img src="previews/{filename}" alt="Validation heatmaps and centres"></figure>'
        )
    chart_keys = [("train/loss", "val/loss"), ("val/count_mae",), ("val/f1",)]
    if "val/scene_count_mae" in row and "val/patch_count_mae" in row:
        chart_keys.append(("val/scene_count_mae", "val/patch_count_mae"))
    validation_note = ""
    if "patch_val/count_mae" in row:
        chart_keys.append(("patch_val/count_mae",))
        validation_note = ("<p>Checkpoint selection and early stopping use full-scene count MAE "
                           "(rods per scene). Additional patch validation is logged separately "
                           "under patch_val/ (rods per patch).</p>")
    charts = "".join(
        metric_chart(rows, keys)
        for keys in chart_keys
    )
    history = " ".join(
        f'<a href="epoch_{r["epoch"]:03d}.html">Epoch {r["epoch"]}</a>' for r in rows
    )
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="30"><title>Training progress</title>'
        "<style>body{font:16px system-ui;margin:24px;background:#fafafa;color:#222}"
        "img{max-width:100%}figure{margin:16px 0}figcaption{margin:8px 0}"
        ".charts{display:flex;flex-wrap:wrap;gap:20px}.charts figure{width:380px}"
        "a{margin-right:12px}</style></head><body>"
        f'<h1>Epoch {row["epoch"]}</h1><p>Best count MAE: {row["best/count_mae"]:.4f}'
        f' at epoch {row["best/epoch"]}. Refreshes every 30 seconds.</p>'
        f'{validation_note}<div class="charts">{charts}</div><nav>{history}</nav>'
        "<h2>Validation predictions</h2>" + "".join(images) + "</body></html>"
    )
    (run_dir / f'epoch_{row["epoch"]:03d}.html').write_text(html, encoding="utf-8")
    (run_dir / "progress.html").write_text(html, encoding="utf-8")


def save_run_config(args, heatmaps):
    args.run_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["heatmap_dir"] = str(heatmaps)
    config["torch_version"] = str(torch.__version__)
    config["python"] = sys.version
    config["gpu"] = torch.cuda.get_device_name()
    config["split_sha256"] = {
        name: hashlib.sha256((args.data / "splits" / name).read_bytes()).hexdigest()
        for name in ["train_labels.json", "validation_labels.json"]
    }
    config["heatmaps"] = json.loads((heatmaps / "summary.json").read_text())
    if config.get("zoom", 0):
        config["zoom_probability"] = 0.5
    if config.get("validation_scenes"):
        root = Path(config["validation_scenes"])
        labels = root / "splits/validation_labels.json"
        records = json.loads(labels.read_text(encoding="utf-8"))
        config["validation_mode"] = "full_scenes"
        config["checkpoint_metric"] = "val/count_mae (rods per full scene)"
        config["validation_scene_batch_size"] = 1
        config["validation_scene_split_sha256"] = hashlib.sha256(labels.read_bytes()).hexdigest()
        config["validation_scene_files"] = sorted(r["filename"] for r in records.values()
                                                   if r.get("source_patches"))
        config["validation_scene_heatmaps"] = json.loads(
            (root / "heatmaps/summary.json").read_text(encoding="utf-8"))
    preparation = args.data / "summary.json"
    if preparation.exists():
        config["data_preparation"] = json.loads(preparation.read_text(encoding="utf-8"))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    config["git_commit"] = git.stdout.strip() if git.returncode == 0 else "unknown"
    config["git_dirty"] = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout
    )
    (args.run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    (args.run_dir / "environment.txt").write_text(packages.stdout, encoding="utf-8")
    return config
