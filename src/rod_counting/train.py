"""Train the heatmap U-Net with local metrics, previews and checkpoints."""

import time

import torch

from .config import parse_args, seed_everything
from .dataloader import prepare_loaders, validation_loader
from .losses import heatmap_loss
from .model import build_model
from .reporting import save_progress, save_run_config, select_previews
from .test_loader import check_batch, check_validation_batch
from .validation import validate


def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    total_loss = 0.0
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=not args.no_amp):
            loss = heatmap_loss(model(images), targets, args.loss_alpha)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss.")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


def run_training(model, train_loader, val_loader, device, args):
    config = save_run_config(args, args.heatmap_dir)
    patch_loader = None
    if getattr(args, "validation_scenes", None):
        patch_loader = validation_loader(
            args.data, args.heatmap_dir, args.batch_size, args.workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
    preview_indices = select_previews(val_loader.dataset, args.preview_count)
    print(f'Local progress: {(args.run_dir / "progress.html").resolve()}', flush=True)
    if patch_loader is not None:
        print("Checkpoint selection: full-scene validation count MAE; "
              "patch validation metrics are logged separately.", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    best = float("inf")
    stopping_best = float("inf")
    stale = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args
        )
        metrics, previews = validate(model, val_loader, device, args, preview_indices)
        row = {
            "epoch": epoch,
            "train/loss": train_loss,
            **{f"val/{key}": value for key, value in metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - start,
        }
        if patch_loader is not None:
            patch_metrics, _ = validate(model, patch_loader, device, args)
            row.update({f"patch_val/{key}": value for key, value in patch_metrics.items()})
            row["seconds"] = time.perf_counter() - start
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "metrics": metrics,
        }
        torch.save(checkpoint, args.run_dir / "last.pt")
        if metrics["count_mae"] < best:
            best = metrics["count_mae"]
            best_epoch = epoch
            torch.save(checkpoint, args.run_dir / "best.pt")
        row.update({"best/count_mae": best, "best/epoch": best_epoch})

        if metrics["count_mae"] < stopping_best - args.min_delta:
            stopping_best = metrics["count_mae"]
            stale = 0
        else:
            stale += 1
        save_progress(args.run_dir, row, previews)
        print(
            f"Epoch {epoch}: train loss {train_loss:.5f}, "
            f'val MAE {metrics["count_mae"]:.3f}, F1 {metrics["f1"]:.3f}, '
            f"best MAE {best:.3f}",
            flush=True,
        )
        if stale >= args.patience:
            print("Early stopping; use best.pt, not necessarily last.pt.")
            break


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader = prepare_loaders(args)
    model = build_model(args.encoder, not args.no_pretrained).to(device)
    if args.check_only:
        check_batch(model, train_loader, device, args)
        if args.validation_scenes:
            check_validation_batch(model, val_loader, device, args)
        return
    run_training(model, train_loader, val_loader, device, args)


if __name__ == "__main__":
    main()
