"""One-batch wiring check; never takes an optimizer step."""

import torch

from .config import parse_args, seed_everything
from .dataloader import prepare_loaders
from .losses import heatmap_loss
from .model import build_model


def check_batch(model, train_loader, device, args):
    model.eval()
    images, targets, _ = next(iter(train_loader))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        model.train()
    with torch.set_grad_enabled(device.type == "cuda"):
        with torch.autocast(
            device_type=device.type, enabled=device.type == "cuda" and not args.no_amp
        ):
            logits = model(images.to(device))
            loss = heatmap_loss(logits, targets.to(device), args.loss_alpha)
        assert logits.shape == targets.shape and torch.isfinite(loss)
        if device.type == "cuda":
            check_scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
            check_scaler.scale(loss).backward()
            if not all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            ):
                raise ValueError("Nonfinite gradients during GPU check.")
            print(
                f"Forward/backward peak allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB; "
                "optimizer-state memory still needs headroom."
            )
    print(
        f"CHECK ONLY: {tuple(logits.shape)}, loss={loss.item():.6f}, "
        f"device={device}, torch={torch.__version__}; no optimizer step or checkpoint."
    )
    return


def main(argv=None):
    """Run the same --check-only path without starting training."""
    import sys

    arguments = sys.argv[1:] if argv is None else argv
    args = parse_args([*arguments, "--check-only"])
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader = prepare_loaders(args)
    model = build_model(args.encoder, not args.no_pretrained).to(device)
    check_batch(model, train_loader, device, args)
    if args.validation_scenes:
        check_validation_batch(model, val_loader, device, args)


def check_validation_batch(model, loader, device, args):
    """Full-scene validation uses eval mode and float32, without weight updates."""
    model.zero_grad(set_to_none=True)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    images, targets, _ = next(iter(loader))
    with torch.inference_mode():
        logits = model(images.to(device))
        loss = heatmap_loss(logits, targets.to(device), args.loss_alpha)
        assert logits.shape == targets.shape and torch.isfinite(loss)
    if device.type == "cuda":
        print(f"Validation forward peak allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB.")
    print(f"VALIDATION CHECK ONLY: {tuple(logits.shape)}, loss={loss.item():.6f}, "
          f"{len(loader.dataset)} unaugmented full scenes; no optimizer step or checkpoint.")


if __name__ == "__main__":
    main()
