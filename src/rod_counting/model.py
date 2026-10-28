"""ResNet encoder + U-Net decoder + one-channel heatmap output."""

import segmentation_models_pytorch as smp
import torch


def build_model(encoder="resnet50", pretrained=True):
    """SMP supplies the encoder, skip connections and decoder blocks.

    Return raw logits; losses.py and validation.py apply sigmoid.
    """
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def load_model(checkpoint_path, device):
    """Load an existing run without downloading ImageNet weights again."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["config"]["encoder"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
