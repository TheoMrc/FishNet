"""Load the released FishNet checkpoint and run PyTorch inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fish_net.load_data import CLASSIFIER_CROP_SIZE, MIDLINE_POINTS, ZONE_SIZE
from fish_net.models import DEVICE, FishNet

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "model.json"
DEFAULT_WEIGHTS = REPOSITORY_ROOT / "models" / "fishnet.pt"


def load_model(
    weights_path: str | Path = DEFAULT_WEIGHTS,
    config_path: str | Path = DEFAULT_CONFIG,
    device: str | torch.device = DEVICE,
) -> tuple[FishNet, dict]:
    """Build FishNet from its published configuration and load its state dict."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    expected_constants = {
        "MIDLINE_POINTS": MIDLINE_POINTS,
        "ZONE_SIZE": ZONE_SIZE,
        "CLASSIFIER_CROP_SIZE": CLASSIFIER_CROP_SIZE,
    }
    for name, expected in expected_constants.items():
        if config[name] != expected:
            raise ValueError(
                f"{name}={config[name]} is incompatible with this source ({expected})"
            )

    model = FishNet(
        midline_points=config["MIDLINE_POINTS"],
        backbone_channels=config["BACKBONE_CHANNELS"],
        midline_channels=config["MIDLINE_CHANNELS"],
        backbone_layers=config["BACKBONE_LAYERS"],
        midline_layers=config["MIDLINE_LAYERS"],
        kernel_size=config["KERNEL_SIZE"],
        classifier_channels=config["CLASSIFIER_CHANNELS"],
    ).to(device)
    state = torch.load(Path(weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, config


def predict_logits(
    frames: np.ndarray,
    head_positions: np.ndarray | None = None,
    *,
    model: FishNet | None = None,
    config: dict | None = None,
) -> dict[str, np.ndarray]:
    """Predict head logits and, when heads are supplied, midline and roll logits.

    frames has shape (batch, 2, height, width). The two channels are the
    normalized grayscale frame and its normalized background image.
    head_positions has shape (batch, fish, 2) in (x, y) order; use (0, 0)
    for padding.
    """
    if model is None or config is None:
        model, config = load_model()
    device = next(model.parameters()).device
    frame_tensor = torch.as_tensor(frames, dtype=torch.float32, device=device)
    with torch.no_grad():
        result = {"head_logits": model(frame_tensor).cpu().numpy()}
        if head_positions is not None:
            heads = torch.as_tensor(head_positions, dtype=torch.long, device=device)
            result["midline_logits"] = (
                model.midline_forward(heads, zone_size=config["ZONE_SIZE"])
                .cpu()
                .numpy()
            )
            result["roll_logits"] = model.classifier_forward().cpu().numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NumPy .npy array shaped (B, 2, H, W)")
    parser.add_argument(
        "--head-positions", type=Path, help="Optional .npy array shaped (B, N, 2)"
    )
    parser.add_argument("--output", type=Path, default=Path("fishnet_predictions.npz"))
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=str(DEVICE))
    args = parser.parse_args()

    model, config = load_model(args.weights, args.config, args.device)
    heads = np.load(args.head_positions) if args.head_positions else None
    output = predict_logits(np.load(args.input), heads, model=model, config=config)
    np.savez_compressed(args.output, **output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
