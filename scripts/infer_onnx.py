"""Run the released FishNet ONNX graph on NumPy arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path)
    parser.add_argument("head_positions", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("fishnet_onnx_predictions.npz")
    )
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "fishnet.onnx")
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    values = session.run(
        ["head_logits", "midline_logits", "roll_logits"],
        {
            "frames": np.load(args.frames).astype(np.float32),
            "head_positions": np.load(args.head_positions).astype(np.int64),
        },
    )
    np.savez_compressed(
        args.output,
        head_logits=values[0],
        midline_logits=values[1],
        roll_logits=values[2],
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
