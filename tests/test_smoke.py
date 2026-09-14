from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fish_net.inference import load_model, predict_logits

ROOT = Path(__file__).resolve().parents[1]


def test_released_checkpoint_runs_all_heads():
    model, config = load_model(device="cpu")
    frames = np.zeros((1, 2, 96, 96), dtype=np.float32)
    heads = np.array([[[48, 48], [0, 0]]], dtype=np.int64)
    output = predict_logits(frames, heads, model=model, config=config)

    assert output["head_logits"].shape == (1, 1, 96, 96)
    assert output["midline_logits"].shape == (
        1,
        config["MIDLINE_POINTS"],
        config["ZONE_SIZE"],
        config["ZONE_SIZE"],
    )
    assert output["roll_logits"].shape == (1, 1)
    assert all(np.isfinite(value).all() for value in output.values())


@pytest.mark.slow
def test_onnx_matches_pytorch():
    ort = pytest.importorskip("onnxruntime")
    model, config = load_model(device="cpu")
    rng = np.random.default_rng(1234)
    frames = rng.normal(size=(1, 2, 96, 96)).astype(np.float32)
    heads = np.array([[[48, 48], [0, 0]]], dtype=np.int64)
    expected = predict_logits(frames, heads, model=model, config=config)

    session = ort.InferenceSession(
        str(ROOT / "models" / "fishnet.onnx"),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(
        ["head_logits", "midline_logits", "roll_logits"],
        {"frames": frames, "head_positions": heads},
    )
    for name, value in zip(
        ["head_logits", "midline_logits", "roll_logits"], actual, strict=True
    ):
        np.testing.assert_allclose(expected[name], value, rtol=1e-4, atol=1e-4)


def test_model_configuration_matches_source_constants():
    config = json.loads((ROOT / "configs" / "model.json").read_text())
    assert config["MIDLINE_POINTS"] == 9
    assert config["ZONE_SIZE"] == 81
    assert config["CLASSIFIER_CROP_SIZE"] == 7
    assert sum(parameter.numel() for parameter in load_model()[0].parameters()) > 0
