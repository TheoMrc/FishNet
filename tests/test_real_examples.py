from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")
Image = pytest.importorskip("PIL.Image")

from annotation_app.app import create_app  # noqa: E402
from fish_net.load_data import MIDLINE_POINTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "examples" / "annotation_data"
EXPERIMENT = "2023_02_27_CPF_n=3_1h"
VIDEO = "CPF_1_microM_2"
UNANNOTATED_EXPERIMENT = "FishNet_demo_unannotated"
UNANNOTATED_VIDEO = "demo_video"
UNANNOTATED_FRAMES = (
    "demo_frame_000001.jpg",
    "demo_frame_000002.jpg",
    "demo_frame_000003.jpg",
)


def test_real_example_is_complete_and_browsable():
    video_dir = EXAMPLE_ROOT / EXPERIMENT / VIDEO
    annotations = json.loads(
        (video_dir / "annotations.json").read_text(encoding="utf-8")
    )
    frames = sorted(video_dir.glob("*.jpg"))
    annotated_frames = [path for path in frames if path.name != "background.jpg"]

    assert len(annotated_frames) == 3
    assert set(annotations) == {path.name for path in annotated_frames}
    assert Image.open(video_dir / "background.jpg").size == (512, 512)
    assert all(Image.open(path).size == (512, 512) for path in annotated_frames)
    assert all(
        len(fish["midline_points"]) == MIDLINE_POINTS
        for fish_rows in annotations.values()
        for fish in fish_rows
    )

    client = create_app(EXAMPLE_ROOT).test_client()
    assert client.get("/").status_code == 200
    assert client.get(f"/experiment/{EXPERIMENT}").status_code == 200
    assert client.get(f"/experiment/{EXPERIMENT}/{VIDEO}").status_code == 200
    first_frame = annotated_frames[0].name
    assert (
        client.get(f"/experiment/{EXPERIMENT}/{VIDEO}/{first_frame}").status_code == 200
    )
    image_response = client.get(f"/data/{EXPERIMENT}/{VIDEO}/{first_frame}")
    assert image_response.status_code == 200
    assert image_response.mimetype == "image/jpeg"


def test_real_example_supports_model_assisted_annotation():
    video_dir = EXAMPLE_ROOT / EXPERIMENT / VIDEO
    annotations = json.loads(
        (video_dir / "annotations.json").read_text(encoding="utf-8")
    )
    frame, fish_rows = next(iter(annotations.items()))
    head = fish_rows[0]["midline_points"][0]

    response = (
        create_app(EXAMPLE_ROOT)
        .test_client()
        .post(
            "/auto_annotate_midline",
            json={
                "experiment": EXPERIMENT,
                "video": VIDEO,
                "frame": frame,
                "head_pos": [head["y"], head["x"]],
            },
        )
    )
    assert response.status_code == 200
    points = response.get_json()["predicted_points"]
    assert len(points) == MIDLINE_POINTS
    assert all(np.isfinite([point["x"], point["y"]]).all() for point in points)


def test_unannotated_demo_images_are_discoverable_and_editable():
    video_dir = EXAMPLE_ROOT / UNANNOTATED_EXPERIMENT / UNANNOTATED_VIDEO
    assert (video_dir / "background.jpg").exists()
    assert {path.name for path in video_dir.glob("*.jpg")} == {
        "background.jpg",
        *UNANNOTATED_FRAMES,
    }

    client = create_app(EXAMPLE_ROOT).test_client()
    assert client.get("/").status_code == 200
    assert (
        client.get(
            f"/experiment/{UNANNOTATED_EXPERIMENT}/{UNANNOTATED_VIDEO}"
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/experiment/{UNANNOTATED_EXPERIMENT}/{UNANNOTATED_VIDEO}/review"
        ).status_code
        == 200
    )
    image_response = client.get(
        f"/data/{UNANNOTATED_EXPERIMENT}/{UNANNOTATED_VIDEO}/{UNANNOTATED_FRAMES[0]}"
    )
    assert image_response.status_code == 200
    assert image_response.mimetype == "image/jpeg"
