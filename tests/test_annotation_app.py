import json

import numpy as np
import pytest

pytest.importorskip("flask")
Image = pytest.importorskip("PIL.Image")

from annotation_app.app import create_app  # noqa: E402
from fish_net.load_data import MIDLINE_POINTS, SupervisedDataset  # noqa: E402


def test_saved_annotations_are_loadable_by_fishnet(tmp_path):
    data_root = tmp_path / "midline-annotations"
    video_dir = data_root / "experiment" / "video"
    video_dir.mkdir(parents=True)
    pixels = np.full((512, 512), 128, dtype=np.uint8)
    Image.fromarray(pixels).save(video_dir / "background.jpg")
    Image.fromarray(pixels).save(video_dir / "frame_000001.jpg")

    app = create_app(data_root)
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert (
        client.get("/experiment/experiment/video/frame_000001.jpg").status_code == 200
    )

    points = [
        {"x": 200.0 + index, "y": 210.0 + index, "visible_bool": True}
        for index in range(MIDLINE_POINTS)
    ]
    response = client.post(
        "/experiment/experiment/video/frame_000001.jpg/save?reviewed=1",
        json=[{"midline_points": points, "rolling_proba": 1.0}],
    )
    assert response.status_code == 200

    saved = json.loads((video_dir / "annotations.json").read_text(encoding="utf-8"))
    assert list(saved) == ["frame_000001.jpg"]
    assert len(saved["frame_000001.jpg"][0]["midline_points"]) == MIDLINE_POINTS
    assert json.loads((video_dir / "review_state.json").read_text())["frame_000001.jpg"]

    dataset = SupervisedDataset(
        str(data_root), ["experiment"], head_sigma=4.0, midline_sigma=2.0
    )
    assert len(dataset) == 1
    assert len(dataset.data[0].annotations_midline[0]) == MIDLINE_POINTS
    assert dataset.data[0].rolling_states == [True]


def test_incomplete_midline_is_rejected(tmp_path):
    video_dir = tmp_path / "experiment" / "video"
    video_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(video_dir / "frame.jpg")
    response = (
        create_app(tmp_path)
        .test_client()
        .post(
            "/experiment/experiment/video/frame.jpg/save",
            json=[{"midline_points": [{"x": 1, "y": 2}]}],
        )
    )
    assert response.status_code == 400


def test_auto_annotation_uses_released_checkpoint(tmp_path):
    video_dir = tmp_path / "experiment" / "video"
    video_dir.mkdir(parents=True)
    Image.fromarray(np.full((512, 512), 150, dtype=np.uint8)).save(
        video_dir / "background.jpg"
    )
    Image.fromarray(np.full((512, 512), 120, dtype=np.uint8)).save(
        video_dir / "frame.jpg"
    )
    response = (
        create_app(tmp_path)
        .test_client()
        .post(
            "/auto_annotate_midline",
            json={
                "experiment": "experiment",
                "video": "video",
                "frame": "frame.jpg",
                "head_pos": [256, 256],
            },
        )
    )
    assert response.status_code == 200
    points = response.get_json()["predicted_points"]
    assert len(points) == MIDLINE_POINTS
    assert all(np.isfinite([point["x"], point["y"]]).all() for point in points)
