"""Flask application for annotating FishNet midlines and rolling posture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from PIL import Image
from scipy.interpolate import interp1d, splev, splprep

from fish_net.inference import DEFAULT_WEIGHTS, load_model
from fish_net.load_data import MIDLINE_POINTS, ZONE_SIZE
from fish_net.models import DEVICE

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _inside(root: Path, *parts: str) -> Path:
    root = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        abort(404)
    return candidate


def _frame_names(video_dir: Path, annotations: dict[str, Any]) -> list[str]:
    names = {name for name in annotations if not name.endswith("_reviewed")}
    if video_dir.exists():
        names.update(
            path.name
            for path in video_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and path.name != "background.jpg"
            and not path.stem.endswith("_previous")
        )
    return sorted(names)


def _clean_legacy_annotations(data: dict[str, Any]) -> dict[str, Any]:
    """Remove review flags formerly mixed into annotations.json."""
    return {key: value for key, value in data.items() if not key.endswith("_reviewed")}


def _review_state(video_dir: Path, annotations: dict[str, Any]) -> dict[str, bool]:
    state = _read_json(video_dir / "review_state.json", {})
    # Read old in-file flags without writing them back to annotations.json.
    for key, value in annotations.items():
        if key.endswith("_reviewed") and value is True:
            state[key[: -len("_reviewed")]] = True
    return state


def _validate_annotations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Annotations must be a list of fish objects")
    cleaned: list[dict[str, Any]] = []
    for fish_index, fish in enumerate(value):
        if not isinstance(fish, dict):
            raise ValueError(f"Fish {fish_index + 1} is not an object")
        points = fish.get("midline_points")
        if not isinstance(points, list):
            raise ValueError(
                f"Fish {fish_index + 1} must contain a list of midline points"
            )
        # Match the historical annotation app: a newly placed head may be
        # saved with one point, while completed training annotations contain
        # the nine resampled points required by FishNet. Empty fish are
        # discarded when saving, just as in the reference app.
        if not points:
            continue
        if len(points) not in (1, MIDLINE_POINTS):
            raise ValueError(
                f"Fish {fish_index + 1} must contain 1 or {MIDLINE_POINTS} midline points"
            )
        clean_points = []
        for point_index, point in enumerate(points):
            if not isinstance(point, dict):
                raise ValueError(
                    f"Point {point_index + 1} of fish {fish_index + 1} is not an object"
                )
            try:
                row = float(point["x"])
                column = float(point["y"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Point {point_index + 1} of fish {fish_index + 1} needs numeric x and y"
                ) from error
            if not np.isfinite(row) or not np.isfinite(column):
                raise ValueError("Coordinates must be finite")
            clean_points.append(
                {
                    "x": row,
                    "y": column,
                    "visible_bool": bool(point.get("visible_bool", True)),
                }
            )
        cleaned.append(
            {
                "case": str(fish.get("case", "reliable")),
                "rolling_proba": float(fish.get("rolling_proba", 0.0)),
                "midline_points": clean_points,
            }
        )
    return cleaned


def _smooth_and_resample(points: list[list[float]]) -> np.ndarray:
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("At least two two-dimensional points are required")
    keep = np.r_[True, np.any(np.diff(coordinates, axis=0) != 0, axis=1)]
    coordinates = coordinates[keep]
    if len(coordinates) < 2:
        raise ValueError("At least two distinct points are required")

    if len(coordinates) >= 4:
        weights = np.ones(len(coordinates))
        weights[[0, -1]] = 10
        spline, _ = splprep(
            [coordinates[:, 0], coordinates[:, 1]],
            w=weights,
            k=min(3, len(coordinates) - 1),
            s=0.5,
        )
        parameter = np.linspace(0, 1, max(100, len(coordinates) * 10))
        dense = np.asarray(splev(parameter, spline)).T
    else:
        dense = coordinates

    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    if distance[-1] == 0:
        raise ValueError("The midline has zero length")
    target = np.linspace(0, distance[-1], MIDLINE_POINTS)
    return np.column_stack(
        [interp1d(distance, dense[:, axis])(target) for axis in range(2)]
    )


def _weighted_argmax(probability: np.ndarray, radius: int = 2) -> tuple[float, float]:
    center = np.unravel_index(int(np.argmax(probability)), probability.shape)
    row0, column0 = center
    rows, columns = np.ogrid[: probability.shape[0], : probability.shape[1]]
    mask = (rows - row0) ** 2 + (columns - column0) ** 2 <= radius**2
    weights = np.where(mask, probability, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return float(row0), float(column0)
    return float((weights * rows).sum() / total), float(
        (weights * columns).sum() / total
    )


def create_app(
    data_dir: str | Path | None = None,
    weights_path: str | Path = DEFAULT_WEIGHTS,
) -> Flask:
    app = Flask(__name__)
    repository_root = Path(__file__).resolve().parents[1]
    root = (
        Path(
            data_dir
            or os.environ.get(
                "FISHNET_ANNOTATION_DATA", repository_root / "annotation_data"
            )
        )
        .expanduser()
        .resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    app.config.update(DATA_ROOT=root, WEIGHTS_PATH=Path(weights_path).resolve())
    model_cache: dict[str, Any] = {}

    def experiments() -> list[Path]:
        return sorted(
            (path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name
        )

    def videos(experiment: str) -> list[Path]:
        directory = _inside(root, experiment)
        if not directory.is_dir():
            abort(404)
        return sorted(
            (path for path in directory.iterdir() if path.is_dir()),
            key=lambda p: p.name,
        )

    def video_record(
        experiment: str, video: str
    ) -> tuple[Path, dict[str, Any], dict[str, bool]]:
        directory = _inside(root, experiment, video)
        if not directory.is_dir():
            abort(404)
        raw = _read_json(directory / "annotations.json", {})
        if not isinstance(raw, dict):
            abort(500, "annotations.json must contain a JSON object")
        return directory, _clean_legacy_annotations(raw), _review_state(directory, raw)

    @app.get("/")
    def index():
        rows = []
        total_frames = reviewed_frames = 0
        for experiment_dir in experiments():
            exp_total = exp_reviewed = 0
            for video_dir in videos(experiment_dir.name):
                raw = _read_json(video_dir / "annotations.json", {})
                annotation_data = _clean_legacy_annotations(raw)
                state = _review_state(video_dir, raw)
                frames = _frame_names(video_dir, annotation_data)
                exp_total += len(frames)
                exp_reviewed += sum(bool(state.get(frame)) for frame in frames)
            rows.append(
                {
                    "name": experiment_dir.name,
                    "total": exp_total,
                    "reviewed": exp_reviewed,
                }
            )
            total_frames += exp_total
            reviewed_frames += exp_reviewed
        return render_template(
            "index.html", experiments=rows, total=total_frames, reviewed=reviewed_frames
        )

    @app.get("/experiment/<experiment>")
    def list_videos(experiment: str):
        rows = []
        for video_dir in videos(experiment):
            raw = _read_json(video_dir / "annotations.json", {})
            annotation_data = _clean_legacy_annotations(raw)
            state = _review_state(video_dir, raw)
            frames = _frame_names(video_dir, annotation_data)
            rows.append(
                {
                    "name": video_dir.name,
                    "total": len(frames),
                    "reviewed": sum(bool(state.get(frame)) for frame in frames),
                }
            )
        return render_template("videos.html", experiment=experiment, videos=rows)

    @app.get("/experiment/<experiment>/<video>")
    def list_frames(experiment: str, video: str):
        directory, annotation_data, state = video_record(experiment, video)
        frames = [
            {"name": name, "reviewed": bool(state.get(name))}
            for name in _frame_names(directory, annotation_data)
        ]
        return render_template(
            "frames.html", experiment=experiment, video=video, frames=frames
        )

    @app.get("/experiment/<experiment>/<video>/review")
    def review_video(experiment: str, video: str):
        directory, annotation_data, state = video_record(experiment, video)
        frames = _frame_names(directory, annotation_data)
        target = next((frame for frame in frames if not state.get(frame)), None)
        if target is None:
            return render_template("complete.html", experiment=experiment, video=video)
        return edit_frame(experiment, video, target)

    @app.get("/experiment/<experiment>/<video>/<frame>")
    def edit_frame(experiment: str, video: str, frame: str):
        directory, annotation_data, state = video_record(experiment, video)
        image_path = _inside(directory, frame)
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            abort(404)
        frames = _frame_names(directory, annotation_data)
        if frame not in frames:
            abort(404)
        position = frames.index(frame)
        next_frame = next(
            (name for name in frames[position + 1 :] if not state.get(name)), None
        )
        return render_template(
            "edit_frame.html",
            experiment=experiment,
            video=video,
            frame=frame,
            annotations=annotation_data.get(frame) or [],
            image_url=url_for("data_file", filename=f"{experiment}/{video}/{frame}"),
            next_url=(
                url_for(
                    "edit_frame", experiment=experiment, video=video, frame=next_frame
                )
                if next_frame
                else None
            ),
            current=position + 1,
            total=len(frames),
        )

    @app.post("/experiment/<experiment>/<video>/<frame>/save")
    def save_annotations(experiment: str, video: str, frame: str):
        directory, annotation_data, state = video_record(experiment, video)
        if frame not in _frame_names(directory, annotation_data):
            abort(404)
        try:
            cleaned = _validate_annotations(request.get_json())
        except ValueError as error:
            return jsonify(status="error", message=str(error)), 400
        annotation_data[frame] = cleaned
        _write_json(directory / "annotations.json", annotation_data)
        if request.args.get("reviewed") == "1":
            state[frame] = True
            _write_json(directory / "review_state.json", state)
        return jsonify(status="success")

    @app.post("/smooth_points")
    def smooth_points():
        try:
            smoothed = _smooth_and_resample(request.get_json()["points"])
        except (KeyError, TypeError, ValueError) as error:
            return jsonify(status="error", message=str(error)), 400
        return jsonify(smoothed.tolist())

    @app.post("/auto_annotate_midline")
    def auto_annotate_midline():
        payload = request.get_json()
        try:
            experiment = str(payload["experiment"])
            video = str(payload["video"])
            frame = str(payload["frame"])
            head_column, head_row = (float(value) for value in payload["head_pos"])
        except (KeyError, TypeError, ValueError) as error:
            return jsonify(status="error", message=f"Invalid request: {error}"), 400

        directory, _, _ = video_record(experiment, video)
        frame_path = _inside(directory, frame)
        if not frame_path.is_file():
            abort(404)
        frame_image = (
            np.asarray(Image.open(frame_path).convert("L"), dtype=np.float32) / 255.0
        )
        background_path = directory / "background.jpg"
        if not background_path.exists():
            return jsonify(
                status="error", message="background.jpg is required for auto-annotation"
            ), 400
        background = (
            np.asarray(Image.open(background_path).convert("L"), dtype=np.float32)
            / 255.0
        )
        if frame_image.shape != background.shape:
            return jsonify(
                status="error", message="Frame and background sizes differ"
            ), 400

        if "model" not in model_cache:
            model_cache["model"], model_cache["config"] = load_model(
                app.config["WEIGHTS_PATH"], device=DEVICE
            )
        model = model_cache["model"]
        image = np.stack([frame_image, frame_image - background])[None]
        tensor = torch.from_numpy(image).float()
        head = torch.tensor([[[head_column, head_row]]], dtype=torch.long)
        with torch.no_grad():
            model(tensor)
            logits = model.midline_forward(head, zone_size=ZONE_SIZE)[0]
            probabilities = (
                torch.softmax(logits.flatten(1), dim=1).reshape_as(logits).cpu().numpy()
            )

        points = []
        for heatmap in probabilities:
            local_row, local_column = _weighted_argmax(heatmap)
            points.append(
                {
                    "x": local_row + head_row - ZONE_SIZE // 2,
                    "y": local_column + head_column - ZONE_SIZE // 2,
                    "visible_bool": True,
                }
            )
        return jsonify(predicted_points=points)

    @app.get("/rolling")
    def rolling():
        fish_rows = []
        for experiment_dir in experiments():
            for video_dir in videos(experiment_dir.name):
                _, annotation_data, state = video_record(
                    experiment_dir.name, video_dir.name
                )
                for frame in _frame_names(video_dir, annotation_data):
                    if not state.get(frame):
                        continue
                    for fish_index, fish in enumerate(annotation_data.get(frame) or []):
                        points = fish.get("midline_points", [])
                        if len(points) != MIDLINE_POINTS:
                            continue
                        fish_rows.append(
                            {
                                "experiment": experiment_dir.name,
                                "video": video_dir.name,
                                "frame": frame,
                                "fish_index": fish_index,
                                "head": points[0],
                                "rolling": float(fish.get("rolling_proba", 0.0)) > 0.5,
                                "image_url": url_for(
                                    "data_file",
                                    filename=f"{experiment_dir.name}/{video_dir.name}/{frame}",
                                ),
                            }
                        )
        return render_template("rolling.html", fish_rows=fish_rows)

    @app.post("/rolling/save")
    def save_rolling():
        updates = request.get_json()
        if not isinstance(updates, list):
            return jsonify(status="error", message="Expected a list"), 400
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for update in updates:
            grouped.setdefault(
                (str(update["experiment"]), str(update["video"])), []
            ).append(update)
        for (experiment, video), group in grouped.items():
            directory, annotation_data, _ = video_record(experiment, video)
            for update in group:
                frame = str(update["frame"])
                fish_index = int(update["fish_index"])
                annotation_data[frame][fish_index]["rolling_proba"] = (
                    1.0 if bool(update["rolling"]) else 0.0
                )
            _write_json(directory / "annotations.json", annotation_data)
        return jsonify(status="success")

    @app.get("/data/<path:filename>")
    def data_file(filename: str):
        return send_from_directory(root, filename)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, help="Root of the midline annotation dataset"
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    create_app(args.data_dir, args.weights).run(
        host=args.host, port=args.port, debug=False
    )


if __name__ == "__main__":
    main()
