"""
This module contains classes for loading and processing datasets for supervised and self-supervised
learning.It includes functionality for loading annotations, generating data batches, and applying
data augmentations for self-supervised and supervised training.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from fish_net.fish import Position

DATABASES_FOLDER = "databases"
HEAD_ANNOTATIONS_FOLDER = os.path.join(DATABASES_FOLDER, "head-annotations")
MIDLINE_ANNOTATIONS_FOLDER = os.path.join(DATABASES_FOLDER, "midline-annotations")
IMAGE_SIZE = (512, 512)
ZONE_SIZE = 81
PATCH_SIZE = 81
MAX_FISH_PER_FRAME = 22
PAD_HEADS = 22
HEAD_SIGMA_START = 2
HEAD_SIGMA_END = 1.2
MIDLINE_POINTS = 9
MIDLINE_SIGMA_START = 2
MIDLINE_SIGMA_END = 1.2
CLASSIFIER_CROP_SIZE = 7


def generate_gaussian(
    shape: tuple[int, int], center: tuple[float, float], sigma: float
) -> np.ndarray:
    """
    Generate a 2D Gaussian distribution.
    :param shape: shape of the output array
    :param center: tuple of (x, y) for the Gaussian center
    :param sigma: standard deviation of the Gaussian
    :return: 2D numpy array with Gaussian values
    """
    x = np.arange(0, shape[0], 1, float)
    y = np.arange(0, shape[1], 1, float)[:, np.newaxis]
    x0, y0 = center
    gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    return gaussian


@dataclass
class ImageData:
    """
    Dataclass to store image data.
    """

    experiment: str
    video: str
    filename: str
    image: np.ndarray
    annotations_head_mask: np.ndarray
    annotations_midline: list[list[Position]] | None = None
    annotations_head: list[Position] | None = None
    annotations_midline_masks: np.ndarray | None = None
    rolling_states: list[int] | None = None


def copy_image_data(data: ImageData) -> ImageData:
    return ImageData(
        experiment=data.experiment,
        video=data.video,
        filename=data.filename,
        image=np.copy(data.image),
        annotations_head_mask=np.copy(data.annotations_head_mask)
        if data.annotations_head_mask is not None
        else None,
        annotations_midline=[
            [Position(p.x, p.y) for p in m] for m in data.annotations_midline
        ]
        if data.annotations_midline is not None
        else None,
        annotations_head=[Position(h.x, h.y) for h in data.annotations_head]
        if data.annotations_head is not None
        else None,
        annotations_midline_masks=np.copy(data.annotations_midline_masks)
        if data.annotations_midline_masks is not None
        else None,
        rolling_states=data.rolling_states[:]
        if data.rolling_states is not None
        else None,
    )


class SupervisedDataset(Dataset):
    """Dataset class for supervised learning."""

    def __init__(
        self,
        annotations_folder: str,
        experiments: list[str],
        head_sigma: float,
        midline_sigma: float,
        augment: bool = False,
    ):
        super().__init__()

        self.annotations_folder = annotations_folder
        self.mode = annotations_folder.split(os.path.sep)[-1].split("-")[0]
        self.data: list[ImageData] = []
        self.augment = augment
        self.head_sigma = head_sigma
        self.midline_sigma = midline_sigma
        self.backgrounds: dict[tuple[str, str], np.ndarray] = {}
        for experiment in tqdm(experiments):
            self.add_experiment_to_data(experiment)

        # print description of the dataset including % of rolling fish if midline mode
        if self.mode == "midline":
            rolling_fish = sum(
                [
                    sum(image_data.rolling_states)
                    for image_data in self.data
                    if image_data.rolling_states is not None
                ]
            )
            non_rolling_fish = sum(
                [
                    len(image_data.rolling_states) - sum(image_data.rolling_states)
                    for image_data in self.data
                    if image_data.rolling_states is not None
                ]
            )
            print(
                f"Dataset with {len(self.data)} images with {rolling_fish} "
                f"rolling fish and {non_rolling_fish} non-rolling fish"
            )

    def add_experiment_to_data(self, experiment: str) -> None:
        """
        Add video images and annotations to the dataset.
        :param experiment: experiment name
        :return: None
        """
        experiment_folder = os.path.join(self.annotations_folder, experiment)

        videos = os.listdir(experiment_folder)
        for video in videos:
            with open(
                os.path.join(experiment_folder, video, "annotations.json"),
                "rb",
            ) as f:
                annotations_video = json.loads(f.read())

            self.backgrounds[(experiment, video)] = (
                plt.imread(
                    os.path.join(experiment_folder, video, "background.jpg")
                ).astype(np.float32)
                / 255
            )

            for filename, image_annotations in annotations_video.items():
                if image_annotations is not None:
                    image = (
                        plt.imread(os.path.join(experiment_folder, video, filename))
                        / 255
                    )

                    if self.mode == "midline":
                        # The historical annotation app allows a newly placed
                        # head to be saved before its nine-point midline is
                        # completed. Such fish are intentionally excluded from
                        # supervised midline training, together with their
                        # rolling label, until annotation is complete.
                        image_annotations = [
                            fish_annotations
                            for fish_annotations in image_annotations
                            if len(fish_annotations.get("midline_points", []))
                            == MIDLINE_POINTS
                        ]

                    annotations_head = self.get_annotations_head(
                        image_annotations, mode=self.mode
                    )
                    image_data = ImageData(
                        experiment=experiment,
                        image=image[np.newaxis, :, :].astype(np.float32),
                        annotations_head=annotations_head,
                        video=video,
                        filename=filename,
                        annotations_head_mask=self.annotations_head_to_mask(
                            annotations_head,
                            image_shape=(image.shape[0], image.shape[1]),
                            sigma=self.head_sigma,
                        ),
                    )
                    if self.mode == "midline":
                        annotations_midline = [
                            [
                                Position(
                                    midline_point["x"], midline_point["y"]
                                ).reversed
                                for midline_point in fish_annotations["midline_points"]
                            ]
                            for fish_annotations in image_annotations
                        ]
                        assert len(annotations_head) == len(annotations_midline)
                        image_data.annotations_midline = annotations_midline
                        annotations_midline_masks = np.array(
                            [
                                self.annotations_midline_to_masks(
                                    head_annotation,
                                    midline_annotation,
                                    zone_size=ZONE_SIZE,
                                    sigma=self.midline_sigma,
                                )
                                for head_annotation, midline_annotation in zip(
                                    annotations_head,
                                    annotations_midline,
                                    strict=True,
                                )
                            ]
                        )
                        annotations_midline_masks = annotations_midline_masks[
                            :MAX_FISH_PER_FRAME
                        ]
                        # pad the array first dimension to MAX_FISH_PER_FRAME
                        annotations_midline_masks = np.pad(
                            annotations_midline_masks,
                            (
                                (
                                    0,
                                    MAX_FISH_PER_FRAME
                                    - annotations_midline_masks.shape[0],
                                ),
                                (0, 0),
                                (0, 0),
                                (0, 0),
                            ),
                            mode="constant",
                            constant_values=0,
                        )
                        image_data.annotations_midline_masks = annotations_midline_masks
                        image_data.rolling_states = [
                            fish_annotations.get("rolling_proba", 0) > 0.5
                            for fish_annotations in image_annotations
                        ]
                    self.data.append(image_data)

    @staticmethod
    def get_annotations_head(
        annotations: list[dict[str, Any]], mode: str = "head"
    ) -> list[Position]:
        """
        Get the head annotations from the annotations.
        :param annotations: List of annotated fish positions
        :param mode: can be 'head' or 'midline' depending on the database
        :return: List of head positions
        """
        if mode == "head":
            return [
                Position(annotation["x"], annotation["y"]).reversed
                for annotation in annotations
                if annotation
            ]
        if mode == "midline":
            return [
                Position(
                    annotation["midline_points"][0]["x"],
                    annotation["midline_points"][0]["y"],
                ).reversed
                for annotation in annotations
            ]

        raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def annotations_head_to_mask(
        annotations: list[Position],
        image_shape: tuple[int, int],
        sigma: float,
    ) -> np.ndarray:
        """
        Convert annotations to a binary mask.
        :param annotations: List of annotated fish positions
        :param image_shape: Shape of the image
        :param sigma: Standard deviation of the Gaussian at head position
        :return:
        """
        mask = np.zeros(image_shape, dtype=np.float32)
        for fish_annotation in annotations:
            gaussian_mask = generate_gaussian(
                image_shape, (fish_annotation.x, fish_annotation.y), sigma
            )
            mask += gaussian_mask / gaussian_mask.max()
        return mask[np.newaxis, :, :].clip(0, 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[np.ndarray, ...]:
        image_data = self.data[index]
        head_positions = image_data.annotations_head
        assert head_positions is not None
        if self.augment and self.mode == "midline":
            for _ in range(5):
                if len(head_positions) < MAX_FISH_PER_FRAME - 1:
                    image_data = self.add_artificial_crossing(
                        copy_image_data(image_data)
                    )

        head_positions = image_data.annotations_head
        assert head_positions is not None
        image = image_data.image
        background = self.backgrounds[(image_data.experiment, image_data.video)]
        backgroundless_image = image - background
        image = np.concatenate([image, backgroundless_image], axis=0)
        head_label = image_data.annotations_head_mask
        midline_annotations = image_data.annotations_midline or []
        midline_positions = midline_annotations[:MAX_FISH_PER_FRAME]
        midline_positions = [
            position for positions in midline_positions for position in positions
        ]
        # Head-only annotations use an all-zero midline target.
        midline_masks = image_data.annotations_midline_masks
        midline_label = (
            midline_masks.astype(np.float32)
            if midline_masks is not None
            else np.zeros(
                (MAX_FISH_PER_FRAME, MIDLINE_POINTS, ZONE_SIZE, ZONE_SIZE),
                dtype=np.float32,
            )
        )
        if self.augment:
            (image, head_label, midline_label), (head_positions, midline_positions) = (
                self.augment_sample(
                    arrays=[image, head_label, midline_label],
                    position_lists=[head_positions, midline_positions],
                )
            )

        head_positions = np.array(
            [
                [round(position.x), round(position.y)]
                for position in head_positions[: midline_label.shape[0]]
            ],
            dtype=np.int32,
        )

        head_positions = np.pad(
            head_positions,
            ((0, MAX_FISH_PER_FRAME - head_positions.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        midline_positions = np.array(
            [[position.x, position.y] for position in midline_positions],
            dtype=np.float32,
        ).reshape(-1, MIDLINE_POINTS, 2)

        midline_positions = np.pad(
            midline_positions,
            (
                (0, MAX_FISH_PER_FRAME - midline_positions.shape[0]),
                (0, 0),
                (0, 0),
            ),
            mode="constant",
            constant_values=0,
        )

        rolling_states = (
            np.array(image_data.rolling_states[:MAX_FISH_PER_FRAME])
            if image_data.rolling_states is not None
            else np.zeros((MAX_FISH_PER_FRAME,), dtype=int)
        )
        rolling_states = np.pad(
            rolling_states,
            (0, MAX_FISH_PER_FRAME - rolling_states.shape[0]),
            mode="constant",
            constant_values=0,
        )

        return (
            image,
            head_label,
            midline_label,
            head_positions,
            midline_positions,
            rolling_states,
        )

    def __iter__(self):
        for image_data in self.data:
            yield (
                image_data.image,
                image_data.annotations_head_mask,
                image_data.annotations_head,
                image_data.rolling_states,
            )

    @staticmethod
    def annotations_midline_to_masks(
        head_annotation: Position,
        midline_annotation: list[Position],
        zone_size: int,
        sigma: float,
    ) -> np.ndarray:
        """
        Convert midline annotations to binary masks.
        :param head_annotation: head position
        :param midline_annotation: midline points for one fish
        :param zone_size: size of the zone centered on the head
        :param sigma: standard deviation of the Gaussian
        :return: one mask for each midline point
        """

        mask = np.zeros((len(midline_annotation), zone_size, zone_size))
        for i, position in enumerate(midline_annotation):
            x_centered = min(
                zone_size - 1,
                max(0, position.x - round(head_annotation.x) + zone_size // 2),
            )
            y_centered = min(
                zone_size - 1,
                max(0, position.y - round(head_annotation.y) + zone_size // 2),
            )
            assert 0 <= x_centered < zone_size and 0 <= y_centered < zone_size

            gaussian_mask = generate_gaussian(
                (zone_size, zone_size), (x_centered, y_centered), sigma
            )
            gaussian_mask /= np.sum(gaussian_mask)
            mask[i] = gaussian_mask
        return mask

    @staticmethod
    def augment_sample(
        arrays: list[np.ndarray], position_lists: list[list[Position]]
    ) -> tuple[list[np.ndarray], list[list[Position]]]:
        """Applies random flips and rotations to leverage radial symmetry."""

        # horizontal flip
        if np.random.random() > 0.5:
            for i, array in enumerate(arrays):
                arrays[i] = np.flip(array, axis=-1)
            for i, _position_list in enumerate(position_lists):
                position_lists[i] = [
                    pos.flip_horizontal(IMAGE_SIZE) for pos in position_lists[i]
                ]
        # vertical flip
        if np.random.random() > 0.5:
            for i, array in enumerate(arrays):
                arrays[i] = np.flip(array, axis=-2)
            for i, position_list in enumerate(position_lists):
                position_lists[i] = [
                    pos.flip_vertical(IMAGE_SIZE) for pos in position_list
                ]
        # diagonal flip
        if np.random.random() > 0.5:
            for i, array in enumerate(arrays):
                arrays[i] = np.swapaxes(array, -1, -2)
            for i, position_list in enumerate(position_lists):
                position_lists[i] = [
                    pos.flip_diagonal(IMAGE_SIZE) for pos in position_list
                ]

        return [arr.copy() for arr in arrays], position_lists

    def add_artificial_crossing(self, image_data: ImageData) -> ImageData:
        if (
            self.mode != "midline"
            or image_data.annotations_head is None
            or image_data.annotations_midline is None
            or image_data.rolling_states is None
        ):
            return image_data
        heads = image_data.annotations_head
        midlines = image_data.annotations_midline
        rollings = image_data.rolling_states

        if len(heads) < 2:
            return image_data

        img_center = Position(IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2)
        circle_radius = 175
        in_circle_indices = [
            i for i, h in enumerate(heads) if h.distance(img_center) < circle_radius
        ]

        if len(in_circle_indices) < 2:
            return image_data

        isol_dist = 60
        isolated_indices = [
            i
            for i in in_circle_indices
            if all(
                heads[i].distance(heads[j]) >= isol_dist
                for j in in_circle_indices
                if j != i
            )
        ]

        if not isolated_indices:
            return image_data

        src_idx = random.choice(isolated_indices)
        tgt_idx = random.choice([j for j in in_circle_indices if j != src_idx])

        src_head = heads[src_idx]
        src_midline = midlines[src_idx]
        src_rolling = rollings[src_idx]

        half_size = PATCH_SIZE // 2
        min_x = int(src_head.x - half_size - 1)
        min_y = int(src_head.y - half_size - 1)
        max_x = min_x + PATCH_SIZE
        max_y = min_y + PATCH_SIZE

        if min_x < 0 or min_y < 0 or max_x > IMAGE_SIZE[0] or max_y > IMAGE_SIZE[1]:
            return image_data

        patch = image_data.image[0, min_y:max_y, min_x:max_x].copy()
        rel_midline = [
            p - src_head + Position(half_size, half_size) for p in src_midline
        ]
        patch_arrays, rel_position_lists = self.augment_sample([patch], [rel_midline])
        patch = patch_arrays[0]
        rel_midline = rel_position_lists[0]

        rel_head = rel_midline[0]

        m = random.randint(1, 7)
        k = random.randint(1, 7)
        tgt_point = midlines[tgt_idx][k]
        rel_overlap_point = rel_midline[m]

        offset = rel_overlap_point - rel_head
        new_head = tgt_point - offset

        if new_head.distance(img_center) >= circle_radius:
            return image_data

        new_midline = [new_head + (pos - rel_head) for pos in rel_midline]

        paste_min_x = int(new_head.x - rel_head.x)
        paste_min_y = int(new_head.y - rel_head.y)
        paste_max_x = paste_min_x + PATCH_SIZE
        paste_max_y = paste_min_y + PATCH_SIZE

        if (
            paste_min_x < 0
            or paste_min_y < 0
            or paste_max_x > IMAGE_SIZE[0]
            or paste_max_y > IMAGE_SIZE[1]
        ):
            return image_data

        slice_y = slice(paste_min_y, paste_max_y)
        slice_x = slice(paste_min_x, paste_max_x)
        target_region = image_data.image[0, slice_y, slice_x]

        box_size = 20
        percentile = 95
        corners = [
            (slice(0, box_size), slice(0, box_size)),
            (slice(0, box_size), slice(-box_size, None)),
            (slice(-box_size, None), slice(0, box_size)),
            (slice(-box_size, None), slice(-box_size, None)),
        ]
        bg_patch = [
            np.percentile(patch[sl_y, sl_x], percentile) for sl_y, sl_x in corners
        ]
        bg_target = [
            np.percentile(target_region[sl_y, sl_x], percentile)
            for sl_y, sl_x in corners
        ]
        diffs = [bt - bp for bt, bp in zip(bg_target, bg_patch, strict=True)]
        centers = [
            (box_size / 2, box_size / 2),
            (PATCH_SIZE - box_size / 2, box_size / 2),
            (box_size / 2, PATCH_SIZE - box_size / 2),
            (PATCH_SIZE - box_size / 2, PATCH_SIZE - box_size / 2),
        ]
        A = np.array([[cx, cy, 1] for cx, cy in centers])
        sol = np.linalg.lstsq(A, diffs, rcond=None)[0]
        a, b, c = sol
        xx, yy = np.meshgrid(np.arange(PATCH_SIZE), np.arange(PATCH_SIZE))
        adjustment = a * xx + b * yy + c
        patch += adjustment
        patch = np.clip(patch, 0, 1)
        blended = np.minimum(target_region, patch)
        image_data.image[0, slice_y, slice_x] = blended

        if len(heads) >= MAX_FISH_PER_FRAME:
            return image_data

        heads.append(new_head)
        midlines.append(new_midline)
        rollings.append(src_rolling)

        image_data.annotations_head_mask = self.annotations_head_to_mask(
            image_data.annotations_head, image_shape=IMAGE_SIZE, sigma=self.head_sigma
        )
        new_midline_mask = self.annotations_midline_to_masks(
            new_head, new_midline, zone_size=ZONE_SIZE, sigma=self.midline_sigma
        )
        if image_data.annotations_midline_masks is None:
            image_data.annotations_midline_masks = np.zeros(
                (MAX_FISH_PER_FRAME, MIDLINE_POINTS, ZONE_SIZE, ZONE_SIZE)
            )
        assert image_data.annotations_midline_masks is not None
        image_data.annotations_midline_masks[len(heads) - 1] = new_midline_mask

        return image_data

    def update_midline_annotations(self, gamma: float) -> None:
        """Update midline annotations with a lower sigma. Triggered at the end of an epoch"""
        self.midline_sigma *= gamma
        for image_data in self.data:
            assert (
                image_data.annotations_head is not None
                and image_data.annotations_midline is not None
            )

            annotations_midline_masks = np.array(
                [
                    self.annotations_midline_to_masks(
                        head_annotation,
                        midline_annotation,
                        zone_size=ZONE_SIZE,
                        sigma=self.midline_sigma,
                    )
                    for head_annotation, midline_annotation in zip(
                        image_data.annotations_head,
                        image_data.annotations_midline,
                        strict=True,
                    )
                ]
            )
            annotations_midline_masks = annotations_midline_masks[:MAX_FISH_PER_FRAME]

            annotations_midline_masks = np.pad(
                annotations_midline_masks,
                (
                    (
                        0,
                        MAX_FISH_PER_FRAME - annotations_midline_masks.shape[0],
                    ),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0,
            )
            image_data.annotations_midline_masks = annotations_midline_masks

    def update_head_annotations(self, gamma: float) -> None:
        """Update head annotations with a lower sigma. Triggered at the end of an epoch"""
        self.head_sigma *= gamma
        for image_data in self.data:
            assert image_data.annotations_head is not None

            image_data.annotations_head_mask = self.annotations_head_to_mask(
                image_data.annotations_head,
                image_shape=image_data.image.shape[-2:],
                sigma=self.head_sigma,
            )
