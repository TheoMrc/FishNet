"""Module containing the models used for training and inference and related classes or functions."""

import os
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fish_net.fish import Position, get_fishes_positions, get_zone_center
from fish_net.load_data import CLASSIFIER_CROP_SIZE, ZONE_SIZE


def get_device():
    """
    Get the device to use for the model.
    :return: device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")
    return device


DEVICE = get_device()


class FishModel:
    """
    FishModel class to handle the prediction of the fish positions on a frame or a video.
    """

    models_path = os.path.join(
        pathlib.Path(__file__).parent.parent.absolute(), "models"
    )

    def __init__(self, model_name: str):
        model_path = os.path.join(FishModel.models_path, model_name)
        assert model_path.endswith(".pt")
        self.model = torch.load(model_path)

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Call the underlying model and handle reshapes"""
        prediction = self.model(np.expand_dims(img, (0, -1))).squeeze((0, -1))
        return prediction

    def get_poses_on_frame(self, img: np.ndarray) -> list[Position]:
        """
        Get the fishes positions on a frame by prediction.
        :param img: frame
        :return: list of fishes positions
        """
        prediction = self.model(img.reshape((1, 512, 512, 1))).reshape((512, 512))
        fishes_positions = get_fishes_positions(prediction, get_zone_center)
        return fishes_positions


FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34]


class MidlineNet(nn.Module):
    """
    DirectNet model made of convolutional layers without stride, pooling or upsampling.
    """

    def __init__(
        self,
        midline_points: int,
        backbone_channels: int,
        midline_channels: int,
        backbone_layers: int,
        midline_layers: int,
        kernel_size: int,
    ):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=backbone_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=backbone_channels,
                        out_channels=backbone_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding="same",
                    ),
                    nn.BatchNorm2d(backbone_channels),
                )
                for dilation in FIBONACCI[:backbone_layers]
            ]
        )

        # HEAD BRANCH
        self.head_out = nn.Sequential(
            nn.Conv2d(
                in_channels=backbone_channels,
                out_channels=backbone_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=backbone_channels,
                out_channels=1,
                kernel_size=5,
                padding="same",
            ),
        )

        self.last_feature_map: torch.Tensor | None = None

        # MIDLINE BRANCH
        middle = ZONE_SIZE // 2
        y, x = np.ogrid[:ZONE_SIZE, :ZONE_SIZE]
        distance_to_middle = np.round(
            np.sqrt((x - middle) ** 2 + (y - middle) ** 2)
        ).astype(int)
        self.register_buffer(
            "distance_tensor",
            torch.tensor(distance_to_middle, dtype=torch.long),
            persistent=False,
        )
        self.positional_encoding = nn.Parameter(
            torch.randn((1, midline_channels, distance_to_middle.max() + 1))
        )

        self.midline_in = nn.Sequential(
            nn.Conv2d(
                in_channels=backbone_channels,
                out_channels=midline_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(midline_channels),
        )

        self.midline_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=midline_channels,
                        out_channels=midline_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding="same",
                    ),
                    nn.BatchNorm2d(midline_channels),
                )
                for dilation in FIBONACCI[:midline_layers]
            ]
        )

        self.midline_out = nn.Sequential(
            nn.Conv2d(
                in_channels=midline_channels,
                out_channels=midline_points,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(midline_points),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=midline_points,
                out_channels=midline_points,
                kernel_size=11,
                dilation=2,
                padding="same",
            ),
        )

    def get_feature_maps(
        self, head_positions: torch.Tensor, zone_size: int
    ) -> torch.Tensor:
        """
        Extract patches of the feature map centered on the head positions.
        :param head_positions: the positions of the heads
        :param zone_size: the size of the zone to extract
        :return: patches of the feature map centered on the head positions
        """
        # Compute half_size for centering the patch
        half_size = zone_size // 2

        assert self.last_feature_map is not None
        # Pad the entire feature map just once
        padded = F.pad(
            self.last_feature_map,
            (half_size, half_size, half_size, half_size),
            mode="constant",
            value=0,
        )

        # Filter out invalid head positions (those equal to (0,0))
        mask = head_positions.sum(dim=-1) != 0

        # Extract valid positions and their batch indices
        valid_head_positions = head_positions[mask]
        batch_indices = mask.nonzero()[:, 0]

        # Compute top-left corners of patches
        top_left_x = valid_head_positions[:, 0]
        top_left_y = valid_head_positions[:, 1]

        # Number of patches we need to extract
        num_patches = valid_head_positions.size(0)

        # Create a grid of coordinates for a single patch
        current_device = head_positions.device
        y_grid, x_grid = torch.meshgrid(
            torch.arange(zone_size, device=current_device),
            torch.arange(zone_size, device=current_device),
            indexing="ij",
        )

        # Expand the grid to match the number of patches,
        # then offset by each patch's top-left corner
        # shape after unsqueeze: (1, zone_size, zone_size)
        # after broadcast: (num_patches, zone_size, zone_size)
        y_grid = y_grid.unsqueeze(0) + top_left_y.unsqueeze(-1).unsqueeze(-1)
        x_grid = x_grid.unsqueeze(0) + top_left_x.unsqueeze(-1).unsqueeze(-1)

        # Expand batch indices accordingly
        batch_indices = batch_indices.view(-1, 1, 1).expand(
            num_patches, zone_size, zone_size
        )

        # Use advanced indexing to gather all patches simultaneously
        # padded shape: (B, C, H+padding*2, W+padding*2)
        # indexing with [batch_indices, :, y_grid, x_grid]
        # yields shape: (num_patches, C, zone_size, zone_size)
        patches = (
            padded[batch_indices, :, y_grid, x_grid].permute(0, 3, 1, 2).contiguous()
        )

        return patches

    def forward(self, x):
        """
        Forward pass of the DirectNet model.
        :param x: input tensor
        :return: output tensor
        """
        x = self.input_conv(x)
        for layer in self.conv_layers:
            x = F.relu(x + layer(x))
        self.last_feature_map = x

        h = self.head_out(x)
        return h

    def midline_forward(self, head_positions: torch.Tensor, zone_size: int):
        """
        Forward pass of the DirectNet model.
        """
        self.midline_patches = self.get_feature_maps(head_positions, zone_size)
        x = F.relu(self.midline_in(self.midline_patches))

        positional_embedding = self.positional_encoding[:, :, self.distance_tensor]
        x = x + positional_embedding

        for layer in self.midline_layers:
            x = F.relu(layer(x))

        x = self.midline_out(x)
        return x


class FishNet(MidlineNet):
    """
    FishNet class to handle the prediction of:
    - the fish heads on a frame
    - the fish midline positions
    - the fish balance-loss states (binary classifier)
    """

    def __init__(
        self,
        midline_points: int,
        backbone_channels: int,
        midline_channels: int,
        backbone_layers: int,
        midline_layers: int,
        kernel_size: int,
        classifier_channels: int,
    ):
        """
        Inherit from MidlineNet and extend with a binary classifier branch.
        """
        super().__init__(
            midline_points,
            backbone_channels,
            midline_channels,
            backbone_layers,
            midline_layers,
            kernel_size,
        )

        # Classifier branch
        self.classifier_in = nn.Sequential(
            nn.Conv2d(
                in_channels=backbone_channels,
                out_channels=classifier_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(classifier_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier_out = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(classifier_channels, 1),
        )

    def classifier_forward(self) -> torch.Tensor:
        top_left = (ZONE_SIZE - CLASSIFIER_CROP_SIZE) // 2
        patches = self.midline_patches[
            :,
            :,
            top_left : top_left + CLASSIFIER_CROP_SIZE,
            top_left : top_left + CLASSIFIER_CROP_SIZE,
        ]
        patches = patches.detach()
        x = self.classifier_in(patches)
        x = self.classifier_out(x)
        return x
