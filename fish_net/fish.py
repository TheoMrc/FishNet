"""
Fish module to handle the fishes positions.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cutils.zones import get_zones

Coordinate = tuple[int, int]


@dataclass(frozen=True)
class Position:
    """
    Position class to handle the position of the fishes.
    """

    x: int | float
    y: int | float

    def __hash__(self):
        return hash((self.x, self.y))

    def distance(self, other: "Position") -> float:
        """
        Get the distance between two positions.
        """
        return ((other.x - self.x) ** 2 + (other.y - self.y) ** 2) ** 0.5

    @property
    def reversed(self) -> "Position":
        """
        Get the position with swapped x and y.
        """
        return Position(self.y, self.x)

    def flip_horizontal(self, img_size):
        """Flip a position horizontally."""
        return Position(img_size[0] - 1 - self.x, self.y)

    def flip_vertical(self, img_size):
        """Flip a position vertically."""
        return Position(self.x, img_size[1] - 1 - self.y)

    def flip_diagonal(self, _):
        """Flip a position diagonally."""
        return Position(self.y, self.x)

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Position") -> "Position":
        return Position(self.x - other.x, self.y - other.y)


def get_zone_center(zone: set[Coordinate], _: np.ndarray) -> tuple[float, float]:
    """
    Get the center of the zone.
    :param zone: list of coordinates forming the zone
    :param prediction: prediction array
    :return: mass center of the zone
    """
    x_center = float(np.mean([x for x, y in zone]))
    y_center = float(np.mean([y for x, y in zone]))
    return x_center, y_center


def get_zone_weighted_center(
    zone: set[Coordinate], prediction: np.ndarray
) -> tuple[float, float]:
    """
    Get the weighted center of the zone.
    :param zone: list of coordinates forming the zone
    :param prediction: prediction array
    :return: weighted center of the zone
    """
    x_weighted_center = float(
        np.average([x for x, y in zone], weights=[prediction[x, y] for x, y in zone])
    )
    y_weighted_center = float(
        np.average([y for x, y in zone], weights=[prediction[x, y] for x, y in zone])
    )
    return x_weighted_center, y_weighted_center


def get_zone_maximum(zone: set[Coordinate], prediction: np.ndarray) -> Coordinate:
    """
    Get the maximum coordinates in the zone.
    :param zone: list of coordinates forming the zone
    :param prediction: prediction array
    :return: maximum coordinates in the zone
    """
    x_max, y_max = max(
        zone, key=lambda coordinates: prediction[coordinates[0], coordinates[1]]
    )
    return x_max, y_max


def get_fishes_positions(
    prediction: np.ndarray,
    get_zone_function: Callable = get_zone_weighted_center,
    thresh_over_pred: float = 0.5,
) -> list[Position]:
    """
    Get the fishes positions from the prediction.
    :param prediction: output of the model
    :param get_zone_function: function to get the zone center
    :param thresh_over_pred: threshold over the prediction
    :return: list of fishes positions found on prediction
    """
    prediction_bool = prediction > thresh_over_pred
    zones = get_zones(prediction_bool)
    fishes_positions = [
        Position(*get_zone_function(zone, prediction)) for zone in zones
    ]
    return fishes_positions
