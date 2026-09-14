import numpy as np

def get_zone(
    x: int,
    y: int,
    boolean_array: np.ndarray,
    zone: list,
    method: str,
) -> list[tuple[int, int]]:
    """
    Adds all contiguous True-pixels connected to the pixel at `(x, y)` on a given 2D array to the `zone` set.
    If `method` is "diag", considers diagonal pixels as well.

        Args:
        - x (int): The starting x coordinate.
        - y (int): The starting y coordinate.
        - boolean_array (np.ndarray[np.uint8_t, ndim=2]): A 2D array with True/False values.
        - zone (list): The list to append the connected True pixels to.
        - method (str): The method of movement, either "adjacent" or "diag".

    """

def get_zones(boolean_array: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    Gathers all contiguous True-pixel zones on a given 2D array.

        Args:
        - boolean_array (np.ndarray[np.uint8_t, ndim=2]): A 2D array with True/False values.
        Returns:
        - list: A list of list, each set containing the connected True pixels for a single zone.

    """
