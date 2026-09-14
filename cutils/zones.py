"""Portable connected-component helpers used by FishNet postprocessing.

The original project also includes the equivalent Cython implementation in
zones.pyx. This fallback keeps a source checkout runnable without a C compiler;
compiling the extension simply replaces it with the faster version.
"""

from __future__ import annotations

import numpy as np


def get_zone(
    x: int,
    y: int,
    boolean_array: np.ndarray,
    zone: list[tuple[int, int]],
    method: str = "adjacent",
) -> None:
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if method == "diag":
        offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    rows, cols = boolean_array.shape
    boolean_array[x, y] = 0
    zone.append((x, y))
    stack = [(x, y)]
    while stack and len(zone) < 1000:
        current_x, current_y = stack.pop()
        for offset_x, offset_y in offsets:
            new_x = current_x + offset_x
            new_y = current_y + offset_y
            if 0 <= new_x < rows and 0 <= new_y < cols and boolean_array[new_x, new_y]:
                boolean_array[new_x, new_y] = 0
                zone.append((new_x, new_y))
                stack.append((new_x, new_y))


def get_zones(boolean_array: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected foreground components without modifying the input."""
    array = np.asarray(boolean_array, dtype=np.uint8).copy()
    zones: list[list[tuple[int, int]]] = []
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            if array[x, y]:
                zone: list[tuple[int, int]] = []
                get_zone(x, y, array, zone)
                zones.append(zone)
    return zones
