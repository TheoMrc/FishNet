# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True, infer_types=True

import numpy as np

cimport numpy as np
from cython cimport boundscheck, cdivision, wraparound


@boundscheck(False)
@wraparound(False)
cpdef void get_zone(
    int x,
    int y,
    np.uint8_t[:, :] boolean_array,
    list zone,
    str method="adjacent",
):
    cdef int new_x, new_y, i, stack_size
    cdef int n_offset = 8 if method == "diag" else 4
    cdef int offsets_x[8]
    cdef int offsets_y[8]
    cdef int[::1] stack_x = np.empty(boolean_array.size, dtype=np.int32)
    cdef int[::1] stack_y = np.empty(boolean_array.size, dtype=np.int32)
    cdef int rows = boolean_array.shape[0]
    cdef int cols = boolean_array.shape[1]

    offsets_x = [1, -1, 0, 0, 1, 1, -1, -1]
    offsets_y = [0, 0, 1, -1, 1, -1, 1, -1]

    boolean_array[x, y] = 0
    zone.append((x, y))
    stack_size = 1
    stack_x[0] = x
    stack_y[0] = y

    while stack_size > 0 and len(zone) < 1000:
        stack_size -= 1
        x = stack_x[stack_size]
        y = stack_y[stack_size]

        for i in range(n_offset):
            new_x = x + offsets_x[i]
            new_y = y + offsets_y[i]
            if (
                0 <= new_x < rows
                and 0 <= new_y < cols
                and boolean_array[new_x, new_y]
            ):
                boolean_array[new_x, new_y] = 0
                zone.append((new_x, new_y))
                stack_x[stack_size] = new_x
                stack_y[stack_size] = new_y
                stack_size += 1
    # No return needed as 'zone' is modified in place

@boundscheck(False)
@wraparound(False)
cpdef list get_zones(np.ndarray[np.uint8_t, ndim=2] boolean_array_np):
    cdef list zones = []
    cdef int rows = boolean_array_np.shape[0]
    cdef int cols = boolean_array_np.shape[1]
    cdef int i, j
    cdef np.uint8_t[:, :] boolean_array = boolean_array_np
    cdef list zone

    for i in range(rows):
        for j in range(cols):
            if boolean_array[i, j]:
                zone = []
                get_zone(i, j, boolean_array, zone, method="adjacent")
                zones.append(zone)

    return zones
