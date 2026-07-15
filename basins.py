"""Watershed basin detection on the smoothed expression field.

Local minima of the field seed a watershed segmentation. Each basin's
representative minimum is returned in data (tissue) coordinates.
"""

import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def watershed_basins(G, gx, gy, min_sep=30):
    """Segment the field into basins around its local minima.

    Parameters
    ----------
    G : ndarray
        Interpolated expression field.
    gx, gy : ndarray
        Grid axes matching ``G``.
    min_sep : int
        Minimum separation (grid pixels) between accepted minima.

    Returns
    -------
    labels : ndarray
        Basin label per grid node (0 is background).
    basin_xy : ndarray, shape (n_basins, 2)
        Representative minimum of each basin in data coordinates.
    """
    G_shift = G - np.nanmin(G)
    mins = peak_local_max(-G_shift, min_distance=min_sep, exclude_border=False)

    markers = np.zeros(G.shape, dtype=int)
    for i, (r, c) in enumerate(mins, start=1):
        markers[r, c] = i
    labels = watershed(G_shift, markers=markers, watershed_line=False)

    basin_xy = []
    for lab in np.unique(labels):
        if lab == 0:
            continue
        region = labels == lab
        r0, c0 = np.argwhere(region)[np.argmin(G_shift[region])]
        basin_xy.append([gx[c0], gy[r0]])

    return labels, np.asarray(basin_xy)
