"""Construct a smoothed expression field and its diffusion vector field.

The pipeline z-scores a gene, applies distance-weighted kNN smoothing on
the cell centroids, interpolates the smoothed values onto a regular grid,
and takes the negative gradient of the grid as the diffusion field
``-grad(gene)``.
"""

import numpy as np
from scipy.sparse import issparse
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors


def get_coords(adata, orient="xy"):
    """Return centroid coordinates, optionally swapping axes."""
    x = adata.obs["x_centroid"].to_numpy()
    y = adata.obs["y_centroid"].to_numpy()
    return (x, y) if orient == "xy" else (y, x)


def get_gene(adata, gene):
    """Return a gene's expression vector, preferring ``layers['counts']``."""
    idx = np.where(adata.var_names.str.upper() == gene.upper())[0]
    if len(idx) == 0:
        raise ValueError(f"{gene} not in var_names.")
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    col = X[:, idx[0]]
    arr = col.toarray().ravel() if issparse(col) else np.asarray(col).ravel()
    return arr.astype(float)


def zscore(v):
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)


def knn_smooth(coords, values, k=25, bw=25.0):
    """Distance-weighted kNN smoothing of ``values`` over ``coords``.

    Weights follow a Gaussian in Euclidean distance with bandwidth ``bw``.
    """
    nn = NearestNeighbors(n_neighbors=k).fit(coords)
    dists, nei = nn.kneighbors(return_distance=True)
    w = np.exp(-(dists ** 2) / (2 * bw ** 2))
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    return (w * values[nei]).sum(axis=1)


def build_grid(x, y, nx=180, ny=180):
    """Return grid axes ``(gx, gy)`` and meshgrid ``(GX, GY)``."""
    gx = np.linspace(x.min(), x.max(), nx)
    gy = np.linspace(y.min(), y.max(), ny)
    GX, GY = np.meshgrid(gx, gy)
    return gx, gy, GX, GY


def interpolate_to_grid(x, y, values, GX, GY):
    """Linear interpolation onto the grid, nearest-filled at the edges."""
    G = griddata((x, y), values, (GX, GY), method="linear")
    m = np.isnan(G)
    if m.any():
        G[m] = griddata((x, y), values, (GX[m], GY[m]), method="nearest")
    return G


def diffusion_field(G, gx, gy, sigma=1.0, sign=-1):
    """Return the (optionally smoothed) gradient field of ``G``.

    ``sign=-1`` gives the diffusion direction ``-grad(G)``. Also returns the
    per-node magnitude.
    """
    G_smooth = gaussian_filter(G, sigma=sigma, mode="nearest")
    dx = np.gradient(gx).mean()
    dy = np.gradient(gy).mean()
    Fy, Fx = np.gradient(G_smooth, dy, dx)
    Fx, Fy = sign * Fx, sign * Fy
    mag = np.hypot(Fx, Fy) + 1e-12
    return Fx, Fy, mag
