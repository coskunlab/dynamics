"""Gradient-descent trajectories along the diffusion field.

Seeds placed at high-expression cells are integrated down the field
``-grad(gene)`` with an adaptive, backtracking step that guarantees
descent and stays inside a tissue mask. Each trajectory terminates in a
watershed basin.
"""

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree
from skimage.morphology import disk
from scipy.ndimage import binary_erosion
from matplotlib.path import Path


def _bilinear(gx, gy, A, x0, y0):
    ix = np.clip(np.searchsorted(gx, x0) - 1, 0, len(gx) - 2)
    iy = np.clip(np.searchsorted(gy, y0) - 1, 0, len(gy) - 2)
    x1, x2 = gx[ix], gx[ix + 1]
    y1, y2 = gy[iy], gy[iy + 1]
    tx = (x0 - x1) / (x2 - x1 + 1e-12)
    ty = (y0 - y1) / (y2 - y1 + 1e-12)
    A11, A21 = A[iy, ix], A[iy, ix + 1]
    A12, A22 = A[iy + 1, ix], A[iy + 1, ix + 1]
    return ((1 - tx) * (1 - ty) * A11 + tx * (1 - ty) * A21
            + (1 - tx) * ty * A12 + tx * ty * A22)


def _cell_to_idx(gx, gy, x, y):
    ix = np.clip(np.searchsorted(gx, x) - 1, 0, len(gx) - 2)
    iy = np.clip(np.searchsorted(gy, y) - 1, 0, len(gy) - 2)
    return iy, ix


def tissue_mask(gx, gy, x, y, shave_um=5.0):
    """Convex-hull tissue mask on the grid, eroded inward by ``shave_um``."""
    hull = ConvexHull(np.c_[x, y]).vertices
    P = Path(np.c_[x, y][hull])
    GX, GY = np.meshgrid(gx, gy)
    inside = P.contains_points(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)

    pix_x = (gx.max() - gx.min()) / (len(gx) - 1)
    pix_y = (gy.max() - gy.min()) / (len(gy) - 1)
    pix = np.sqrt(max(pix_x * pix_y, 1e-12))
    r = max(1, int(round(shave_um / pix)))
    return binary_erosion(inside, structure=disk(r))


def farthest_point_seeds(coords, values, n_seeds=6, top_q=0.95,
                         pool=2000, min_spacing=80.0):
    """Pick well-separated seeds among the top-quantile cells."""
    thr = np.nanpercentile(values, 100 * top_q)
    cand_idx = np.where(values >= thr)[0]
    if len(cand_idx) == 0:
        raise RuntimeError("No candidate seeds above top_q; lower it.")
    if len(cand_idx) > pool:
        cand_idx = np.random.choice(cand_idx, size=pool, replace=False)

    cand_xy = coords[cand_idx]
    chosen = [int(np.argmax(values[cand_idx]))]
    while len(chosen) < n_seeds and len(chosen) < len(cand_idx):
        dmin = np.full(len(cand_idx), np.inf)
        for j in chosen:
            d = np.hypot(cand_xy[:, 0] - cand_xy[j, 0],
                         cand_xy[:, 1] - cand_xy[j, 1])
            dmin = np.minimum(dmin, d)
        dmin[dmin < min_spacing] = -np.inf
        nxt = int(np.argmax(dmin))
        if not np.isfinite(dmin[nxt]):
            break
        chosen.append(nxt)
    return cand_xy[chosen]


def descend_trajectories(seeds, gx, gy, G, Fx, Fy, labels, basin_xy, mask,
                         base_step=1.1, max_steps=1500, leave_seed_r=250.0,
                         req_z_drop=0.6, dz_min=0.01, step_min_frac=0.15,
                         stick_k=8, grad_eps=1e-6):
    """Integrate seeds down ``-grad(gene)`` until they settle in a basin.

    Returns a list of ``(path, endpoint, basin_label, z_drop)`` tuples,
    where ``path`` is an ``(n, 2)`` array of visited coordinates.
    """
    grid_step = 0.5 * ((gx.max() - gx.min()) / (len(gx) - 1)
                       + (gy.max() - gy.min()) / (len(gy) - 1))
    step_min = step_min_frac * grid_step
    basin_tree = KDTree(basin_xy)

    def z_at(x, y):
        iy, ix = _cell_to_idx(gx, gy, x, y)
        return G[iy, ix]

    def inside(x, y):
        iy, ix = _cell_to_idx(gx, gy, x, y)
        return bool(mask[iy, ix])

    def vec(x, y):
        return _bilinear(gx, gy, Fx, x, y), _bilinear(gx, gy, Fy, x, y)

    results = []
    for xs, ys in seeds:
        x, y = xs, ys
        z0 = z_at(x, y)
        pts = [(x, y)]
        last_lab, stick, end_lab = 0, 0, None

        for _ in range(max_steps):
            must_leave = np.hypot(x - xs, y - ys) < leave_seed_r
            ux, uy = vec(x, y)
            gnorm = np.hypot(ux, uy)
            if gnorm < grad_eps:
                break

            step = base_step * min(1.0, grid_step / (gnorm + 1e-12))
            accepted = False
            z_curr = z_at(x, y)
            while step >= step_min:
                x_try, y_try = x + step * ux, y + step * uy
                if not inside(x_try, y_try):
                    step *= 0.5
                    continue
                if (z_curr - z_at(x_try, y_try)) >= dz_min:
                    x, y = x_try, y_try
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break

            pts.append((x, y))
            r, c = _cell_to_idx(gx, gy, x, y)
            lab_here = labels[r, c] if mask[r, c] else 0
            if (not must_leave) and lab_here != 0:
                if lab_here == last_lab:
                    stick += 1
                else:
                    stick, last_lab = 1, lab_here
                if stick >= stick_k and (z0 - z_at(x, y)) >= req_z_drop:
                    end_lab = lab_here
                    break

        path = np.array(pts)
        if end_lab is None:
            _, idx = basin_tree.query([path[-1]], k=1)
            bxy = basin_xy[idx[0][0]]
            results.append((path, tuple(bxy), idx[0][0] + 1,
                            z0 - _bilinear(gx, gy, G, bxy[0], bxy[1])))
        else:
            results.append((path, (x, y), end_lab, z0 - z_at(x, y)))

    return results
