"""Load a 10x Xenium output directory into an AnnData object.

Reads the cell-feature matrix, joins per-cell metadata and spatial
centroids, and builds a Shapely polygon per cell from the boundary
vertices. Gene symbols are used as ``var_names``.
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc
import geopandas as gpd
from shapely.geometry import Polygon


def load_xenium(dir_path, matrix="cell_feature_matrix.h5",
                cells="cells.parquet", boundaries="cell_boundaries.parquet"):
    """Load a Xenium run into an AnnData with spatial coords and polygons.

    Parameters
    ----------
    dir_path : str
        Path to the Xenium output directory.
    matrix, cells, boundaries : str
        File names within ``dir_path``.

    Returns
    -------
    anndata.AnnData
        ``obs`` carries the cell metadata, polygon ``geometry`` and cell
        ``area``; ``obsm['spatial']`` holds the centroid coordinates.
    """
    dir_path = os.path.expanduser(dir_path)
    h5_path = os.path.join(dir_path, matrix)
    cells_pq = os.path.join(dir_path, cells)
    bounds_pq = os.path.join(dir_path, boundaries)

    adata = sc.read_10x_h5(h5_path)
    adata.obs_names.name = "cell_id"

    for col in ("gene_symbols", "feature_names", "gene_names", "name"):
        if col in adata.var.columns:
            adata.var_names = adata.var[col].astype(str)
            break
    adata.var_names_make_unique()

    cells_df = pd.read_parquet(cells_pq).set_index("cell_id")
    common_ids = adata.obs_names.intersection(cells_df.index)
    adata = adata[common_ids, :].copy()
    adata.obs = adata.obs.join(cells_df.loc[common_ids], how="left")
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy()

    bound_df = pd.read_parquet(bounds_pq)[["cell_id", "vertex_x", "vertex_y"]]
    bound_df = bound_df[bound_df["cell_id"].isin(adata.obs_names)]

    def to_polygon(group):
        return Polygon(zip(group["vertex_x"], group["vertex_y"]))

    poly_series = bound_df.groupby("cell_id", sort=False).apply(
        to_polygon, include_groups=False
    )
    gdf = gpd.GeoDataFrame(
        {"cell_id": poly_series.index, "geometry": poly_series.values}
    ).set_index("cell_id")

    adata.obs = adata.obs.join(gdf, how="left")
    adata.obs["area"] = adata.obs["geometry"].apply(
        lambda g: np.nan if g is None else g.area
    )
    return adata


def subset_roi(adata, cell_ids):
    """Return the subset of ``adata`` whose cells are listed in ``cell_ids``."""
    ids = pd.Index(cell_ids).astype(str)
    return adata[adata.obs_names.isin(ids)].copy()
