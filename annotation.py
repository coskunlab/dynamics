"""Marker-signature cell-type annotation.

Each cell is scored against a set of positive and negative marker panels
and assigned the highest-scoring label. The composite score is
``2 * mean(positive) - 2 * mean(negative)``.
"""

import numpy as np
import pandas as pd

DEFAULT_SIGNATURES = {
    "GC B cells": {"pos": ["MS4A1", "MKI67", "TOP2A", "CXCR4"], "neg": ["TCL1A", "CD27"]},
    "Memory B cells": {"pos": ["MS4A1", "CD27", "CD79A"], "neg": ["MKI67", "PRDM1", "TCL1A"]},
    "Plasma cells": {"pos": ["PRDM1", "MZB1", "TNFRSF17"], "neg": ["MS4A1", "CD19"]},
    "CD4 T cells": {"pos": ["CD4", "CD3D", "CD3E", "IL7R"], "neg": ["CD8A", "FOXP3"]},
    "CD8 T cells": {"pos": ["CD8A", "CD3D", "CD3E"], "neg": ["CD4"]},
    "Endothelial cells": {"pos": ["PECAM1", "VWF", "ERG"], "neg": ["CD3D", "MS4A1"]},
    "Fibroblasts (Stromal cells)": {"pos": ["PDGFRA", "PDGFRB", "THY1"], "neg": ["CD3D", "MS4A1", "CD68"]},
    "Myeloid cells": {"pos": ["CD14", "CD68", "CD163", "MRC1", "CD1C", "FCER1A"], "neg": ["MS4A1", "CD3D"]},
}


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def score_cell_types(adata, signatures=DEFAULT_SIGNATURES, key="cell_type",
                     verbose=False):
    """Assign each cell the highest-scoring signature label.

    Writes the label into ``adata.obs[key]`` as a categorical and returns
    the modified ``adata``. Markers absent from ``var_names`` are ignored.
    """
    scores = {}
    for cell_type, markers in signatures.items():
        pos = [m for m in markers["pos"] if m in adata.var_names]
        neg = [m for m in markers["neg"] if m in adata.var_names]
        if len(pos) == 0:
            if verbose:
                print(f"{cell_type}: no positive markers found, skipping")
            continue

        score = _dense(adata[:, pos].X).mean(axis=1) * 2.0
        if neg:
            score = score - _dense(adata[:, neg].X).mean(axis=1) * 2.0
        scores[cell_type] = np.asarray(score).ravel()
        if verbose:
            print(f"{cell_type}: {len(pos)} pos + {len(neg)} neg markers")

    if not scores:
        raise RuntimeError("No signatures matched any genes in var_names.")

    labels = list(scores.keys())
    score_array = np.column_stack([scores[c] for c in labels])
    adata.obs[key] = pd.Categorical(
        [labels[i] for i in np.argmax(score_array, axis=1)]
    )
    return adata
