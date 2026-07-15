# DYNAMICS

**DY**namic ph**Y**sical-diffusion **N**on-contact intercellular co**MM**un**IC**ation model for **S**patial transcriptomics.

DYNAMICS infers cytokine-mediated, non-contact cell-cell communication from single-cell resolved spatial transcriptomics. It constructs a diffusion field from a chosen ligand or cytokine, predicts communication trajectories by gradient descent along that field, detects sink regions with watershed segmentation, and relates the resulting geometry to cell-type composition and vessel-zone stratification.

This repository accompanies the DYNAMICS manuscript and provides the core pipeline as a small importable package together with a worked example on a human lymph node Xenium dataset.

## Method overview

The pipeline runs in five steps:

1. **Data loading.** A 10x Xenium run is read into an AnnData object with per-cell spatial centroids and Shapely cell polygons (`dynamics.data`).
2. **Cell-type annotation.** Cells are labelled by marker-signature scoring against positive and negative panels (`dynamics.annotation`).
3. **Diffusion field.** A gene is z-scored, smoothed over the tissue by distance-weighted kNN, interpolated onto a regular grid, and differentiated to give the diffusion field `-grad(gene)` (`dynamics.field`).
4. **Basins.** Local minima of the field seed a watershed segmentation into communication sink regions (`dynamics.basins`).
5. **Trajectories.** Seeds at high-expression cells are integrated down the field with an adaptive, backtracking, mask-aware step until they settle in a basin (`dynamics.trajectories`).

## Datasets

The manuscript applies the pipeline to five spatial datasets: a healthy human lymph node (Xenium), reactive follicular hyperplasia and follicular lymphoid hyperplasia tonsil (Xenium), and two minor salivary gland donors from Sjögren's disease (MERSCOPE). Raw data are not redistributed here; see the manuscript for accession details.

## Installation

```bash
git clone https://github.com/coskunlab/DYNAMICS.git
cd DYNAMICS
pip install -r requirements.txt
```

The package targets Python 3.10+.

## Usage

```python
import dynamics as dyn

adata = dyn.load_xenium("/path/to/Xenium_hLymphNode/")
adata = dyn.score_cell_types(adata)

x, y = dyn.field.get_coords(adata)
z = dyn.field.zscore(dyn.field.get_gene(adata, "CXCR4"))
z_smooth = dyn.knn_smooth(np.c_[x, y], z, k=25, bw=25.0)

gx, gy, GX, GY = dyn.build_grid(x, y, nx=180, ny=180)
G = dyn.interpolate_to_grid(x, y, z_smooth, GX, GY)
Fx, Fy, mag = dyn.diffusion_field(G, gx, gy)

labels, basin_xy = dyn.watershed_basins(G, gx, gy)
mask = dyn.trajectories.tissue_mask(gx, gy, x, y)
seeds = dyn.trajectories.farthest_point_seeds(np.c_[x, y], z_smooth)
paths = dyn.descend_trajectories(seeds, gx, gy, G, Fx, Fy, labels, basin_xy, mask)
```



## Repository layout

```
DYNAMICS/
├── dynamics/            core pipeline package
│   ├── data.py          Xenium loading and ROI subsetting
│   ├── annotation.py    marker-signature cell-type scoring
│   ├── field.py         kNN smoothing, grid interpolation, diffusion field
│   ├── basins.py        watershed basin detection
│   ├── trajectories.py  seeding and gradient-descent integration
```

## Citation

If you use this code, please cite the DYNAMICS manuscript (in preparation):

> He YA, Fang Z, Dembowitz S, Larson MC, Adam R, McCoy SS, Coskun AF. DYNAMICS: Integrating reaction–diffusion physics in spatial transcriptomics to quantify non-contact signaling in tissue niches

## Contact

Coskun Lab, Wallace H. Coulter Department of Biomedical Engineering, Georgia Institute of Technology and Emory University. Correspondence: ahmet.coskun@bme.gatech.edu.
