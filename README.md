# plant-shape-analysis

Tools for working with temporal 3D plant point clouds: PyTorch dataloaders for the
TrackPlant3D dataset, a processed version of that dataset with leaf keypoint
annotations and cleaned organ segmentations, and the preprocessing that produced it
(organ alignment, tracking, segmentation, leaf meshing and trait extraction).

The dataloaders are the part other projects depend on -- see
[**Dataloader**](#dataloader) for the supported API. They are used by
[GrowFields](https://github.com/Joaquin-Gajardo/growfields) (ECCV 2026).


## Installation

### Minimal installation (dataloaders only)
For using just the dataloaders in other projects (this is how GrowFields consumes it):
```bash
git clone https://github.com/Joaquin-Gajardo/plant-shape-analysis.git
cd plant-shape-analysis
pip install -e .

# Or directly from GitHub:
pip uninstall plant-shape-analysis -y && pip install git+https://github.com/Joaquin-Gajardo/plant-shape-analysis.git
```
This installs only core dependencies: numpy, torch, open3d, if not already installed in the environment.

### Complete setup for development
For full functionality including visualization, alignment, and experiments:
```bash
conda create -n plant-shape-analysis python>=3.9 # was tested with 3.12
conda activate plant-shape-analysis
conda install ipykernel  # for Jupyter notebooks in VSCode
pip install -e .[full]   # install package with all dependencies
```

Use a **separate environment** for this. The `[full]` extras pull `scikit-image` and
`scipy`, which resolve numpy to 2.x; installing them into an environment built for a
downstream project that pins numpy 1.x (such as GrowFields, where torch, pytorch3d and
open3d are all built against it) will break that environment. For using the dataloaders
from another project, the minimal install above is what you want.


## Datasets
### TrackPlant3D
The TrackPlant3D dataset (Li et al., COMPAG 2024) contains 3D point clouds of plants at different growth stages with organ instance segmentation, which were sourced from different datasets, annotated and downsampled to 10'000 points per plant. The original dataset can be accessed [here](https://github.com/entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping).

#### Download dataset
We provide a processed version, with leaf keypoint annotations, cleaned segmentations and matched dense point clouds, archived in the ETH Research Collection under CC BY 4.0: <https://hdl.handle.net/20.500.11850/806887>. Please cite that record and the original datasets if you use this data (see the README inside the dataset). Two versions are deposited: **v2** (235 MB, aligned, with normals, plus automatic organ labels and the segmentation model behind them) and **v1** (394 MB, unaligned, with the matched dense point clouds).

**Automatic download:** The dataset will be downloaded automatically when you first use the dataloader if not found (v2, 235 MB).

The archives were produced with this repository at commit `cb8b125` for v1 (2025-10-01) and `e3c7566` for v2 (2025-10-27).

```python
from plant_shape_analysis import PlantSequencesDataset

dataset = PlantSequencesDataset("data/TrackPlant3D/versions") # will auto-download v2 if not found
```

**Manual download (optional):**
```bash
# Download v2 of the dataset (235 MB)
wget -O v2.zip https://www.research-collection.ethz.ch/server/api/core/bitstreams/f9202885-a045-4c43-bc5d-765ace0c563f/content
folder=data/TrackPlant3D/versions && mkdir -p $folder && unzip v2.zip -d $folder
rm v2.zip
```

<details>
<summary><b>Known issue:</b> the matrices in v2's <code>transformations/</code> are wrong. They are only a record of the alignment and nothing uses them; the point clouds are fine.</summary>

You need them only if you want to recover v1 (the original data) from v2 yourself. The
alignment itself was correct, but its `composed_matrix` and `inverse_matrix` were recorded
wrongly in these JSONs (fixed in v0.4.1). To regenerate them, rerun the same alignment on v1
with v0.4.1 or later. It writes one JSON per sequence to
`data/TrackPlant3D/versions/v1/transformations/`, which you can move to `v2/transformations/`
to replace the deposited ones:

```python
from plant_shape_analysis import PlantSequencesDataset

manual_z_rotations = {  # as used for v2, see scripts/save_aligned_plants_with_normals.py
    "tobacco_control_plant1": {6: -146.0},
    "tobacco_shade_plant3": {3: -89, 2: 61, 1: 203, 0: 113},
}
PlantSequencesDataset("data/TrackPlant3D/versions", version="v1", alignment_method="stem_based",
                      manual_z_rotations=manual_z_rotations, save_transformations=True)
```

`inverse_matrix` then maps each v2 cloud back onto v1 as loaded. Where `y_to_z_corrected_on_load`
is true (sorghum, tobacco, tomato1), the loader had first turned v1's raw files from Y-up to Z-up
(`point_cloud_utils.correct_orientation`).

</details>

#### Dataloader
We provide PyTorch Dataset classes for loading the TrackPlant3D dataset:

- `PlantSequencesDataset`: Load plants as temporal sequences
- `LeafSequencesDataset`: Load individual leaves as temporal sequences with optional PCA-based alignment

These two classes, together with `plant_shape_analysis.dataloaders.normal_estimation.estimate_normals_for_point_cloud`, are the **supported public API**: they are what downstream projects import ([GrowFields](https://github.com/Joaquin-Gajardo/growfields), CanFields), so changes to them are breaking changes. The rest of `src/` is research code — usable, but not a stable interface.

Example usage:
```python
from pathlib import Path

from plant_shape_analysis import PlantSequencesDataset, LeafSequencesDataset
from plant_shape_analysis.vis.plot_functions import visualize_leaf_sequence

dataset_path = Path("data/TrackPlant3D/versions")

# Load leaf sequences with alignment
leaf_dataset = LeafSequencesDataset(
    dataset_path,
    apply_alignment=True,
)

# Get a specific leaf timeseries by its unique sequence name
leaf_seq = leaf_dataset.get_timeseries_by_sequence_name('maize_control_plant1_leaf1')
print(leaf_seq)

# Visualize the leaf timeseries
visualize_leaf_sequence(leaf_seq, window_name=f"{leaf_seq['sequence_name']}", spacing=50, show_coordinate_frame=True, show_connections=True)

# Other utilities
print(leaf_dataset.get_sequence_names()) # list all leaf sequence names

# Get all leaves from a specific plant sequence
all_leaves = leaf_dataset.get_timeseries_by_plant_sequence('maize_control_plant1')
print(len(all_leaves))  # number of leaves in that plant

```


#### Visualization

Run dataloader as main to visualize three best examples of dense leaf sequences:
```bash
python src/plant_shape_analysis/dataloaders/trackplant3D.py
```

#### Automatic organ labels (`v2-autoseg`)

`version="v2-autoseg"` loads the v2 clouds with organ labels predicted by PSegNet (stem
vs. leaf, then instance clustering) and tracked over time with TrackPlant3D, instead of
the ground truth. These are the labels behind GrowFields' *w/ model segmentation*
experiment. PSegNet was retrained on **all** sequences, test plants included, so they
are not a held-out segmentation benchmark. They contain no stem points on days 8-14 of
`sorghum_heat_plant1`; exclude that sequence (`exclude_sequences=["sorghum_heat_plant1"]`)
when aligning leaves, which needs a stem in every frame.

The deposited labels are the reference. Regenerating them needs `pip install -e .[full]`
and a GPU.

```bash
# 1. Retrain PSegNet on every sequence, or skip this and use the published checkpoint,
#    downloaded with v2 to <dataset dir>/v2/psegnet_checkpoint/best_model.pth.
#    All other settings are the defaults:
#    200 epochs, batch size 8, 2 semantic classes, 15% of frames for validation.
python scripts/train_psegnet.py --use_all_sequences --lr 5e-3
# 2. Segment and track every sequence with the best checkpoint.
python scripts/run_autoseg_pipeline.py \
    --checkpoint outputs/psegnet_trackplant3d/<run>/checkpoints/best_model.pth \
    --num-classes 2 --output-dir <your output dir>
```


## Citation

If you use the processed dataset, please cite the original TrackPlant3D dataset
(Li et al., *Computers and Electronics in Agriculture*, 2024), the source datasets
it was assembled from, as well as our work [GrowFields](https://github.com/Joaquin-Gajardo/growfields) (ECCV 2026).
The processed dataset is redistributed with the permission of TrackPlant3D's lead author (Prof. Dawei Li) and Pheno4D's contact person (Prof. Lasse Klingbeil); the Conn et al. scans are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## License

MIT, see [LICENSE](LICENSE). Third-party code included here, and its terms, are
listed in [NOTICE](NOTICE): SIREN (`models/siren.py`), DeepSDF
(`utils/sdf_meshing.py`), pycpd (`tracking/trackplant3d/_cpd/`) and PSegNet
(`segmentation/psegnet/`).
