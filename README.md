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


## Datasets
### TrackPlant3D
The TrackPlant3D dataset (Li et al., COMPAG 2024) contains 3D point clouds of plants at different growth stages with organ instance segmentation, which were sourced from different datasets, annotated and downsampled to 10'000 points per plant. The original dataset can be accessed [here](https://github.com/entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping).

#### Download dataset
We provide a processed version, with leaf keypoint annotations, cleaned segmentations and matched dense point clouds, archived in the ETH Research Collection under CC BY 4.0: <https://hdl.handle.net/20.500.11850/806887>. Please cite that record and the original datasets if you use this data (see the README inside the dataset). Two versions are deposited: **v2** (229 MB, aligned, with normals and auto-segmentation labels) and **v1** (394 MB, unaligned, with the matched dense point clouds).

**Automatic download:** The dataset will be downloaded automatically when you first use the dataloader if not found (v2, 229 MB).

```python
from plant_shape_analysis import PlantSequencesDataset

dataset = PlantSequencesDataset("data/TrackPlant3D/versions") # will auto-download v2 if not found
```

**Manual download (optional):**
```bash
# Download v2 of the dataset (229 MB)
wget -O v2.zip https://www.research-collection.ethz.ch/server/api/core/bitstreams/7d1aeb38-5441-4526-93d8-7acd5e0f12b6/content
folder=data/TrackPlant3D/versions && mkdir -p $folder && unzip v2.zip -d $folder
rm v2.zip
```

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


## Citation

If you use the processed dataset, please cite the original TrackPlant3D dataset
(Li et al., *Computers and Electronics in Agriculture*, 2024), the source datasets
it was assembled fromm, as well as our work [GrowFields](https://github.com/Joaquin-Gajardo/growfields) (ECCV 2026).
The processed version of TrackPlant3D dataset redistributed here with the permission of TrackPlant3D's lead author (Prof. Dawei Li) and Pheno4D dataset's contact person (Prof. Lasse Klingbeil).

## License

MIT, see [LICENSE](LICENSE). Third-party code included here, and its terms, are
listed in [NOTICE](NOTICE): SIREN (`models/siren.py`), DeepSDF
(`utils/sdf_meshing.py`), pycpd (`tracking/trackplant3d/_cpd/`) and PSegNet
(`segmentation/psegnet/`).
