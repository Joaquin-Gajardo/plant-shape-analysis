# plant-shape-analysis
A repository for reconstructing and analyzing plant shape evolution using implicit neural representations, with a focus on leaf surface fitting and trait extraction from 3D point clouds.


## Installation

### Minimal installation (dataloaders only)
For using just the dataloaders in other projects (e.g., CanFields):
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
# conda install -c conda-forge gcc gxx  # Optional: for X forwarding open3d window with PuTTy and XLaunch
conda install ipykernel  # for Jupyter notebooks in VSCode
pip install -e .[full]   # install package with all dependencies
```

<details>
<summary>labelCloud fork for labelling keypoints (optional)</summary>

This is a fork of the original labelCloud repository that allows you to label and verify keypoint annotations in point clouds interactively. We create a separate conda environment for this to avoid conflicts with other packages.
```bash
git clone https://github.com/Joaquin-Gajardo/labelCloud-dev
cd labelCloud-dev
conda create -n labelCloud-dev python=3.9
conda activate labelCloud-dev
pip install -r requirements.txt
pip install -e . # install package
conda install -c conda-forge gcc gxx # For X forwarding open3d window with PuTTy and XLaunch (optional)
```
Usage:
1. Type `labelCloud` in the terminal to start the application
2. Set up folder in `File -> Set point cloud folder`
1. Set label folder in `File -> Set label folder`
2. Start labelling points by clicking on `Pick sphere`
5. When happy with the sphere, assign it to the class and press `Assign label`
6. Repeat steps 4 and 5 for all leafs you want to label
1. Press `ctrl + s` to save labels (.json + .bin file)
8. Press `Next >>` to go to the next point cloud

</details>

## Data
### TrackPlant3D
The TrackPlant3D dataset (Li et al., COMPAG 2024) dataset contains 3D point clouds of plants at different growth stages with organ instance segmentation, which were sourced from different datasets, annotated and downsampled to 10'000 points per plant. The original dataset can be accessed [here](https://github.com/entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping).

#### Download dataset
We provide a processed version, with leaf keypoint annotations, cleaned segmentations and matched dense point clouds, available in the following [link](https://polybox.ethz.ch/index.php/apps/files/files/4247413277?dir=/Share/datasets/TrackPlant3D). Please cite the original datasets if you use this dataset (see README within the dataset).

**Automatic download:** The dataset will be downloaded automatically when you first use the dataloader if not found (130 MB).

```python
from plant_shape_analysis import PlantSequencesDataset

dataset = PlantSequencesDataset("data/TrackPlant3D/versions") # will auto-download v2 if not found
```

**Manual download (optional):**
```bash
# Download v2 of the dataset (130.2 MB)
wget -O v2.zip https://polybox.ethz.ch/index.php/s/7XwferiX92aogn5/download
folder=data/TrackPlant3D/versions && mkdir -p $folder && unzip v2.zip -d $folder
rm v2.zip
```

#### Dataloader
We provide PyTorch Dataset classes for loading the TrackPlant3D dataset:

- `PlantSequencesDataset`: Load plants as temporal sequences
- `LeafSequencesDataset`: Load individual leaves as temporal sequences with optional PCA-based alignment

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

# Other utilites
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

## Training
Make sure you have the full installation with all dependencies.
### Fit static leaf surface

#### SIREN
The following is an example of fitting a single leaf surface from a point cloud using Siren (Sitzmann et al., 2020). This can be used as a starting point for implementing other implicit neural representations to plant point clouds, and as a baseline for static leaf surface fitting or to expand to dynamic surface fitting.
```bash
python scripts/fit_leaf_surface.py -p data/TrackPlant3D/versions/v1 --use_ply -s tomato2_control_plant2_leaf1 -t 0  # run with --help to see other CL options
```
