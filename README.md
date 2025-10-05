# plant-shape-analysis
A repository for reconstructing and analyzing plant shape evolution using implicit neural representations, with a focus on leaf surface fitting and trait extraction from 3D point clouds.


## Setup
Create a python environment with pymeshlab, plyfile, open3D, pytorch, e.g. using conda:
```bash
conda create -n plant-shape-analysis python==3.12
conda activate plant-shape-analysis
# conda install -c conda-forge gcc gxx  # Optional: for X forwarding open3d window with PuTTy and XLaunch
conda install ipykernel # for Jupyter notebooks in VSCode
pip install -e . # install package
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

```bash
# Download v1 of the dataset (751 MB): minimal cleaning of segmentations, keypoint annotations for leaves, and matched dense point clouds
wget -O v1.zip https://polybox.ethz.ch/index.php/s/mxiZwKfCfd39Rxx/download
folder=data/TrackPlant3D/versions && mkdir -p $folder && unzip v1.zip -d $folder
rm v1.zip
```

#### Dataloader
We provide a dataloader for the TrackPlant3D dataset in [src/plant_shape_analysis/dataloaders/trackplant3D.py](src/plant_shape_analysis/dataloaders/trackplant3D.py). It contains the following classes:

- `PlantSequencesDataset`: A PyTorch Dataset class for loading the TrackPlant3D dataset plants as temporal sequence.
- `LeafSequencesDataset`: A PyTorch Dataset class for loading the TrackPlant3D dataset as organ temporal sequences, using the organ segmentation label on each plant. One can estimate normals and align the leaves using PCA.

Example usage of the `LeafSequencesDataset`:
```python
from src.plant_shape_analysis.dataloaders.trackplant3D import LeafSequencesDataset

dataset_path = Path("data/TrackPlant3D/versions/v1")
leaf_dataset = LeafSequencesDataset(dataset_path, min_timepoints=2, align_pca=True, estimate_normals=True)

# List all sequence names
print(leaf_dataset.get_sequence_names())

# Get a specific leaf timeseries by its unique sequence name
maize_leaf1 = leaf_dataset.get_timeseries_by_sequence_name('maize_control_plant1_leaf1')
print(maize_leaf1)

# Get all leaves from a specific plant sequence
maize_leaves = leaf_dataset.get_timeseries_by_plant_sequence('maize_control_plant1')
print(len(maize_leaves))  # number of leaves in that plant
```


#### Visualization

Run dataloader as main to visualize three best examples of dense leaf sequences:
```bash
python src/plant_shape_analysis/dataloaders/trackplant3D.py
```

## Training
### Fit static leaf surface

#### SIREN
The following is an example of fitting a single leaf surface from a point cloud using Siren (Sitzmann et al., 2020). This can be used as a starting point for implementing other implicit neural representations to plant point clouds, and as a baseline for static leaf surface fitting or to expand to dynamic surface fitting.
```bash
python fit_leaf_surface.py -p data/TrackPlant3D/versions/v1 --use_ply -s tomato2_control_plant2_leaf1 -t 0  # run with --help to see other CL options
```
