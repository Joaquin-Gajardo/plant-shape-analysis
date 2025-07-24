# plant-shape-analysis


## Setup
Create a python environment with pymeshlab, open3D, pyorch, e.g. using conda:
```bash
conda create -n plant-shape-analysis python==3.12
conda activate plant-shape-analysis
conda install -c conda-forge gcc gxx  # For X forwarding open3d window with PuTTy and XLaunch (optional)
conda install ipykernel # for Jupyter notebooks in VSCode
pip install -e .  # Install the package in editable mode
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
pip install -e . # editable mode works
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

### Usage: TrackPlant3D dataset
Download the TrackPlant3D dataset from [here](https://example.com/trackplant3d), unzip and place in the `data` directory.

```python
from plant_shape_analysis.dataloders.trackplant3D import PlantSequencesDataset

dataset_path = "data/TrackPlant3D"
dataset = PlantSequencesDataset(dataset_path)
print(dataset[0])  # Get first plant sequence (43 in total)
```
