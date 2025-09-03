# plant-shape-analysis


## Setup
Create a python environment with pymeshlab, plyfile, open3D, pytorch, e.g. using conda:
```bash
conda create -n plant-shape-analysis python==3.12
conda activate plant-shape-analysis
conda install -c conda-forge gcc gxx  # For X forwarding open3d window with PuTTy and XLaunch (optional)
conda install ipykernel # for Jupyter notebooks in VSCode
pip install -e . # install package

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

## Training
### Fit single leaf surface

Using Siren (Sitzmann et al., 2020):
```bash
python fit_leaf_surface.py # By defaults using first leaf, run with --help to see CL options
