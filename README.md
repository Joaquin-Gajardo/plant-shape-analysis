# plant-shape-analysis


## Environment setup

### Trackplant3D keypoints
Create a python environment with pymeshlab and open3D, e.g. using conda:
```bash
conda create -n plant-shape-analysis python==3.12
conda activate plant-shape-analysis
conda install -c conda-forge gcc gxx  # For X forwarding open3d window with PuTTy and XLaunch (optional)
conda install ipykernel # for Jupyter notebooks in VSCode
pip install open3d plyfile pymeshlab
```

### (Optional) Using my fork of labelCloud
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
3. Set label folder in `File -> Set label folder`
4. Start labelling points by clicking on `Pick sphere`
5. When happy with the sphere, assign it to the class and press `Assign label`
6. Repeat steps 4 and 5 for all leafs you want to label
7. Press `ctrl + s` to save labels (.json + .bin file)
8. Press `Next >>` to go to the next point cloud

