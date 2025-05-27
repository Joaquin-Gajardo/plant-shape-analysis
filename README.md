# plant-shape-analysis


## Data

### Key point labelling in point clouds
`point_cloud_labeler.py` is a script that allows you to label point clouds. It uses Open3D for visualization and PyQt5 for the GUI. The script loads a point cloud from a .ply file, displays it, and allows you to label points by clicking on them. The labels are saved in a .json file.

Environment (old):
```bash
conda create -n point-cloud-labeler python
conda activate point-cloud-labeler
conda install -c conda-forge gcc gxx # For X forwarding open3d window with XLaunch
pip install numpy open3d pandas
```
Usage:
```bash
python point_cloud_labeler.py my_leaves.ply # or .txt 
```

### Using my fork of labelCloud
Clone https://github.com/Joaquin-Gajardo/labelCloud-dev
```bash
git clone https://github.com/Joaquin-Gajardo/labelCloud-dev
conda create -n labelCloud-dev python=3.9
conda activate labelCloud-dev
pip install -r requirements.txt
pip install -e . # editable mode works
conda install -c conda-forge gcc gxx # For X forwarding open3d window with XLaunch
```
Usage:
```bash
labelCloud my_leaves.ply
```
