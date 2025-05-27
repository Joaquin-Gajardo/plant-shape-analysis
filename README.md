# plant-shape-analysis


## Data

### Key point labelling in point clouds (old, use [labelCloud](#using-my-fork-of-labelcloud) instead)
`point_cloud_labeler.py` is a script that allows you to label point clouds. It uses Open3D for visualization and PyQt5 for the GUI. The script loads a point cloud from a .ply file, displays it, and allows you to label points by clicking on them. The labels are saved in a .json file.

```bash
conda create -n point-cloud-labeler python==3.12
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
cd labelCloud-dev
conda create -n labelCloud-dev python=3.9
conda activate labelCloud-dev
pip install -r requirements.txt
pip install -e . # editable mode works
conda install -c conda-forge gcc gxx # For X forwarding open3d window with XLaunch
```
Usage:
1. Type `labelCloud`
2. Set up folder in `File -> Set point cloud folder`
3. Set label folder in `File -> Set label folder`
4. Start labelling points by clicking on `Pick sphere`
5. When happy with the sphere, assign it to the class and press `Assign label`
6. Repeat steps 4 and 5 for all leafs you want to label
7. Press `ctrl + s` to save labels (.json + .bin file)
8. Press `Next >>` to go to the next point cloud

