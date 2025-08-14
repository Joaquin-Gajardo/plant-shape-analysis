# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A plant shape analysis framework focused on analyzing 3D point cloud data from the TrackPlant3D dataset. The codebase performs leaf tip detection, meshing operations, and visualization of plant structures using geometric analysis and surface reconstruction techniques.

## Development Environment Setup

Create and activate conda environment:
```bash
conda create -n plant-shape-analysis python==3.12
conda activate plant-shape-analysis
conda install -c conda-forge gcc gxx  # For X forwarding open3d window
conda install ipykernel  # for Jupyter notebooks in VSCode
pip install -e .  # Install the package in editable mode
```

## Common Development Tasks

### Running Leaf Tip Detection
Process leaf tip detection for specific crops:
```bash
python src/data_preprocessing/detect_leaf_tips_trackplant3D.py --crop maize --visualize
python src/data_preprocessing/detect_leaf_tips_trackplant3D.py --crop sorghum --meshing_method poisson --meshing_backend pymeshlab
```

### Testing Single Plant Analysis
Process a single plant file with visualization:
```bash
python src/data_preprocessing/detect_leaf_tips_trackplant3D.py --single_plant_path data/TrackPlant3D/gt/maize/1_maize_control_plant1_D00.txt
```

### Running Mesh Analysis
Test different meshing approaches:
```bash
python src/data_preprocessing/leaf_meshing.py --method ball_pivoting --backend open3d
python src/data_preprocessing/leaf_meshing.py --method poisson --backend pymeshlab
```

### Jupyter Notebook Analysis
Launch the exploration notebook:
```bash
jupyter notebook notebooks/1_explore-datasets.ipynb
```

## Architecture Overview

### Core Modules

#### `src/dataloaders/trackplant3D.py`
- **PlantSequencesDataset**: Main dataset class for loading TrackPlant3D sequences
- Organizes plant point clouds by crop type, treatment, and time series
- Handles loading of point clouds (x,y,z,label) and leaf tip annotations
- Supports filtering by crop type and treatment conditions

#### `src/data_preprocessing/detect_leaf_tips_trackplant3D.py`
- **process_plant_keypoints()**: Main function for extracting leaf insertion points and tips
- **extract_leaf_keypoints()**: Uses geodesic distance on mesh surfaces to find tip points
- **mesh_leaf()**: Creates 3D mesh from point cloud using ball pivoting or Poisson reconstruction
- Supports both Open3D and PyMeshLab backends for meshing operations
- Outputs keypoints in JSON format and labelCloud spheres format for manual verification

#### `src/data_preprocessing/leaf_meshing.py`
- **mesh_ball_pivoting_o3d/pymeshlab()**: Ball pivoting algorithm implementations
- **mesh_poisson_o3d/pymeshlab()**: Poisson surface reconstruction implementations
- **calculate_ball_pivoting_radius()**: Adaptive radius calculation for ball pivoting

#### `src/vis/plot_functions.py`
- **visualize_point_cloud()**: Open3D point cloud visualization with color mapping
- **visualize_pymeshlab_mesh()**: Mesh visualization with normal mapping
- Color mapping for semantic labels (stem=0, leaves=1+)

### Data Organization

- **data/TrackPlant3D/gt/**: Point cloud files (x,y,z,organ_instance_label format)
- **data/TrackPlant3D/keypoints/leaf_tips/**: Manual leaf tip annotations
- **data/TrackPlant3D/keypoints_autolabel*/**: Auto-generated keypoint outputs
- **data/TrackPlant3D/keypoints_manual_labelCloud/**: LabelCloud annotation files

### Dataset Structure
- Supports 4 crop types: maize, sorghum, tobacco, tomato
- Multiple treatment conditions: control, drought, heat, shade, highlight
- Time series data with day labels (D00, D01, etc.)
- Plant sequences identified by crop_treatment_plantN pattern

### Key Parameters and Configuration

#### Meshing Options
- **meshing_method**: "ball_pivoting" (default) or "poisson"
- **meshing_backend**: "pymeshlab" (default) or "open3d"
- Ball pivoting generally better for preserving original point indices
- Poisson reconstruction better for smooth surfaces but may alter geometry

#### Output Formats
- JSON keypoints with global/local indices and 3D coordinates
- LabelCloud sphere format for manual verification and editing
- Configurable sphere radius (default: 0.5) for annotation spheres

### Dependencies and Key Libraries
- **open3d**: 3D data processing, visualization, and meshing
- **pymeshlab**: Alternative meshing backend with advanced algorithms
- **torch**: Data loading utilities via Dataset interface
- **numpy**: Numerical operations and point cloud manipulations
- **plyfile**: PLY format point cloud reading (secondary format)

## Important Notes

- Point cloud files use format: x, y, z, organ_instance_label
- Label 0 = stem, labels 1+ = individual leaves
- Geodesic distance computation requires mesh creation from point clouds
- Leaf tip detection finds the two most geodesically distant points, then determines which is closer to stem (insertion) vs tip
- Use median centroid instead of mean for robustness to outliers
- Always verify auto-generated keypoints using labelCloud visualization before using for analysis