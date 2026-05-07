# Plan: Predicted Organ Labels Pipeline for PlantSequencesDataset

## Context

The GT labels in `PlantSequencesDataset v2` come from manually corrected, temporally consistent organ IDs in the TrackPlant3D paper. To assess the downstream sensitivity of plant shape analysis results to label quality, we need a second annotation produced automatically by running PSegNet (segmentation) + TrackPlant3D tracking (temporal consistency). The output is a new PLY dataset directory where each file has the same fields as v2 but adds a `predicted_organ_label` int32 scalar field. These can be loaded by a new dataset version ("v3") that behaves identically to v2 except it reads from the predicted directory.

---

## Architecture Overview

```
v2 PLY (~10k pts per timepoint)
  ↓ VFPS → 4096 pts
  ↓ PSegNet inference → semantic (6 classes) + instance embeddings (5-dim)
  ↓ MeanShift clustering → per-point instance IDs
  ↓ Map to organ_label format: stem=0, leaf_i=1,2,3,...
  ↓ 3DEPS (boundary-preserving FPS) → 256 pts with predicted labels
  ↓ CPD non-rigid registration (consecutive frame pairs, pycpd)
  ↓ Hungarian matching (organ_matching logic) → temporally consistent predicted IDs
  ↓ KNN propagation: 256 pts → 4096 pts → 10k pts  (cKDTree, already in codebase)
  ↓ Save as PLY with predicted_organ_label field in new directory
```

---

## Decision: Reimplement vs Subprocess

**Recommended: Python imports, not subprocess calls.**

TrackPlant3D's `run.sh` has hardcoded paths, fixed directory structures, and expects TXT files. Calling it via subprocess requires I/O round-trips and brittle path management. Instead, directly import the relevant Python modules from both repos (via `sys.path` injection) and call them in-memory. This is a one-time preprocessing script, so the added setup cost is acceptable.

---

## New Files

**Single new script:**
```
src/plant_shape_analysis/data_preprocessing/create_predicted_labels.py
```

**Output dataset directory:**
```
data/TrackPlant3D/versions/v2/predicted_labels/{crop}/{id}_{crop}_{treatment}_{plant}_D{day}.ply
```
Each PLY file = copy of v2 fields (positions, normals, is_leaf_tip, organ_label) + new `predicted_organ_label` int32 field.

---

## Implementation Steps

### Step 1 — PSegNet Wrapper

Import PSegNet model directly (add to sys.path):
```python
sys.path.insert(0, "/mnt/Data/jgajardo/code/PlantNet-and-PSegNet/PSegNet/PSegNet_pytorch/models")
sys.path.insert(0, "/mnt/Data/jgajardo/code/PlantNet-and-PSegNet/PSegNet/PSegNet_pytorch/utils")
import model_pytorch  # plantnet_model class
from clustering import MeanShiftTorch  # or sklearn MeanShift as fallback
```

**Function to implement:**
```python
def run_psegnet_inference(
    points: np.ndarray,          # (N, 3) original cloud
    model,                        # loaded plantnet_model
    device: str,
    n_points: int = 4096,
    bandwidth: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    # Returns: (predicted_labels_N, vfps_indices)
    #   predicted_labels_N: (N,) labels for ALL original points (via KNN backprop)
    #   vfps_indices: (4096,) indices into original cloud
```

**Internals:**
1. VFPS (voxel downsampling + FPS) to exactly 4096 pts — reuse logic from `Data_preprocessing/VFPS/003TXT2H5.py` or use Open3D voxel downsampling + FPS from PointNet++ utils
2. Normalize: center + scale to unit sphere (same as training)
3. Forward pass: `pred_sem, pred_ins, _ = model(pts_tensor)` → semantic argmax + instance features
4. MeanShift on instance features (bandwidth=0.6) → cluster IDs
5. Determine stem cluster: argmax of PSegNet semantic class 0 (or whichever class maps to stem — **verify with training data class definitions**)
6. Map: stem cluster → 0, other clusters → 1, 2, 3, ...
7. KNN back to original N pts: use existing `transfer_semantic_labels_knn` from `get_dense_gt_TrackPlant3D.py:107`

**Key file**: `/mnt/Data/jgajardo/code/PlantNet-and-PSegNet/PSegNet/PSegNet_pytorch/models/checkpoints/model_epoch199.pth`

### Step 2 — TrackPlant3D Tracking Wrapper

Import tracking components:
```python
sys.path.insert(0, "/mnt/Data/jgajardo/code/TrackPlant3D")
from downsampling.3DEPS_python import edge_separation_of_point_cloud, merge_edge_and_core, downsampling_of_point_cloud  # NOTE: filename has space — use importlib
sys.path.insert(0, "/mnt/Data/jgajardo/code/TrackPlant3D/registration")
from pycpd.deformable_registration import DeformableRegistration
sys.path.insert(0, "/mnt/Data/jgajardo/code/TrackPlant3D/tracking")
from utils import build_matrix, extend_matrix, assign_new_label, label_refresh, pointnet_norm
```

Note: `3DEPS(python).py` filename has a `(` in it — use `importlib.util.spec_from_file_location()` to import it.

**Function to implement:**
```python
def run_tracking_pipeline(
    sequence_points: list[np.ndarray],   # [(N_t, 3), ...] per timepoint
    sequence_labels: list[np.ndarray],   # [(N_t,), ...] per-point PSegNet labels
) -> list[np.ndarray]:
    # Returns: list of (N_t,) temporally consistent labels per timepoint
```

**Internals:**
1. For each timepoint: apply 3DEPS pipeline → 256 pts (`downsample_path` in memory)
2. For each consecutive pair (t-1, t): CPD registration
3. For each pair: Hungarian matching via `build_matrix` + `linear_sum_assignment` + `assign_new_label`
4. Map labels back to original resolution via `label_refresh()` (256→N_t mapping via KNN)

### Step 3 — Save New PLY Files

For each timepoint, load original v2 PLY, add `predicted_organ_label` field, save to new directory:
```python
pcd = o3d.t.io.read_point_cloud(str(v2_file_path))
pcd.point["predicted_organ_label"] = o3d.core.Tensor(
    predicted_labels.astype(np.int32).reshape(-1, 1)
)
o3d.t.io.write_point_cloud(str(output_path), pcd)
```

### Step 4 — Dataset Loader Extension

Add "v3" config to `DATASET_CONFIGS` in `trackplant3D.py` and extend `_load_timepoint_data()` to optionally load `predicted_organ_label`:

```python
DATASET_CONFIGS["v3"] = {
    "data_dirs": {"sparse": "predicted_labels", "dense": None},
    "orientation_corrected": True,
}
```

Or, simpler: keep v2 config and add a `load_predicted_labels: bool` flag that reads `predicted_organ_label` from the PLY file if it exists, returning it alongside `labels` in the timepoint dict as `"predicted_labels"`.

---

## Critical Files to Modify

| File | Change |
|------|--------|
| `src/plant_shape_analysis/dataloaders/trackplant3D.py` | Add v3 config or `load_predicted_labels` flag; surface `predicted_organ_label` in returned dict |
| `src/plant_shape_analysis/data_preprocessing/get_dense_gt_TrackPlant3D.py` | Reuse `transfer_semantic_labels_knn()` — no changes needed |

## Critical Files to Create

| File | Purpose |
|------|---------|
| `src/plant_shape_analysis/data_preprocessing/create_predicted_labels.py` | Main pipeline script |

---

## Things to Verify Before/During Implementation

1. **PSegNet class semantics**: Run `model_epoch199.pth` on the provided `test.h5` and inspect which of the 6 semantic classes dominates stem-region points. The class index for stem must be identified before building the stem→0 mapping.

2. **Coordinate system compatibility**: v2 PLY files are Z-up (orientation corrected). PSegNet was trained on data in its own coordinate frame. Test whether a Z-up cloud produces sensible segmentation or if a pre-rotation is needed.

3. **Maize generalization**: PSegNet was trained on tomato/tobacco/sorghum — not maize. Segmentation quality on maize may be lower; worth flagging in the paper.

4. **3DEPS importlib workaround**: The filename `3DEPS(python).py` cannot be imported with a normal `import` statement. Use:
   ```python
   spec = importlib.util.spec_from_file_location("deps3", "/mnt/.../downsampling/3DEPS(python).py")
   deps3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(deps3)
   ```

5. **Memory**: CPD on 256 pts is fast, but running it for all timepoint pairs across all species/plants at once may be slow. The script should process one sequence at a time and print progress.

---

## Verification

1. Run on a single maize sequence (`maize_control_plant1`) and visualize both `organ_label` and `predicted_organ_label` side-by-side using the existing notebook 12 visualization.
2. Check temporal consistency: inspect that leaf IDs don't swap between consecutive timepoints.
3. Compare instance counts (number of unique labels) between GT and predicted per sequence.
4. Load via `PlantSequencesDataset` v3 (or with `load_predicted_labels=True`) and confirm the predicted labels appear correctly in `get_sequence_data()` output.
