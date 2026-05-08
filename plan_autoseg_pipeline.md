# Plan: Predicted Organ Labels Pipeline (v3 Dataset)

## Context

The GT labels in `PlantSequencesDataset v2` come from manually corrected, temporally consistent organ IDs. To benchmark downstream analysis sensitivity to label quality, we need a second annotation produced by running PSegNet (segmentation) + TrackPlant3D tracking (temporal consistency) automatically. The output is a v3 dataset directory where each PLY file adds a `predicted_organ_label` int32 field, with the dataset class optionally loading it.

---

## Package Architecture

Two new subpackages added to `plant_shape_analysis`, each with a method-specific subdirectory (to allow future methods to coexist), plus a dataset creation script:

```
src/plant_shape_analysis/
├── segmentation/                     ← NEW subpackage
│   ├── __init__.py                   ← re-exports public API
│   └── psegnet/                      ← one subdir per method (future: plantnet/, etc.)
│       ├── __init__.py
│       ├── model.py                  ← PSegNet architecture (adapted from PlantNet-and-PSegNet repo)
│       ├── pointnet2_utils.py        ← PointNet++ SA/FP layers (adapted from same repo)
│       └── inference.py             ← Clean inference API: load model, run on point cloud
│
├── tracking/                         ← NEW subpackage
│   ├── __init__.py                   ← re-exports public API
│   └── trackplant3d/                 ← one subdir per method (future: other_tracker/, etc.)
│       ├── __init__.py
│       ├── downsampling.py           ← 3DEPS boundary-preserving sampling (adapted from TrackPlant3D)
│       ├── organ_matching.py         ← Hungarian matching + cost matrix (adapted from TrackPlant3D)
│       └── pipeline.py              ← High-level: run full tracking on a sequence

scripts/
└── create_predicted_labels_v3.py     ← NEW: uses both modules to produce v3 dataset
```

"Adapted" means the core algorithmic code is ported into the package as proper Python modules — no `sys.path` hacks, no subprocess calls. Only the relevant functions are taken (not the CLI boilerplate).

The top-level `segmentation/__init__.py` and `tracking/__init__.py` re-export the key public functions so callers can write `from plant_shape_analysis.segmentation import predict_organ_labels` without knowing the method subdir.

---

## Dependencies to Add in `pyproject.toml`

```toml
[project.optional-dependencies]
predicted-labels = [
    "scikit-learn",  # MeanShift clustering for PSegNet instance segmentation
    "scipy",         # cKDTree for KNN label propagation, linear_sum_assignment for tracking
]
```

`torch` and `open3d` are already core dependencies. `pycpd` is **not on PyPI** (the PyPI package is unrelated); the TrackPlant3D repo copy was bundled directly into `tracking/trackplant3d/_cpd/` as a private subpackage.

---

## Module Details

### `segmentation/psegnet/model.py`
Adapt `PlantNet-and-PSegNet/PSegNet/PSegNet_pytorch/models/model_pytorch.py`:
- Copy `plantnet_model(num_classes=6)` class verbatim (or with minor cleanups)
- Keep `forward()` signature: `(B, N, 3) → (sem_logits B×N×6, ins_embed B×N×5, simmat B×N×N)`

### `segmentation/psegnet/pointnet2_utils.py`
Adapt `PlantNet-and-PSegNet/PSegNet/PSegNet_pytorch/utils/pointnet2_util_pytorch.py`:
- Copy SA (Set Abstraction) and FP (Feature Propagation) layers
- Copy FPS utility

### `segmentation/psegnet/inference.py`
Clean public API (also re-exported by `segmentation/__init__.py`):
```python
def load_psegnet(checkpoint_path: str | Path, device: str = "cuda") -> nn.Module:
    ...

def predict_organ_labels(
    points: np.ndarray,              # (N, 3) — original full-resolution cloud
    model: nn.Module,
    device: str = "cuda",
    n_input_points: int = 4096,
    bandwidth: float = 0.6,
    stem_semantic_class: int = 0,    # verified: class 0 = stem for model_epoch199.pth
    pre_rotation: np.ndarray | None = None,  # optional (3,3) rotation applied before inference
) -> np.ndarray:                     # (N,) predicted organ_label (0=stem, 1,2,3...=leaves)
```

Internal steps:
1. Apply `pre_rotation` if provided (see coordinate system note below)
2. FPS to 4096 pts (pure numpy greedy FPS)
3. Normalize (center + unit sphere) — same as training
4. Forward pass → semantic argmax + MeanShift on instance embeddings
5. Map: stem semantic class → label 0; other clusters → 1, 2, 3, ...
6. KNN back-propagation to full N pts via `cKDTree`

**Coordinate system (important):** The checkpoint `model_epoch199.pth` was trained on Y-up point clouds. v2 PLYs for sorghum/tobacco/tomato1 have been orientation-corrected to Z-up. Feeding Z-up data directly collapses nearly all predictions to a single semantic class. Fix: pass `pre_rotation = R_ZUP_TO_YUP` (= `[[1,0,0],[0,0,1],[0,-1,0]]`) for those species. `create_predicted_labels_v3.py` does this automatically via `needs_orientation_correction(sequence_name)`. maize and tomato2 are natively Z-up — no rotation needed.

### `tracking/trackplant3d/downsampling.py`
Adapt `TrackPlant3D/downsampling/3DEPS(python).py`:
- Port `edge_separation_of_point_cloud()`, `merge_edge_and_core()`, `downsampling_of_point_cloud()` as pure functions operating on numpy arrays (not filesystem paths)
- Strip all the I/O and directory-management code
- Keep the 3DEPS algorithm: boundary/core split → merge 4096+4096 → FPS → 256 pts

Key function:
```python
def downsample_3deps(points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Returns: (downsampled_points (256, 3), downsampled_labels (256,))
```

### `tracking/trackplant3d/organ_matching.py`
Adapt `TrackPlant3D/tracking/utils.py` and `organ_matching.py`:
- Port `build_matrix()`, `extend_matrix()`, `assign_new_label()`, `label_refresh()`
- Port `chamfer_distance_numpy()`, `pointnet_norm()`

Key function:
```python
def match_organs(
    points_prev: np.ndarray,   # (256, 3) normalized downsampled cloud at t-1
    labels_prev: np.ndarray,   # (256,) tracked labels from t-1
    points_curr: np.ndarray,   # (256, 3) CPD-registered cloud at t
    labels_curr: np.ndarray,   # (256,) raw predicted labels at t
    start_label: int,
) -> tuple[np.ndarray, int]:   # (new_labels (256,), updated start_label)
```

### `tracking/trackplant3d/pipeline.py`
High-level API (also re-exported by `tracking/__init__.py`):
```python
def run_tracking_pipeline(
    sequence_points: list[np.ndarray],   # [(N_t, 3), ...] per timepoint (full res)
    sequence_labels: list[np.ndarray],   # [(N_t,), ...] per-point PSegNet predictions
) -> list[np.ndarray]:                   # [(N_t,) temporally consistent labels, ...]
```

Internal flow per sequence:
1. For each timepoint: `downsample_3deps()` → 256 pts
2. For consecutive pairs (t-1, t): CPD registration via `pycpd.DeformableRegistration`
3. For each pair: `match_organs()` → consistent labels at 256-pt resolution
4. Back-propagate to full resolution: `transfer_semantic_labels_knn()` (already in codebase)

---

## Dataset Loader Change

In `src/plant_shape_analysis/dataloaders/trackplant3D.py`, add a v3 entry to `DATASET_CONFIGS`:

```python
DATASET_CONFIGS["v3"] = {
    "data_dirs": {"sparse": "predicted_labels", "dense": None},
    "orientation_corrected": True,
}
```

The `_load_timepoint_data()` function already reads PLY fields generically — if the PLY contains `predicted_organ_label`, it should be surfaced in the returned dict. This likely requires a small addition to load and return it as `"predicted_labels"` alongside `"labels"` (the GT).

---

## Script: `scripts/create_predicted_labels_v3.py`

CLI script that wires everything together:
```
usage: create_predicted_labels_v3.py [--checkpoint PATH] [--dataset-path data/TrackPlant3D/versions]
                                     [--device cuda] [--species maize tomato ...]
```

Per sequence:
1. Load sequence from `PlantSequencesDataset v2`
2. For sorghum/tobacco/tomato1: apply `R_ZUP_TO_YUP` before PSegNet inference
3. Run `predict_organ_labels()` on each timepoint
4. Run `run_tracking_pipeline()` on the sequence
5. For each timepoint: load v2 PLY, add `predicted_organ_label` field, write to `versions/v2/predicted_labels/{crop}/`

Species filtering uses `seq_name.startswith(species)` — needed because sequence names are `tomato1_*` / `tomato2_*`, not `tomato_*`.

Output directory:
```
data/TrackPlant3D/versions/v2/predicted_labels/{crop}/{same_filename_as_v2}.ply
```

Default `--dataset-path` is `data/TrackPlant3D/versions` (not `data/TrackPlant3D`).

---

## Things Verified During Implementation

1. **PSegNet stem class index**: ✅ Class 0 = stem confirmed for `model_epoch199.pth` by cross-tabulating predicted semantic classes against GT organ labels on tomato1/sorghum sequences (with correct Y-up input).

2. **Coordinate system**: ✅ Resolved — see note in `inference.py` section above. Z-up input collapses predictions; Y-up input via `pre_rotation` fixes it for sorghum/tobacco/tomato1.

3. **Maize generalization**: ✅ Confirmed OOD — stem (class 0) is not detected for maize. Leaves are still segmented into instances but without a stem label. Worth noting in the paper.

4. **`pycpd` PyPI version**: ✅ PyPI `pycpd` is unrelated; bundled TrackPlant3D's copy as `tracking/trackplant3d/_cpd/` private subpackage instead.

5. **Models subpackage**: ✅ `segmentation/` is correctly separate from `models/`.

---

## Verification

1. Run script on one maize sequence. Visualize both `organ_label` and `predicted_organ_label` side-by-side in notebook 12.
2. Check temporal consistency: leaf IDs should not swap between consecutive timepoints.
3. Compare unique label counts (GT vs predicted) per sequence.
4. Load via `PlantSequencesDataset("v3")` and confirm `predicted_labels` appears in `get_sequence_data()` output.
