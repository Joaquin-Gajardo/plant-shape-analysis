# Plan: Extend plant-shape-analysis to Support PSegNet Training

## Context

The current implementation added PSegNet for *inference only* (loss functions stripped, forward pass retained). The user wants to retrain PSegNet from scratch on all v2 PlantSequencesDataset sequences except 8 held-out test sequences, so that the model generalises to maize (currently out-of-distribution).

**Key data facts from v2 inspection:**
- 43 total sequences; 35 training sequences (43 − 8 held-out)
- Instance labels: 0 = stem, 1…N = individual leaf IDs (sequential per plant, max 13 in dataset)
- Semantic labels: derived as 0 (stem) vs 1 (leaf) — binary, no richer annotation exists
- → `NUM_CLASSES = 2` is correct for this dataset
- Total training timepoints ≈ 280–300 individual point-cloud frames

**8 held-out test sequences (excluded from training):**
`maize_control_plant2`, `maize_control_plant3`, `sorghum_control_plant2`,
`tobacco_control_plant1`, `tobacco_shade_plant3`, `tomato1_heat_plant3`,
`tomato1_shade_plant1`, `sorghum_highlight_plant2`

**Training mode:** retrain from scratch.

---

## Design Decision: Coordinate System (important — affects inference)

The v2 PLY files are Z-up for most crops (sorghum, tobacco, tomato1). The current inference pipeline applies a Y→Z rotation before feeding to the old checkpoint.

**Option A — Rotate training data to Y-up:** Mirror the inference pre-rotation, keeping consistency with the original model's training distribution. Inference stays as-is.

**Option B — Train entirely on Z-up data (recommended):** Since we're retraining from scratch on our own data anyway, PointNet++ is not rotation-invariant, so training Z-up means the retrained model works directly on v2 PLYs. Inference becomes simpler: remove the pre-rotation step in `predict_organ_labels` for the new checkpoint (or add a `zup=True` flag). Cleaner overall.

→ **Plan assumes Option B.** Need to confirm with user before implementation.

---

## Implementation Plan

### Step 1 — Parameterise `num_classes` in model.py

`src/plant_shape_analysis/segmentation/psegnet/model.py`

- Change `NUM_CLASSES = 6` module constant → `num_classes` constructor argument (default kept at 6 for backward compat with old checkpoints)
- CONV10 output size becomes `num_classes`
- `load_psegnet()` in `inference.py`: expose `num_classes` param (default 6 for old checkpoint, pass 2 for new)

### Step 2 — Port loss functions

New file: `src/plant_shape_analysis/segmentation/psegnet/loss.py`

Port directly from `/mnt/Data/jgajardo/code/PlantNet-and-PSegNet/PSegNet/PSegNet_pytorch/utils/loss_pytorch.py`:

- **`discriminative_loss(embeddings, instance_labels, ...)`**  
  Intra-cluster variance (delta_v=0.5) + inter-cluster distance (delta_d=1.5) + regularisation (1e-3).  
  Takes (B, N, 5) embeddings + (B, N) instance labels.

- **`simmat_loss(simmat, instance_labels, alpha=10, C_same=10, C_diff=80)`**  
  Double-hinge loss on pairwise similarity matrix vs same/different instance ground truth.  
  ⚠️ **Memory warning**: at B=8, N=4096 the simmat is 8×4096×4096×4 bytes ≈ 512 MB forward + ~512 MB gradients. Start with `batch_size=4` or `n_input_points=2048` and scale up if VRAM allows.

- **`psegnet_loss(sem_logits, inst_emb, simmat, sem_labels, inst_labels)`**  
  Combined: `10 × CE_loss + 10 × disc_loss + 1 × simmat_loss`

### Step 3 — Training dataset

New file: `src/plant_shape_analysis/segmentation/psegnet/dataset.py`

```python
class PSegNetDataset(torch.utils.data.Dataset):
    TEST_SEQUENCES = [
        "maize_control_plant2", "maize_control_plant3",
        "sorghum_control_plant2", "sorghum_highlight_plant2",
        "tobacco_control_plant1", "tobacco_shade_plant3",
        "tomato1_heat_plant3", "tomato1_shade_plant1",
    ]
```

- Wraps `PlantSequencesDataset(version='v2')`, flattens to individual (sequence, timepoint) samples, excludes TEST_SEQUENCES
- **Train/val split at timepoint level** (not sequence level): all ~280–300 frames are pooled then split ~85/15 with `sklearn.model_selection.train_test_split(stratify=species)`. This is essential since maize has only 1 remaining training sequence (2 of 3 are held out) — reserving a whole sequence for val would lose all maize training data.
  - `species` label derived from the sequence name prefix (maize/sorghum/tobacco/tomato)
  - Fixed `random_state=42` for reproducibility
  - Split indices saved to disk alongside checkpoints for traceability
- Constructor param `split='train'|'val'|'all'` selects the appropriate subset
- **Coordinate system (Option B):** load raw Z-up data — no rotation applied
- `__getitem__` returns:
  - `points`: (4096, 3) float32 — FPS-downsampled + PointNet-normalised (reuse `_fps_numpy`, `_pointnet_norm` from `inference.py`)
  - `sem_labels`: (4096,) int64 — `(inst_labels > 0).astype(int)`
  - `inst_labels`: (4096,) int64 — original organ label after FPS index selection
- Augmentation (train split only): random Z-rotation, Gaussian jitter (σ=0.01, clip 0.05)
- ⚠️ **FPS bottleneck**: `_fps_numpy` is O(N²) pure Python (few seconds per sample for N~10k). Use `num_workers≥4` in DataLoader, or pre-cache FPS indices with a fixed seed per (sequence, timepoint) sample.

### Step 4 — Training script

New file: `scripts/train_psegnet.py`

Hyperparameters:
- Optimizer: Adam, lr=0.003, weight_decay=1e-3
- LR schedule: ×0.7 every 10 epochs, min 1e-6
- BN momentum: start 0.1, ×0.5 every 10 epochs, min 0.01
- Epochs: 200, batch_size: 4 (conservative given simmat memory; increase if VRAM allows)
- num_classes: 2, n_input_points: 4096

Checkpoint structure (compatible with `load_psegnet`):
```python
{"epoch": e, "model_state_dict": ..., "optimizer_state_dict": ...}
```

Save: every 5 epochs + best model (by validation semantic IoU).

Logging: W&B (already in pyproject.toml full deps) — log train/val loss components separately (CE, disc, simmat) plus per-epoch semantic IoU.

Argparse flags: `--dataset_path`, `--output_dir`, `--epochs`, `--batch_size`, `--lr`, `--n_points`, `--num_classes`, `--resume`, `--wandb_project`.

### Step 5 — Update inference for the retrained checkpoint

`src/plant_shape_analysis/segmentation/psegnet/inference.py`

- `load_psegnet()`: add `num_classes=6` param (pass 2 for new checkpoint)
- `predict_organ_labels()`: add `apply_prerotation=True` param. For the new Z-up checkpoint, callers pass `apply_prerotation=False` to skip the Y→Z rotation that was needed only for the old checkpoint.

---

## Critical Files

| File | Role |
|------|------|
| `src/plant_shape_analysis/segmentation/psegnet/model.py` | Parameterise num_classes |
| `src/plant_shape_analysis/segmentation/psegnet/inference.py` | Expose num_classes + prerotation flag |
| `src/plant_shape_analysis/segmentation/psegnet/loss.py` | New — ported from original repo |
| `src/plant_shape_analysis/segmentation/psegnet/dataset.py` | New — PSegNetDataset |
| `scripts/train_psegnet.py` | New — training entry point |
| `/mnt/Data/jgajardo/code/PlantNet-and-PSegNet/PSegNet/PSegNet_pytorch/utils/loss_pytorch.py` | Source for loss port |
| `/mnt/Data/jgajardo/code/PlantNet-and-PSegNet/PSegNet/PSegNet_pytorch/models/01train.py` | Reference for training loop |

---

## Verification

1. **Smoke test (CPU, 1 batch):**
   ```bash
   conda run -n plant-shape-analysis python scripts/train_psegnet.py \
     --epochs 1 --batch_size 2 --dataset_path data/TrackPlant3D --no_wandb
   ```
   Verify loss is finite and backward pass completes without OOM.

2. **Overfit test:** train 20 epochs on a single sequence — CE loss should drop toward 0, disc loss should decrease.

3. **Full training run:** 200 epochs on 35 training sequences, monitor val IoU curve.

4. **Evaluation:** Run `predict_organ_labels(..., apply_prerotation=False)` with new checkpoint on the 8 held-out test sequences (including maize), inspect in existing notebooks.
