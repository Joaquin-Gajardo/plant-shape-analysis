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

→ **Implemented as Option B.** Confirmed with user. v2 data loaded via `PlantSequencesDataset(version='v2')` has `orientation_corrected=True`, meaning all crops are already Z-up. No pre-rotation needed at inference for the retrained checkpoint — pass `pre_rotation=None` (the default) to `predict_organ_labels()`.

---

## Implementation Plan (completed)

### Step 1 — Parameterise `num_classes` in model.py ✅

`src/plant_shape_analysis/segmentation/psegnet/model.py`

- Change `NUM_CLASSES = 6` module constant → `num_classes` constructor argument (default kept at 6 for backward compat with old checkpoints)
- CONV10 output size becomes `num_classes`
- `load_psegnet()` in `inference.py`: expose `num_classes` param (default 6 for old checkpoint, pass 2 for new)

### Step 2 — Port loss functions ✅

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

### Step 3 — Training dataset ✅

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
- **FPS repeated sampling for augmentation**: following the paper, `n_repeats=10` (default) generates 10 different FPS subsamples per frame per epoch, expanding 255 train frames → ~2550 effective samples/epoch (paper had 3640 from 364 frames × 10). The random starting point of `_fps_numpy` produces genuinely different 4096-pt subsets each call.
- ⚠️ **FPS bottleneck**: `_fps_numpy` is O(N²) pure Python (few seconds per sample for N~10k). Use `num_workers≥4` in DataLoader, or pre-cache FPS indices with a fixed seed per (sequence, timepoint) sample.

### Step 4 — Training script ✅

New file: `scripts/train_psegnet.py`

Hyperparameters (matching original paper where possible):
- Optimizer: Adam, lr=0.003, weight_decay=1e-3
- LR schedule: ×0.7 every 10 epochs, min 1e-6
- BN momentum: start 0.1, ×0.5 every 10 epochs, min 0.01
- Epochs: 200, batch_size: 4 default (conservative given simmat memory; paper used 8 on RTX 2080Ti 11GB)
- num_classes: 2, n_input_points: 4096, n_repeats: 10

Checkpoint structure (compatible with `load_psegnet`):
```python
{"epoch": e, "model_state_dict": ..., "optimizer_state_dict": ..., "val_iou": ..., "args": ...}
```

Save: every 5 epochs + best model (by validation semantic IoU).

Output directory includes a timestamp subfolder: `--output_dir/YYYYMMDD_HHMM/` containing checkpoints, config.json, train_val_split.json, and the W&B `wandb/` directory.

Logging: W&B — log train/val loss components separately (CE, disc, simmat) plus per-epoch semantic IoU. Use `step=epoch+1` (not a key in the metrics dict) to get correct x-axis in W&B charts.

Argparse flags: `--dataset_path`, `--output_dir`, `--epochs`, `--batch_size`, `--lr`, `--n_points`, `--n_repeats`, `--num_classes`, `--resume`, `--wandb_project`, `--no_wandb`.

### Step 5 — Update inference for the retrained checkpoint ✅

`src/plant_shape_analysis/segmentation/psegnet/inference.py` already had `num_classes` and `pre_rotation` params — no changes needed. For the retrained checkpoint, callers pass `num_classes=2` to `load_psegnet()` and omit `pre_rotation` (defaults to `None`).

`scripts/create_predicted_labels_v3.py` updated with:
- `--num-classes` (default 6 for old checkpoint, pass 2 for retrained)
- `--no-prerotation` flag (pass for retrained checkpoint — it was trained on Z-up data)
- `--output-dir` to override the default output path (useful for testing)
- `--sequences` to run only on specific sequence names (overrides `--species`)

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

## Verified During Implementation

1. **Smoke test** ✅ — 2 epochs, batch_size=4, GPU:
   ```
   Epoch 1/2 | train loss=92.6 (ce=0.571 disc=2.129 sm=65.6) | val loss=57.9 mIoU=0.378
   Epoch 2/2 | train loss=60.8 (ce=0.489 disc=1.221 sm=43.7) | val loss=69.0 mIoU=0.374
   ```
   All three loss components decreasing. ~62s/epoch on GPU → full 200-epoch run ≈ 3.5h.

2. **Maize inference at epoch ~90** ✅ — leaf counts match GT well:
   - `maize_control_plant1`: predicted 2→3→4→4→4→5 leaves (GT max: 4) ✓
   - `maize_control_plant2`: predicted 4→5→5→5→6→6 leaves (GT max: 5) ✓
   - `maize_control_plant3`: predicted 2→3→4→4→4 leaves (GT max: 4) ✓

3. **`num_classes=2` is architecturally sound** ✅ — confirmed by reading original implementation:
   - The original used 6 classes = 3 species × 2 organs (stem + leaf per species). For our dataset (annotations are instance IDs, not per-species organs), binary stem/leaf is the correct semantic split.
   - Only `CONV10` (`Conv1d(128 → num_classes)`) changes size — CE loss, simmat loss, and discriminative loss are all num_classes-agnostic.
   - The simmat loss computes same/different semantic class via matrix multiplication on one-hot labels (original) or direct integer equality (our implementation) — equivalent, and independent of the number of classes.

4. **Instance count is unbounded at inference** ✅ — the instance head outputs 5D embeddings, and MeanShift clusters them into however many groups the data supports. The `num_classes` parameter only controls the semantic head, not instance count. A plant with 20 leaves would work fine.

## Commands

**Train from scratch:**
```bash
conda run -n plant-shape-analysis python scripts/train_psegnet.py --dataset_path data/TrackPlant3D/versions --output_dir outputs/psegnet_retrain --epochs 200 --batch_size 4 --no_wandb
```

**Resume from checkpoint:**
```bash
conda run -n plant-shape-analysis python scripts/train_psegnet.py --dataset_path data/TrackPlant3D/versions --output_dir outputs/psegnet_retrain --epochs 200 --batch_size 4 --resume outputs/psegnet_retrain/YYYYMMDD_HHMM/checkpoints/best_model.pth --no_wandb
```

**Run full inference pipeline with retrained checkpoint:**
```bash
conda run -n plant-shape-analysis python scripts/create_predicted_labels_v3.py --checkpoint outputs/psegnet_retrain/YYYYMMDD_HHMM/checkpoints/best_model.pth --num-classes 2 --no-prerotation
```

**Quick inference check on specific sequences:**
```bash
conda run -n plant-shape-analysis python scripts/create_predicted_labels_v3.py --checkpoint PATH --num-classes 2 --no-prerotation --output-dir /tmp/test --sequences maize_control_plant1 maize_control_plant2
```

**Load retrained model in Python:**
```python
from plant_shape_analysis.segmentation import load_psegnet, predict_organ_labels

model = load_psegnet("outputs/psegnet_retrain/.../best_model.pth", num_classes=2)
labels = predict_organ_labels(points, model)  # pre_rotation=None by default (Z-up checkpoint)
```
