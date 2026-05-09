"""
Train PSegNet from scratch on PlantSequencesDataset v2.

Usage:
    conda run -n plant-shape-analysis python scripts/train_psegnet.py \
        --dataset_path data/TrackPlant3D/versions \
        --output_dir outputs/psegnet_retrain

Key flags:
    --no_wandb          Disable Weights & Biases logging
    --batch_size 4      Start here; increase if VRAM allows (simmat is B×N×N)
    --resume PATH       Resume from a checkpoint .pth file
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from plant_shape_analysis.segmentation.psegnet.model import PSegNet
from plant_shape_analysis.segmentation.psegnet.loss import (
    psegnet_loss,
    discriminative_loss,
)
from plant_shape_analysis.segmentation.psegnet.dataset import PSegNetDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Train PSegNet on TrackPlant3D v2")
    p.add_argument("--dataset_path", type=str, default="data/TrackPlant3D/versions")
    p.add_argument("--output_dir", type=str, default="outputs/psegnet_retrain")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Keep at 4 unless you have >24 GB VRAM (simmat is B×N×N)",
    )
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Decay LR and BN momentum every this many epochs",
    )
    p.add_argument("--lr_decay", type=float, default=0.7)
    p.add_argument("--n_points", type=int, default=4096)
    p.add_argument(
        "--n_repeats",
        type=int,
        default=10,
        help="FPS repeats per frame for augmentation (paper uses 10)",
    )
    p.add_argument(
        "--num_classes", type=int, default=2, help="Semantic classes: 2 = stem vs leaf"
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers; FPS is pure-Python so use ≥4",
    )
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument(
        "--save_every",
        type=int,
        default=20,
        help="Save a checkpoint every N epochs (plus always save best)",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth to resume from",
    )
    p.add_argument(
        "--use_all_sequences",
        action="store_true",
        help="Include held-out test sequences in training. Use for a final model after benchmarking is complete.",
    )
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    p.add_argument("--wandb_project", type=str, default="psegnet-trackplant3d")
    return p.parse_args()


def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def set_bn_momentum(model, momentum):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = momentum


def semantic_iou(pred_logits, sem_labels, num_classes):
    """Mean IoU over semantic classes (numpy, no-grad)."""
    pred = pred_logits.argmax(dim=2).cpu().numpy().ravel()
    gt = sem_labels.cpu().numpy().ravel()
    ious = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (gt == c))
        denom = np.sum((pred == c) | (gt == c))
        if denom > 0:
            ious.append(tp / denom)
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = time.strftime("%Y%m%d_%H%M")
    out_dir = Path(args.output_dir) / timestamp
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "train_val_split.json"

    # Save run config for reproducibility
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ------------------------------------------------------------------
    # W&B (optional)
    # ------------------------------------------------------------------
    run = None
    if not args.no_wandb:
        try:
            import wandb

            run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"psegnet_{timestamp}",
                dir=str(out_dir),
            )
        except ImportError:
            print("wandb not installed, continuing without logging")

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    print("Loading datasets …")
    train_ds = PSegNetDataset(
        dataset_path=args.dataset_path,
        split="train",
        n_points=args.n_points,
        n_repeats=args.n_repeats,
        val_fraction=args.val_fraction,
        split_save_path=split_path,
        augment=True,
        use_all_sequences=args.use_all_sequences,
    )
    val_ds = PSegNetDataset(
        dataset_path=args.dataset_path,
        split="val",
        n_points=args.n_points,
        n_repeats=1,  # one sample per frame is enough for validation
        val_fraction=args.val_fraction,
        split_save_path=split_path,
        augment=False,
        use_all_sequences=args.use_all_sequences,
    )
    print(f"  train: {len(train_ds)} frames   val: {len(val_ds)} frames")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda wid: np.random.seed(wid + int(time.time())),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PSegNet(num_classes=args.num_classes).to(device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']} ({args.resume})")
    else:
        model.apply(weights_init)
        print("Initialised model from scratch (Xavier normal)")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_iou = 0.0
    MOMENTUM_INIT = 0.1
    LR_MIN = 1e-6

    for epoch in range(start_epoch, args.epochs):
        # LR + BN momentum decay
        lr = max(args.lr * (args.lr_decay ** (epoch // args.step_size)), LR_MIN)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        bn_mom = max(MOMENTUM_INIT * (0.5 ** (epoch // args.step_size)), 0.01)
        set_bn_momentum(model, bn_mom)

        # --- Train ---
        model.train()
        t0 = time.time()
        train_totals = {"loss": 0.0, "ce": 0.0, "disc": 0.0, "sm": 0.0}
        n_batches = 0

        for batch in train_loader:
            pts = batch["points"].to(device)  # (B, N, 3)
            sem_lbl = batch["sem_labels"].to(device)  # (B, N)
            inst_lbl = batch["inst_labels"].to(device)  # (B, N)

            optimizer.zero_grad()
            sem_logits, inst_embed, simmat = model(pts)
            total, ce, disc, sm = psegnet_loss(
                sem_logits, inst_embed, simmat, sem_lbl, inst_lbl
            )
            total.backward()
            optimizer.step()

            train_totals["loss"] += total.item()
            train_totals["ce"] += ce.item()
            train_totals["disc"] += disc.item()
            train_totals["sm"] += sm.item()
            n_batches += 1

        train_means = {k: v / n_batches for k, v in train_totals.items()}
        elapsed = time.time() - t0

        # --- Validate ---
        model.eval()
        val_totals = {"loss": 0.0, "ce": 0.0, "disc": 0.0, "sm": 0.0}
        val_ious = []
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                pts = batch["points"].to(device)
                sem_lbl = batch["sem_labels"].to(device)
                inst_lbl = batch["inst_labels"].to(device)

                sem_logits, inst_embed, simmat = model(pts)
                total, ce, disc, sm = psegnet_loss(
                    sem_logits, inst_embed, simmat, sem_lbl, inst_lbl
                )
                val_totals["loss"] += total.item()
                val_totals["ce"] += ce.item()
                val_totals["disc"] += disc.item()
                val_totals["sm"] += sm.item()
                val_ious.append(semantic_iou(sem_logits, sem_lbl, args.num_classes))
                n_val += 1

        val_means = {k: v / max(n_val, 1) for k, v in val_totals.items()}
        val_iou = float(np.mean(val_ious))

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"lr={lr:.2e} | {elapsed:.0f}s | "
            f"train loss={train_means['loss']:.3f} (ce={train_means['ce']:.3f} "
            f"disc={train_means['disc']:.3f} sm={train_means['sm']:.3f}) | "
            f"val loss={val_means['loss']:.3f} mIoU={val_iou:.4f}"
        )

        if run is not None:
            run.log(
                {
                    "lr": lr,
                    "train/loss": train_means["loss"],
                    "train/ce": train_means["ce"],
                    "train/disc": train_means["disc"],
                    "train/sm": train_means["sm"],
                    "val/loss": val_means["loss"],
                    "val/ce": val_means["ce"],
                    "val/disc": val_means["disc"],
                    "val/sm": val_means["sm"],
                    "val/mIoU": val_iou,
                },
                step=epoch + 1,
            )

        # --- Checkpoints ---
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_iou": val_iou,
            "args": vars(args),
        }
        if (epoch + 1) % args.save_every == 0:
            torch.save(state, ckpt_dir / f"epoch_{epoch+1:04d}.pth")
        if val_iou >= best_val_iou:
            best_val_iou = val_iou
            torch.save(state, ckpt_dir / "best_model.pth")
            print(f"  → new best val mIoU: {best_val_iou:.4f}")

    print(f"\nTraining complete. Best val mIoU: {best_val_iou:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'best_model.pth'}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
