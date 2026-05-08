"""
PSegNet loss functions for training: discriminative loss + similarity matrix loss.

Ported from PlantNet-and-PSegNet/PSegNet/PSegNet_pytorch/utils/loss_pytorch.py
and models/model_pytorch.py (get_loss class).
"""

import torch
import torch.nn.functional as F


def _discriminative_loss_single(
    embeddings: torch.Tensor,
    inst_labels: torch.Tensor,
    feature_dim: int,
    delta_v: float,
    delta_d: float,
    param_var: float,
    param_dist: float,
    param_reg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Discriminative loss for one sample (no batch dimension)."""
    pred = embeddings.reshape(-1, feature_dim)
    unique_labels, unique_id, counts = torch.unique(
        inst_labels, return_inverse=True, return_counts=True
    )
    counts = counts.float()
    C = unique_labels.shape[0]

    segmented_sum = torch.zeros(C, feature_dim, device=pred.device)
    for i in range(C):
        segmented_sum[i] = pred[inst_labels == unique_labels[i]].sum(dim=0)
    mu = segmented_sum / counts.view(-1, 1)       # (C, D)
    mu_expand = mu[unique_id]                     # (N, D)

    # l_var: intra-cluster pull
    dist = torch.norm(pred - mu_expand, p=1, dim=1)
    dist = torch.clamp(dist - delta_v, min=0.0) ** 2
    l_var = torch.zeros(C, device=pred.device)
    for i in range(C):
        l_var[i] = dist[unique_id == i].sum() / counts[i]
    l_var = l_var.sum() / C

    # l_dist: inter-cluster push
    if C > 1:
        mu_i = mu.repeat(C, 1)
        mu_j = mu.repeat(1, C).view(C * C, feature_dim)
        off_diag = torch.eye(C, device=pred.device).eq(0).view(-1)
        mu_norm = torch.norm((mu_j - mu_i)[off_diag], p=1, dim=1)
        l_dist = torch.clamp(2.0 * delta_d - mu_norm, min=0.0).pow(2).mean()
    else:
        l_dist = pred.new_tensor(0.0)

    # l_reg: pull cluster centres toward origin
    l_reg = torch.norm(mu, p=1, dim=1).mean()

    loss = param_var * l_var + param_dist * l_dist + param_reg * l_reg
    return loss, param_var * l_var, param_dist * l_dist, param_reg * l_reg


def discriminative_loss(
    embeddings: torch.Tensor,
    inst_labels: torch.Tensor,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    param_var: float = 1.0,
    param_dist: float = 1.0,
    param_reg: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Discriminative loss over a batch.

    Args:
        embeddings:  (B, N, D) instance embedding vectors.
        inst_labels: (B, N) integer instance IDs (0 = stem, 1+ = leaves).

    Returns:
        (total_disc_loss, l_var, l_dist, l_reg) — all scalar tensors.
    """
    B, _, D = embeddings.shape
    losses, vars_, dists, regs = [], [], [], []
    for b in range(B):
        loss, lv, ld, lr = _discriminative_loss_single(
            embeddings[b], inst_labels[b], D,
            delta_v, delta_d, param_var, param_dist, param_reg,
        )
        losses.append(loss); vars_.append(lv); dists.append(ld); regs.append(lr)
    return (
        torch.stack(losses).mean(),
        torch.stack(vars_).mean(),
        torch.stack(dists).mean(),
        torch.stack(regs).mean(),
    )


def simmat_loss(
    simmat: torch.Tensor,
    sem_labels: torch.Tensor,
    inst_labels: torch.Tensor,
    alpha: float = 10.0,
    C_same: float = 10.0,
    C_diff: float = 80.0,
) -> torch.Tensor:
    """
    Double-hinge similarity matrix loss.

    For point pairs in the same instance: push simmat toward C_diff.
    For point pairs in different instances but same semantic class: push toward C_same.
    For point pairs in different semantic classes: push toward 0 (already penalised by diff class).

    Args:
        simmat:      (B, N, N) pairwise similarity scores (from SimmatModel).
        sem_labels:  (B, N) integer semantic class indices.
        inst_labels: (B, N) integer instance IDs.

    Returns:
        Scalar loss tensor.
    """
    # Boolean pairwise membership matrices
    same_inst = inst_labels.unsqueeze(2) == inst_labels.unsqueeze(1)    # (B, N, N)
    diff_inst = ~same_inst
    same_sem  = sem_labels.unsqueeze(2) == sem_labels.unsqueeze(1)      # (B, N, N)

    zero = simmat.new_tensor(0.0)
    # Same instance → encourage high similarity (pull toward C_diff)
    pos = same_inst.float() * torch.maximum(C_diff - simmat, zero)
    # Different instance, same semantic class → discourage high similarity (push toward C_same)
    neg_samesem = alpha * (diff_inst & same_sem).float() * torch.maximum(C_same - simmat, zero)
    # Different instance, different semantic class → penalise any similarity
    neg_diffsem = (diff_inst & ~same_sem).float() * simmat

    return (pos + neg_samesem + neg_diffsem).mean()


def psegnet_loss(
    sem_logits: torch.Tensor,
    inst_embed: torch.Tensor,
    simmat: torch.Tensor,
    sem_labels: torch.Tensor,
    inst_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined PSegNet training loss: 10·CE + 10·discriminative + 1·simmat.

    Args:
        sem_logits:  (B, N, num_classes) raw semantic logits.
        inst_embed:  (B, N, 5) instance embedding vectors.
        simmat:      (B, N, N) pairwise similarity matrix from SimmatModel.
        sem_labels:  (B, N) int64 semantic class indices (0=stem, 1=leaf).
        inst_labels: (B, N) int64 instance IDs (0=stem, 1+=individual leaves).

    Returns:
        (total_loss, ce_loss, disc_loss, sm_loss) — all scalar tensors.
    """
    ce = F.cross_entropy(sem_logits.transpose(1, 2), sem_labels)
    disc, _, _, _ = discriminative_loss(inst_embed, inst_labels)
    sm = simmat_loss(simmat, sem_labels, inst_labels)
    total = 10.0 * ce + 10.0 * disc + sm
    return total, ce, disc, sm
