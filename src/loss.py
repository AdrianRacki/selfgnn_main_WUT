import torch
from typing import Optional


def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    _lambda: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes the Barlow Twins loss between two sets of embeddings.

    Args:
        z_a (torch.Tensor): Embeddings from the first view.
        z_b (torch.Tensor): Embeddings from the second view.
        _lambda (Optional[float]): Regularization parameter.

    Returns:
        torch.Tensor: The computed Barlow Twins loss.
    """
    EPS = 1e-15

    batch_size = z_a.size(0)
    feature_dim = z_a.size(1)
    if _lambda is None:
        _lambda = 1 / feature_dim

    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size

    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (1 - c.diagonal()).pow(2).sum()
    loss += _lambda * c[off_diagonal_mask].pow(2).sum()

    return loss
