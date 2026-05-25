import torch
from torch.nn import HuberLoss


class WeightedHuberLoss(HuberLoss):
    def __init__(self, delta=1.0, reduction="none", weight_by_frq=False):
        super().__init__(delta=delta, reduction=reduction)
        self.weight_by_frq = weight_by_frq

    def forward(self, input, target, frequency: torch.Tensor | None = None):
        loss = super().forward(input, target)
        if self.weight_by_frq and frequency is not None:
            loss = loss * 1/(torch.log1p(frequency))
        loss = loss.mean()
        return loss
