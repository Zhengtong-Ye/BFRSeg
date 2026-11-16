from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CEDiceLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.dice = smp.losses.DiceLoss(
            mode="multiclass",
            classes=[1],       # 只计算缺陷类别
            from_logits=True,
        )

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
        )