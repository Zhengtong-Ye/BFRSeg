from __future__ import annotations

import numpy as np
import torch

from common import CLASS_NAMES, NUM_CLASSES


class SegmentationMeter:
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        self.num_classes = num_classes

        if len(CLASS_NAMES) != self.num_classes:
            raise ValueError(
                f"CLASS_NAMES数量={len(CLASS_NAMES)}，"
                f"但num_classes={self.num_classes}"
            )

        self.confusion = np.zeros(
            (num_classes, num_classes),
            dtype=np.int64,
        )

    def update(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
    ) -> None:
        if logits.ndim != 4:
            raise ValueError(
                f"logits应为[B,C,H,W]，实际为{logits.shape}"
            )

        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"模型输出通道数为{logits.shape[1]}，"
                f"但num_classes={self.num_classes}"
            )

        if targets.ndim != 3:
            raise ValueError(
                f"targets应为[B,H,W]，实际为{targets.shape}"
            )

        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                "logits和targets的batch size不一致"
            )

        if logits.shape[2:] != targets.shape[1:]:
            raise ValueError(
                f"logits尺寸={logits.shape[2:]}，"
                f"targets尺寸={targets.shape[1:]}"
            )

        predictions = logits.argmax(dim=1)

        predictions = (
            predictions.detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        targets = (
            targets.detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        valid = (
                (targets >= 0)
                & (targets < self.num_classes)
                & (predictions >= 0)
                & (predictions < self.num_classes)
        )

        indices = (
                self.num_classes * targets[valid]
                + predictions[valid]
        )

        matrix = np.bincount(
            indices,
            minlength=self.num_classes ** 2,
        ).reshape(
            self.num_classes,
            self.num_classes,
        )

        self.confusion += matrix

    @staticmethod
    def _safe_divide(
        numerator: np.ndarray,
        denominator: np.ndarray,
    ) -> np.ndarray:
        result = np.full_like(
            numerator,
            np.nan,
            dtype=np.float64,
        )
        np.divide(
            numerator,
            denominator,
            out=result,
            where=denominator != 0,
        )
        return result

    def compute(self) -> dict:
        matrix = self.confusion.astype(np.float64)
        tp = np.diag(matrix)
        gt = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)

        fp = predicted - tp
        fn = gt - tp

        iou = self._safe_divide(tp, tp + fp + fn)
        dice = self._safe_divide(2 * tp, 2 * tp + fp + fn)
        precision = self._safe_divide(tp, tp + fp)
        recall = self._safe_divide(tp, tp + fn)

        total = matrix.sum()
        pixel_accuracy = float(tp.sum() / total) if total > 0 else float("nan")

        foreground = slice(1, self.num_classes)

        result = {
            "pixel_accuracy": pixel_accuracy,
            "mIoU_all": float(np.nanmean(iou)),
            "mIoU_fg": float(np.nanmean(iou[foreground])),
            "mDice_all": float(np.nanmean(dice)),
            "mDice_fg": float(np.nanmean(dice[foreground])),
            "mPrecision_fg": float(np.nanmean(precision[foreground])),
            "mRecall_fg": float(np.nanmean(recall[foreground])),
        }

        for index, class_name in enumerate(CLASS_NAMES):
            result[f"IoU_{class_name}"] = float(iou[index])
            result[f"Dice_{class_name}"] = float(dice[index])
            result[f"Precision_{class_name}"] = float(precision[index])
            result[f"Recall_{class_name}"] = float(recall[index])

        return result

def calculate_per_image_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
        names,
        class_names=CLASS_NAMES,
) -> list[dict]:
    predictions = logits.argmax(dim=1)

    predictions = (
        predictions.detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    targets = (
        targets.detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    records = []

    for prediction, target, name in zip(
            predictions,
            targets,
            names,
    ):
        confusion = np.zeros(
            (NUM_CLASSES, NUM_CLASSES),
            dtype=np.int64,
        )

        valid = (
                (target >= 0)
                & (target < NUM_CLASSES)
        )

        indices = (
                NUM_CLASSES * target[valid]
                + prediction[valid]
        )

        confusion += np.bincount(
            indices,
            minlength=NUM_CLASSES ** 2,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        matrix = confusion.astype(np.float64)

        true_positive = np.diag(matrix)
        ground_truth = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)

        false_positive = predicted - true_positive
        false_negative = ground_truth - true_positive

        iou_denominator = (
                true_positive
                + false_positive
                + false_negative
        )

        dice_denominator = (
                2 * true_positive
                + false_positive
                + false_negative
        )

        iou = np.divide(
            true_positive,
            iou_denominator,
            out=np.full(
                NUM_CLASSES,
                np.nan,
                dtype=np.float64,
            ),
            where=iou_denominator != 0,
        )

        dice = np.divide(
            2 * true_positive,
            dice_denominator,
            out=np.full(
                NUM_CLASSES,
                np.nan,
                dtype=np.float64,
            ),
            where=dice_denominator != 0,
        )

        foreground_iou = iou[1:]
        foreground_dice = dice[1:]

        record = {
            "name": str(name),
            "mIoU_fg": float(
                np.nanmean(foreground_iou)
            ),
            "mDice_fg": float(
                np.nanmean(foreground_dice)
            ),
        }

        for index, class_name in enumerate(class_names):
            record[f"IoU_{class_name}"] = float(iou[index])
            record[f"Dice_{class_name}"] = float(dice[index])

        records.append(record)

    return records
