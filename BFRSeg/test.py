# -*- coding: utf-8 -*-

支持目录：
    split/sharp/images,masks
    split/real_blur/images,masks


from __future__ import annotations

# ============================ 用户配置区 ============================

from pathlib import Path

DATA_ROOT = Path(
    r"E:\seg_data\split_test"
)

CHECKPOINT_PATH = Path(
    r"E:\seg_project\runs\best.pth"
)

SPLIT = "test"

# 可改为只测试一个域，例如 ("real_blur",)
DOMAINS = (
    "sharp",
    "real_blur",
)

THRESHOLDS = (
    0.35,

)

SELECT_METRIC = "Dice_defect"

BATCH_SIZE = 2
WORKERS = 0
USE_AMP = True

SAVE_PREDICTIONS = True
SAVE_OVERLAYS = True
SAVE_AUXILIARY_MAPS = True
SAVE_PLOTS = True

OVERLAY_ALPHA = 0.22

VAL_SHARP_WEIGHT = 0.40
VAL_REAL_BLUR_WEIGHT = 0.60

OUTPUT_DIR: Path | None = None

# ====================================================

from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    NUM_CLASSES,
    CLASS_NAMES,
    get_device,
    save_json,
)

from dataset_blur import BinarySegDataset
from losses import CEDiceLoss
from metrics import (
    SegmentationMeter,
    calculate_per_image_metrics,
)
from models import build_model
from plot_utils import (
    plot_class_metrics,
    plot_confusion_matrix,
    plot_per_image_metrics,
)


MODEL_NAMES = {
   "BFRSeg",
}

VALID_MODEL_NAMES = (
    MODEL_NAMES
    )


def imread_unicode(
    path: Path,
    flags: int,
) -> np.ndarray:
    data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )
    image = cv2.imdecode(
        data,
        flags,
    )
    if image is None:
        raise FileNotFoundError(
            f"无法读取图像：{path}"
        )
    return image


def imwrite_unicode(
    path: Path,
    image: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    suffix = path.suffix.lower()
    if suffix not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }:
        path = path.with_suffix(
            ".png"
        )
        suffix = ".png"

    ok, encoded = cv2.imencode(
        suffix,
        image,
    )
    if not ok:
        raise RuntimeError(
            f"图像编码失败：{path}"
        )
    encoded.tofile(
        str(path)
    )


def normalize_masks(
    masks: torch.Tensor,
    names,
) -> torch.Tensor:
    if (
        masks.ndim == 4
        and masks.shape[1] == 1
    ):
        masks = masks[:, 0]

    if masks.ndim != 3:
        raise ValueError(
            "标签应为[B,H,W]或[B,1,H,W]，"
            f"实际={tuple(masks.shape)}，"
            f"图片={list(names)}"
        )

    masks = (
        masks > 0.5
    ).long()
    return masks


def build_metric_logits(
    prediction_masks: torch.Tensor,
) -> torch.Tensor:
    prediction = (
        prediction_masks.float()
    )
    return torch.stack(
        [
            1.0 - prediction,
            prediction,
        ],
        dim=1,
    )


def add_binary_metrics(
    metrics: dict[str, Any],
    confusion: np.ndarray,
) -> dict[str, Any]:
    confusion = np.asarray(
        confusion,
        dtype=np.float64,
    )
    if confusion.shape != (
        2,
        2,
    ):
        raise ValueError(
            "Error"
            f"实际={confusion.shape}"
        )

    tn, fp = (
        confusion[0, 0],
        confusion[0, 1],
    )
    fn, tp = (
        confusion[1, 0],
        confusion[1, 1],
    )
    eps = 1e-12

    iou_background = tn / (
        tn + fp + fn + eps
    )
    iou_defect = tp / (
        tp + fp + fn + eps
    )
    metrics[
        "IoU_background"
    ] = float(
        iou_background
    )
    metrics[
        "IoU_defect"
    ] = float(
        iou_defect
    )
    metrics[
        "mIoU_all"
    ] = float(
        0.5
        * (
            iou_background
            + iou_defect
        )
    )
    metrics[
        "Prediction_positive_ratio"
    ] = float(
        (
            tp + fp
        )
        / (
            confusion.sum()
            + eps
        )
    )
    metrics[
        "GT_positive_ratio"
    ] = float(
        (
            tp + fn
        )
        / (
            confusion.sum()
            + eps
        )
    )
    return metrics



def calculate_metrics_from_confusion(
    confusion: np.ndarray,
) -> dict[str, float]:
       confusion = np.asarray(
        confusion,
        dtype=np.float64,
    )
    if confusion.shape != (2, 2):
        raise ValueError(
            "Error"
            f"实际={confusion.shape}"
        )

    tn, fp = confusion[0, 0], confusion[0, 1]
    fn, tp = confusion[1, 0], confusion[1, 1]
    eps = 1e-12

    precision_background = tn / (tn + fn + eps)
    recall_background = tn / (tn + fp + eps)
    iou_background = tn / (tn + fp + fn + eps)
    dice_background = (
        2.0 * tn
        / (2.0 * tn + fp + fn + eps)
    )

    precision_defect = tp / (tp + fp + eps)
    recall_defect = tp / (tp + fn + eps)
    iou_defect = tp / (tp + fp + fn + eps)
    dice_defect = (
        2.0 * tp
        / (2.0 * tp + fp + fn + eps)
    )

    total = confusion.sum()
    pixel_accuracy = (tn + tp) / (total + eps)

    return {
        "pixel_accuracy": float(pixel_accuracy),
        "mIoU_all": float(
            0.5 * (iou_background + iou_defect)
        ),
        "mIoU_fg": float(iou_defect),
        "mDice_all": float(
            0.5 * (dice_background + dice_defect)
        ),
        "mDice_fg": float(dice_defect),
        "mPrecision_fg": float(precision_defect),
        "mRecall_fg": float(recall_defect),
        "IoU_background": float(iou_background),
        "Dice_background": float(dice_background),
        "Precision_background": float(
            precision_background
        ),
        "Recall_background": float(recall_background),
        "IoU_defect": float(iou_defect),
        "Dice_defect": float(dice_defect),
        "Precision_defect": float(precision_defect),
        "Recall_defect": float(recall_defect),
        "Prediction_positive_ratio": float(
            (tp + fp) / (total + eps)
        ),
        "GT_positive_ratio": float(
            (tp + fn) / (total + eps)
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "total_pixels": int(total),
    }


def threshold_tag(
    threshold: float,
) -> str:
    return (
        f"thr{threshold:.2f}"
        .replace(
            ".",
            "p",
        )
    )


def safe_name_from_path(
    path: Path,
    image_root: Path,
) -> str:
    try:
        relative = path.relative_to(
            image_root
        )
        return "__".join(
            relative.with_suffix(
                ""
            ).parts
        )
    except ValueError:
        return path.stem


def save_binary_mask(
    path: Path,
    mask: np.ndarray,
) -> None:
    image = (
        mask.astype(
            np.uint8
        )
        > 0
    ).astype(
        np.uint8
    ) * 255
    imwrite_unicode(
        path,
        image,
    )


def draw_overlay(
    original: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    height, width = (
        original.shape[:2]
    )
    prediction = cv2.resize(
        prediction.astype(
            np.uint8
        ),
        (
            width,
            height,
        ),
        interpolation=(
            cv2.INTER_NEAREST
        ),
    ) > 0
    target = cv2.resize(
        target.astype(
            np.uint8
        ),
        (
            width,
            height,
        ),
        interpolation=(
            cv2.INTER_NEAREST
        ),
    ) > 0

    true_positive = (
        prediction & target
    )
    false_positive = (
        prediction & ~target
    )
    false_negative = (
        ~prediction & target
    )
    active = (
        prediction | target
    )

    color = np.zeros_like(
        original
    )
    color[
        true_positive
    ] = (
        0,
        255,
        255,
    )
    color[
        false_positive
    ] = (
        0,
        0,
        255,
    )
    color[
        false_negative
    ] = (
        0,
        255,
        0,
    )

    overlay = original.copy()
    blended = cv2.addWeighted(
        original,
        1.0 - alpha,
        color,
        alpha,
        0.0,
    )
    overlay[
        active
    ] = blended[
        active
    ]
    return overlay


def save_probability_map(
    path: Path,
    probability: np.ndarray,
) -> None:
    probability = np.clip(
        probability.astype(
            np.float32
        ),
        0.0,
        1.0,
    )
    gray = np.round(
        probability * 255.0
    ).astype(
        np.uint8
    )
    color = cv2.applyColorMap(
        gray,
        getattr(
            cv2,
            "COLORMAP_TURBO",
            cv2.COLORMAP_JET,
        ),
    )
    imwrite_unicode(
        path,
        color,
    )


def resize_if_needed(
    tensor: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    if tensor.shape[-2:] == (
        target_size
    ):
        return tensor
    return F.interpolate(
        tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )


def evaluate_domain(
    *,
    model,
    loader: DataLoader,
    domain: str,
    image_root: Path,
    output_dir: Path,
    device: torch.device,
    amp_enabled: bool,
    criterion,
) -> pd.DataFrame:
    meters = {
        threshold: (
            SegmentationMeter(
                NUM_CLASSES
            )
        )
        for threshold in THRESHOLDS
    }
    records_by_threshold: dict[
        float,
        list[dict[str, Any]],
    ] = {
        threshold: []
        for threshold in THRESHOLDS
    }

    running_loss = 0.0
    sample_count = 0
    cls_probabilities = []
    guide_cls_probabilities = []

    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc=(
                f"{SPLIT} {domain}"
            ),
        )

        for (
            images,
            masks,
            names,
        ) in progress:
            masks = normalize_masks(
                masks,
                names,
            )
            images = images.to(
                device,
                non_blocking=True,
            ).float()
            masks = masks.to(
                device,
                non_blocking=True,
            ).long()

            with torch.amp.autocast(
                device_type=(
                    device.type
                ),
                enabled=(
                    amp_enabled
                ),
            ):
                outputs = model(
                    images
                )
                main_logits = (
                    outputs["out"]
                )
                target_size = tuple(
                    masks.shape[-2:]
                )
                main_logits = (
                    resize_if_needed(
                        main_logits,
                        target_size,
                    )
                )
                loss = criterion(
                    main_logits,
                    masks,
                )

            batch_size = (
                images.shape[0]
            )
            sample_count += (
                batch_size
            )
            running_loss += (
                float(
                    loss.item()
                )
                * batch_size
            )

            defect_probability = (
                torch.softmax(
                    main_logits.float(),
                    dim=1,
                )[:, 1]
            )

            cls_probabilities.extend(
                torch.sigmoid(
                    outputs["cls"].float()
                )
                .reshape(
                    -1
                )
                .detach()
                .cpu()
                .tolist()
            )
            if "guide_cls" in outputs:
                guide_cls_probabilities.extend(
                    torch.sigmoid(
                        outputs[
                            "guide_cls"
                        ].float()
                    )
                    .reshape(
                        -1
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )

            targets_np = (
                masks.detach()
                .cpu()
                .numpy()
                .astype(
                    np.uint8
                )
            )


            if SAVE_AUXILIARY_MAPS:
                auxiliary_keys = (
                    "blur_map",
                    "reliability_map",
                    "fusion_gate",
                    "gate_p",
                    "gate_i",
                    "gate_d",
                    "restoration_energy",
                )
                auxiliary_arrays: dict[
                    str,
                    np.ndarray,
                ] = {}
                for key in (
                    auxiliary_keys
                ):
                    if key in outputs:
                        auxiliary_arrays[
                            key
                        ] = (
                            outputs[
                                key
                            ]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )

                kernel_weights = (
                    outputs.get(
                        "kernel_weights"
                    )
                )
                if kernel_weights is not None:
                    kernel_weights_np = (
                        kernel_weights
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )
                else:
                    kernel_weights_np = (
                        None
                    )

                for batch_index, name in enumerate(
                    names
                ):
                    image_path = Path(
                        name
                    )
                    safe_name = (
                        safe_name_from_path(
                            image_path,
                            image_root,
                        )
                    )
                    for key, array in (
                        auxiliary_arrays.items()
                    ):
                        value = array[
                            batch_index
                        ]
                        if value.ndim == 3:
                            value = (
                                value.mean(
                                    axis=0
                                )
                            )
                        save_probability_map(
                            output_dir
                            / "auxiliary_maps"
                            / key
                            / (
                                f"{safe_name}.png"
                            ),
                            value,
                        )

                    if (
                        kernel_weights_np
                        is not None
                    ):
                        sample_weights = (
                            kernel_weights_np[
                                batch_index
                            ]
                        )
                        for kernel_index in range(
                            sample_weights.shape[
                                0
                            ]
                        ):
                            save_probability_map(
                                output_dir
                                / "auxiliary_maps"
                                / "kernel_weights"
                                / (
                                    "kernel_"
                                    f"{kernel_index}"
                                )
                                / (
                                    f"{safe_name}.png"
                                ),
                                sample_weights[
                                    kernel_index
                                ],
                            )

            for threshold in (
                THRESHOLDS
            ):
                predictions = (
                    defect_probability
                    >= threshold
                ).long()
                metric_logits = (
                    build_metric_logits(
                        predictions
                    )
                )
                meters[
                    threshold
                ].update(
                    metric_logits.detach(),
                    masks.detach(),
                )

                records = (
                    calculate_per_image_metrics(
                        logits=(
                            metric_logits.detach()
                        ),
                        targets=(
                            masks.detach()
                        ),
                        names=names,
                        class_names=(
                            CLASS_NAMES
                        ),
                    )
                )
                for record in (
                    records
                ):
                    record[
                        "threshold"
                    ] = threshold
                    record[
                        "domain"
                    ] = domain
                records_by_threshold[
                    threshold
                ].extend(
                    records
                )

                if (
                    SAVE_PREDICTIONS
                    or SAVE_OVERLAYS
                ):
                    predictions_np = (
                        predictions.detach()
                        .cpu()
                        .numpy()
                        .astype(
                            np.uint8
                        )
                    )
                    result_dir = (
                        output_dir
                        / threshold_tag(
                            threshold
                        )
                    )

                    for (
                        prediction,
                        target,
                        name,
                    ) in zip(
                        predictions_np,
                        targets_np,
                        names,
                    ):
                        image_path = Path(
                            name
                        )
                        original = (
                            imread_unicode(
                                image_path,
                                cv2.IMREAD_COLOR,
                            )
                        )
                        safe_name = (
                            safe_name_from_path(
                                image_path,
                                image_root,
                            )
                        )

                        if SAVE_PREDICTIONS:
                            save_binary_mask(
                                result_dir
                                / "predictions"
                                / (
                                    f"{safe_name}.png"
                                ),
                                prediction,
                            )

                        if SAVE_OVERLAYS:
                            overlay = draw_overlay(
                                original,
                                prediction,
                                target,
                                OVERLAY_ALPHA,
                            )
                            imwrite_unicode(
                                result_dir
                                / "overlays"
                                / (
                                    f"{safe_name}.png"
                                ),
                                overlay,
                            )

            progress.set_postfix(
                loss=(
                    f"{loss.item():.4f}"
                )
            )

    mean_loss = (
        running_loss
        / max(
            sample_count,
            1,
        )
    )

    summary_records = []

    for threshold in (
        THRESHOLDS
    ):
        metrics = meters[
            threshold
        ].compute()
        metrics = (
            add_binary_metrics(
                metrics,
                meters[
                    threshold
                ].confusion,
            )
        )
        metrics[
            "loss"
        ] = mean_loss
        metrics[
            "threshold"
        ] = threshold
        metrics[
            "domain"
        ] = domain
        metrics[
            "sample_count"
        ] = sample_count
        metrics[
            "mean_cls_defect_probability"
        ] = float(
            np.mean(
                cls_probabilities
            )
            if cls_probabilities
            else float(
                "nan"
            )
        )
        metrics[
            "mean_guide_cls_defect_probability"
        ] = float(
            np.mean(
                guide_cls_probabilities
            )
            if guide_cls_probabilities
            else float(
                "nan"
            )
        )

        result_dir = (
            output_dir
            / threshold_tag(
                threshold
            )
        )
        result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_json(
            metrics,
            result_dir
            / "metrics.json",
        )

        per_image_csv = (
            result_dir
            / "per_image_metrics.csv"
        )
        pd.DataFrame(
            records_by_threshold[
                threshold
            ]
        ).to_csv(
            per_image_csv,
            index=False,
            encoding="utf-8-sig",
        )

        if SAVE_PLOTS:
            plots_dir = (
                result_dir
                / "plots"
            )
            plots_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            plot_confusion_matrix(
                confusion_matrix=(
                    meters[
                        threshold
                    ].confusion
                ),
                class_names=(
                    CLASS_NAMES
                ),
                save_path=(
                    plots_dir
                    / "confusion_matrix.png"
                ),
                normalize=False,
            )
            plot_confusion_matrix(
                confusion_matrix=(
                    meters[
                        threshold
                    ].confusion
                ),
                class_names=(
                    CLASS_NAMES
                ),
                save_path=(
                    plots_dir
                    / "normalized_confusion_matrix.png"
                ),
                normalize=True,
            )
            plot_class_metrics(
                metrics=metrics,
                class_names=(
                    CLASS_NAMES
                ),
                output_dir=(
                    plots_dir
                ),
            )
            plot_per_image_metrics(
                csv_path=(
                    per_image_csv
                ),
                output_dir=(
                    plots_dir
                ),
            )

        summary_records.append(
            metrics
        )

    summary_df = pd.DataFrame(
        summary_records
    )
    summary_df.to_csv(
        output_dir
        / "threshold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    confusions = {
        threshold: (
            meters[threshold]
            .confusion
            .copy()
        )
        for threshold in THRESHOLDS
    }

    return (
        summary_df,
        confusions,
        sample_count,
    )


def main() -> None:
    if NUM_CLASSES != 2:
        raise ValueError(
            "Error"
        )
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Error：{DATA_ROOT}"
        )
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Error："
            f"{CHECKPOINT_PATH}"
        )
    if not DOMAINS:
        raise ValueError(
            "Error"
        )
    for domain in DOMAINS:
        if domain not in {
            "sharp",
            "real_blur",
        }:
            raise ValueError(
                f"domain={domain}"
            )
        for leaf in (
            "images",
            "masks",
        ):
            path = (
                DATA_ROOT
                / SPLIT
                / domain
                / leaf
            )
            if not path.is_dir():
                raise FileNotFoundError(
                    f"Error：{path}"
                )

    device = get_device()
    amp_enabled = bool(
        USE_AMP
        and device.type == "cuda"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise KeyError(
            "checkpoint needs model_state_dict。"
        )

    config = checkpoint.get(
        "config",
        {},
    )
    model_name = (
        str(
            config.get(
                "model",
                "BFRSeg",
            )
        )
        .lower()
        .replace(
            "-",
            "_",
        )
    )
    if (
        model_name
        not in VALID_MODEL_NAMES
    ):
        raise ValueError(
            f"Error：{model_name}"
        )

    image_size = int(
        config.get(
            "image_size",
            768,
        )
    )

    model = build_model(
        model_name=model_name,
        encoder_name=None,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        pretrained_path=None,
    ).to(
        device
    )
    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )
    model.eval()

    if OUTPUT_DIR is None:
        output_root = (
            CHECKPOINT_PATH.parent
            / (
                f"{SPLIT}_"
                f"{model_name}_multidomain"
            )
        )
    else:
        output_root = (
            OUTPUT_DIR.resolve()
        )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    criterion = CEDiceLoss()
    domain_summaries: dict[
        str,
        pd.DataFrame,
    ] = {}
    domain_confusions: dict[
        str,
        dict[float, np.ndarray],
    ] = {}
    domain_sample_counts: dict[
        str,
        int,
    ] = {}

    
    for domain in (
        DOMAINS
    ):
        dataset = (
            BinarySegDataset(
                data_root=(
                    DATA_ROOT
                ),
                split=SPLIT,
                domain=domain,
                image_size=(
                    image_size
                ),
                training=False,
                target_format="ce",
            )
        )
        loader = DataLoader(
            dataset,
            batch_size=(
                BATCH_SIZE
            ),
            shuffle=False,
            num_workers=(
                WORKERS
            ),
            pin_memory=(
                device.type
                == "cuda"
            ),
            drop_last=False,
            persistent_workers=(
                WORKERS > 0
            ),
        )
        image_root = (
            DATA_ROOT
            / SPLIT
            / domain
            / "images"
        )
        domain_output_dir = (
            output_root
            / domain
        )
        domain_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            summary,
            confusions,
            domain_sample_count,
        ) = evaluate_domain(
            model=model,
            loader=loader,
            domain=domain,
            image_root=(
                image_root
            ),
            output_dir=(
                domain_output_dir
            ),
            device=device,
            amp_enabled=(
                amp_enabled
            ),
            criterion=criterion,
        )
        domain_summaries[
            domain
        ] = summary
        domain_confusions[
            domain
        ] = confusions
        domain_sample_counts[
            domain
        ] = domain_sample_count

        if (
            SELECT_METRIC
            not in summary.columns
        ):
            raise KeyError(
                f"{domain} summaryError"
                f"{SELECT_METRIC}"
            )
        best_index = (
            summary[
                SELECT_METRIC
            ]
            .astype(
                float
            )
            .idxmax()
        )
        best_row = (
            summary.loc[
                best_index
            ]
        )
        print(
            f"{domain}best："
            f"{float(best_row['threshold']):.2f} | "
            f"IoU={float(best_row['IoU_defect']):.6f} | "
            f"Dice={float(best_row['Dice_defect']):.6f} | "
            f"P={float(best_row['Precision_defect']):.6f} | "
            f"R={float(best_row['Recall_defect']):.6f} | "
            f"mIoU={float(best_row['mIoU_all']):.6f}"
        )

    overall_records: list[dict[str, Any]] = []
    total_sample_count = sum(
        domain_sample_counts.values()
    )

    for threshold in THRESHOLDS:
        pooled_confusion = np.zeros(
            (2, 2),
            dtype=np.int64,
        )
        weighted_loss_sum = 0.0
        weighted_cls_sum = 0.0
        weighted_guide_cls_sum = 0.0
        cls_count = 0
        guide_cls_count = 0

        for domain in DOMAINS:
            pooled_confusion += (
                domain_confusions[
                    domain
                ][threshold]
                .astype(np.int64)
            )

            domain_row = (
                domain_summaries[
                    domain
                ]
                .loc[
                    domain_summaries[
                        domain
                    ]["threshold"]
                    == threshold
                ]
                .iloc[0]
            )
            count = domain_sample_counts[
                domain
            ]
            weighted_loss_sum += (
                float(
                    domain_row["loss"]
                )
                * count
            )

            cls_value = float(
                domain_row[
                    "mean_cls_defect_probability"
                ]
            )
            if np.isfinite(cls_value):
                weighted_cls_sum += (
                    cls_value * count
                )
                cls_count += count

            guide_value = float(
                domain_row[
                    "mean_guide_cls_defect_probability"
                ]
            )
            if np.isfinite(guide_value):
                weighted_guide_cls_sum += (
                    guide_value * count
                )
                guide_cls_count += count

        metrics = calculate_metrics_from_confusion(
            pooled_confusion
        )
        metrics["threshold"] = threshold
        metrics["domain"] = "overall_micro"
        metrics["sample_count"] = (
            total_sample_count
        )
        metrics["loss"] = (
            weighted_loss_sum
            / max(
                total_sample_count,
                1,
            )
        )
        metrics[
            "mean_cls_defect_probability"
        ] = (
            weighted_cls_sum
            / cls_count
            if cls_count > 0
            else float("nan")
        )
        metrics[
            "mean_guide_cls_defect_probability"
        ] = (
            weighted_guide_cls_sum
            / guide_cls_count
            if guide_cls_count > 0
            else float("nan")
        )
        overall_records.append(metrics)

        if SAVE_PLOTS:
            overall_plot_dir = (
                output_root
                / "overall_micro"
                / threshold_tag(
                    threshold
                )
                / "plots"
            )
            overall_plot_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            plot_confusion_matrix(
                confusion_matrix=(
                    pooled_confusion
                ),
                class_names=CLASS_NAMES,
                save_path=(
                    overall_plot_dir
                    / "confusion_matrix.png"
                ),
                normalize=False,
            )
            plot_confusion_matrix(
                confusion_matrix=(
                    pooled_confusion
                ),
                class_names=CLASS_NAMES,
                save_path=(
                    overall_plot_dir
                    / "normalized_confusion_matrix.png"
                ),
                normalize=True,
            )
            plot_class_metrics(
                metrics=metrics,
                class_names=CLASS_NAMES,
                output_dir=(
                    overall_plot_dir
                ),
            )

    overall_df = pd.DataFrame(
        overall_records
    )
    overall_csv = (
        output_root
        / "overall_micro_threshold_summary.csv"
    )
    overall_df.to_csv(
        overall_csv,
        index=False,
        encoding="utf-8-sig",
    )

    if SELECT_METRIC not in overall_df.columns:
        raise KeyError(
            "overall summaryError
            f"{SELECT_METRIC}"
        )

    best_overall_index = (
        overall_df[
            SELECT_METRIC
        ]
        .astype(float)
        .idxmax()
    )
    best_overall = overall_df.loc[
        best_overall_index
    ]

    print(
        "\n===== overall_micro ====="
    )
    overall_display_columns = [
        "threshold",
        "IoU_defect",
        "Dice_defect",
        "Precision_defect",
        "Recall_defect",
        "mIoU_all",
        "pixel_accuracy",
    ]
    print(
        overall_df[
            overall_display_columns
        ].to_string(
            index=False
        )
    )
    print(
        "overall_microbest："
        f"threshold={float(best_overall['threshold']):.2f} | "
        f"IoU={float(best_overall['IoU_defect']):.6f} | "
        f"Dice={float(best_overall['Dice_defect']):.6f} | "
        f"P={float(best_overall['Precision_defect']):.6f} | "
        f"R={float(best_overall['Recall_defect']):.6f} | "
        f"mIoU={float(best_overall['mIoU_all']):.6f}"
    )
    print(
        f"{overall_csv}"
    )


    if {
        "sharp",
        "real_blur",
    }.issubset(
        domain_summaries
    ):
        sharp = (
            domain_summaries[
                "sharp"
            ]
        )
        blur = (
            domain_summaries[
                "real_blur"
            ]
        )
        joint = sharp.merge(
            blur,
            on="threshold",
            suffixes=(
                "_sharp",
                "_real_blur",
            ),
            validate="one_to_one",
        )
        joint[
            "joint_Dice_score"
        ] = (
            VAL_SHARP_WEIGHT
            * joint[
                "Dice_defect_sharp"
            ].astype(
                float
            )
            + VAL_REAL_BLUR_WEIGHT
            * joint[
                "Dice_defect_real_blur"
            ].astype(
                float
            )
        )
        joint[
            "joint_IoU_score"
        ] = (
            VAL_SHARP_WEIGHT
            * joint[
                "IoU_defect_sharp"
            ].astype(
                float
            )
            + VAL_REAL_BLUR_WEIGHT
            * joint[
                "IoU_defect_real_blur"
            ].astype(
                float
            )
        )
        joint[
            "macro_Dice_score"
        ] = (
            0.5
            * (
                joint[
                    "Dice_defect_sharp"
                ].astype(float)
                + joint[
                    "Dice_defect_real_blur"
                ].astype(float)
            )
        )
        joint[
            "macro_IoU_score"
        ] = (
            0.5
            * (
                joint[
                    "IoU_defect_sharp"
                ].astype(float)
                + joint[
                    "IoU_defect_real_blur"
                ].astype(float)
            )
        )
        joint_csv = (
            output_root
            / "joint_threshold_summary.csv"
        )
        joint.to_csv(
            joint_csv,
            index=False,
            encoding="utf-8-sig",
        )

        comparison_ready = joint.merge(
            overall_df[
                [
                    "threshold",
                    "IoU_defect",
                    "Dice_defect",
                    "Precision_defect",
                    "Recall_defect",
                    "mIoU_all",
                    "pixel_accuracy",
                    "sample_count",
                ]
            ],
            on="threshold",
            how="left",
            validate="one_to_one",
        ).rename(
            columns={
                "IoU_defect": (
                    "overall_micro_IoU_defect"
                ),
                "Dice_defect": (
                    "overall_micro_Dice_defect"
                ),
                "Precision_defect": (
                    "overall_micro_Precision_defect"
                ),
                "Recall_defect": (
                    "overall_micro_Recall_defect"
                ),
                "mIoU_all": (
                    "overall_micro_mIoU_all"
                ),
                "pixel_accuracy": (
                    "overall_micro_pixel_accuracy"
                ),
                "sample_count": (
                    "overall_sample_count"
                ),
            }
        )
        comparison_ready_csv = (
            output_root
            / "comparison_ready_summary.csv"
        )
        comparison_ready.to_csv(
            comparison_ready_csv,
            index=False,
            encoding="utf-8-sig",
        )

        best_joint_index = (
            joint[
                "joint_Dice_score"
            ].idxmax()
        )
        best_joint = (
            joint.loc[
                best_joint_index
            ]
        )
        print(
            "\n===== ====="
        )
        display_columns = [
            "threshold",
            "Dice_defect_sharp",
            "Dice_defect_real_blur",
            "macro_Dice_score",
            "joint_Dice_score",
            "IoU_defect_sharp",
            "IoU_defect_real_blur",
            "macro_IoU_score",
            "joint_IoU_score",
        ]
        print(
            joint[
                display_columns
            ].to_string(
                index=False
            )
        )
        print(

            f"{float(best_joint['threshold']):.2f} | "
            "joint Dice="
            f"{float(best_joint['joint_Dice_score']):.6f}"
        )
        print(
            f"{joint_csv}"
        )
        print(
            f"{comparison_ready_csv}"
        )

    print(
        f"root：{output_root}"
    )


if __name__ == "__main__":
    main()
