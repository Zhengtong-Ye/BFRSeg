from __future__ import annotations

from pathlib import Path
from typing import Optional

import segmentation_models_pytorch as smp

from .mkunet import (
    MKUNET_CHANNELS,
    MKUNetBinary,
    build_mkunet,
)

from .pidnet_adapter import (
    PIDNetSAdapter,
    build_pidnet_s,
)

from .pidnet_adapter_cls import (
    PIDNetSAdapter as PIDNetSClsAdapter,
    build_pidnet_s as build_pidnet_s_cls,
)

from .adapter import (
    PIDNetSClsGKMFusionAdapter,
    build_pidnet_s as build_pidnet_s_cls_gkmfusion,
)


# SegNeXt-S: MSCAN-S + LightHamHead
from .segnext_adapter import (
    SegNeXtS,
    build_segnext_s,
)


def normalize_encoder_weights(
    encoder_weights: Optional[str],
) -> Optional[str]:
   
    if encoder_weights is None:
        return None

    value = str(encoder_weights).strip()

    if value.lower() in {
        "",
        "none",
        "null",
        "false",
    }:
        return None

    return value


def build_model(
    model_name: str,
    encoder_name=None,
    encoder_weights=None,
    in_channels: int = 3,
    classes: int = 2,
    pretrained_path: Path | None = None,
):
    
    raw_name = (
        model_name
        .strip()
        .lower()
        .replace("-", "_")
    )

    # --------------------------------------------------
    # 
    # --------------------------------------------------
    if raw_name in {
        "pidnet",
        "pidnet_s",
        "pidnets",
    }:
        if in_channels != 3:
            raise ValueError(
                "Error"
            )

        if classes != 2:
            raise ValueError(
                "要求classes=2，"
                f"当前classes={classes}"
            )

        return build_pidnet_s(
            classes=classes,
            pretrained_path=pretrained_path,
        )

    # --------------------------------------------------
    # 
    # --------------------------------------------------
    if raw_name in {
        "pidnet_cls",
        "pidnet_s_cls",
        "pidnets_cls",
    }:
        if in_channels != 3:
            raise ValueError(
                "Error"
            )

        if classes != 2:
            raise ValueError(
                "要求classes=2，"
                f"当前classes={classes}"
            )

        return build_pidnet_s_cls(
            classes=classes,
            pretrained_path=pretrained_path,
            use_cls_head=True,
            cls_topk_ratio=0.05,
        )


    # --------------------------------------------------
    # --------------------------------------------------
    if raw_name in {
        "pidnet_gkmfusion",
        "pidnet_s_gkmfusion",
        "pidnets_gkmfusion",
    }:
        if in_channels != 3:
            raise ValueError(
                "Error"
            )

        if classes != 2:
            raise ValueError(
                "要求classes=2，"
                f"当前classes={classes}"
            )

        return build_pidnet_s_cls_gkmfusion(
            classes=classes,
            pretrained_path=pretrained_path,
            use_cls_head=True,
            cls_topk_ratio=0.05,
            cls_guidance_floor=1.0,
        )

    # --------------------------------------------------
    # 
    # --------------------------------------------------
    if raw_name in {
        "pidnet_cls_gkmfusion",
        "pidnet_s_cls_gkmfusion",
        "pidnets_cls_gkmfusion",
    }:
        if in_channels != 3:
            raise ValueError(
                "Error"
            )

        if classes != 2:
            raise ValueError(
                "要求classes=2，"
                f"当前classes={classes}"
            )

        return build_pidnet_s_cls_gkmfusion(
            classes=classes,
            pretrained_path=pretrained_path,
            use_cls_head=True,
            cls_topk_ratio=0.05,
            cls_guidance_floor=0.65,
        )


    # --------------------------------------------------
  
    # --------------------------------------------------
    if raw_name in {
        "segnext",
        "segnext_s",
        "segnexts",
    }:
        if in_channels != 3:
            raise ValueError(
                "Error"
            )

        if classes != 2:
            raise ValueError(
                "要求classes=2，"
                f"当前classes={classes}"
            )

        return build_segnext_s(
            classes=classes,
            pretrained_path=pretrained_path,
        )

    # --------------------------------------------------
   
    # --------------------------------------------------
    if raw_name in MKUNET_CHANNELS:
        if classes != 1:
            raise ValueError(
                "Error"
                f"要求classes=1，当前classes={classes}"
            )

        return build_mkunet(
            variant=raw_name,
            in_channels=in_channels,
        )

    # --------------------------------------------------

    # --------------------------------------------------
    model_key = (
        raw_name
        .replace("_", "")
        .replace("+", "plus")
    )

    if encoder_name is None:
        encoder_name = "resnet34"

    weights = normalize_encoder_weights(
        encoder_weights
    )

    common_args = {
        "encoder_name": encoder_name,
        "encoder_weights": weights,
        "in_channels": in_channels,
        "classes": classes,
        "activation": None,
    }

    if model_key == "unet":
        return smp.Unet(
            **common_args,
        )

    if model_key in {
        "unetplusplus",
        "unetpp",
    }:
        return smp.UnetPlusPlus(
            **common_args,
        )

    if model_key in {
        "deeplabv3plus",
        "deeplabv3p",
    }:
        return smp.DeepLabV3Plus(
            **common_args,
        )

    if model_key == "fpn":
        return smp.FPN(
            **common_args,
        )

    if model_key == "manet":
        return smp.MAnet(
            **common_args,
        )

    if model_key == "pspnet":
        return smp.PSPNet(
            **common_args,
        )

    if model_key == "linknet":
        return smp.Linknet(
            **common_args,
        )

    raise ValueError(
        f"不支持的模型：{model_name}。\n"
        "当前支持："
        "pidnet、pidnet_s、pidnets、"
        "pidnet_cls、pidnet_s_cls、pidnets_cls、"
        "pidnet_gkmfusion、pidnet_s_gkmfusion、pidnets_gkmfusion、"
        "pidnet_cls_gkmfusion、pidnet_s_cls_gkmfusion、"
        "segnext、segnext_s、segnexts、"
        "unet、unetplusplus、deeplabv3plus、"
        "fpn、manet、pspnet、linknet、"
        f"{list(MKUNET_CHANNELS.keys())}"
    )


__all__ = [
    "build_model",
    "build_mkunet",
    "MKUNetBinary",
    "MKUNET_CHANNELS",
    "PIDNetSAdapter",
    "PIDNetSClsAdapter",
    "build_pidnet_s",
    "build_pidnet_s_cls",
    "PIDNetSClsGKMFusionAdapter",
    "build_pidnet_s_cls_gkmfusion",
    "SegNeXtS",
    "build_segnext_s",
]