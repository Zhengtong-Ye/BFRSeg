# -*- coding: utf-8 -*-


from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .network import PIDNet


def _strip_prefixes(key: str) -> str:
    for prefix in (
        "module.",
        "model.",
    ):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_pidnet_imagenet_weights(
    network: nn.Module,
    pretrained_path: Path,
) -> None:
    pretrained_path = Path(pretrained_path)
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"找不到预训练权重：{pretrained_path}"
        )

    checkpoint = torch.load(
        pretrained_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        pretrained_state = checkpoint["state_dict"]
    else:
        pretrained_state = checkpoint

    if not isinstance(pretrained_state, dict):
        raise TypeError(
            "无法识别预训练权重格式。"
        )

    model_state = network.state_dict()
    matched_state: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []

    for original_key, value in pretrained_state.items():
        key = _strip_prefixes(str(original_key))
        if key.startswith("network."):
            key = key[len("network."):]

        if (
            key in model_state
            and model_state[key].shape == value.shape
        ):
            matched_state[key] = value
        else:
            skipped_keys.append(str(original_key))

    if not matched_state:
        raise RuntimeError(
            "没有加载到任何预训练参数。"
        )

    load_result = network.load_state_dict(
        matched_state,
        strict=False,
    )

    print("=====预训练权重加载结果 =====")
    print(f"权重文件：{pretrained_path}")
    print(f"成功匹配张量数：{len(matched_state)}")
    print(f"跳过张量数：{len(skipped_keys)}")
    print(
        "模型未加载键数量："
        f"{len(load_result.missing_keys)}"
    )
    print(
        "意外键数量："
        f"{len(load_result.unexpected_keys)}"
    )
    if load_result.missing_keys:
        print("部分未加载参数：")
        for key in load_result.missing_keys[:30]:
            print(f"  {key}")


class BFRSegAdapter(nn.Module):

    def __init__(
        self,
        classes: int = 2,
        pretrained_path: Path | None = None,
        use_cls_head: bool = True,
        cls_topk_ratio: float = 0.05,
        cls_guidance_floor: float = 0.65,
    ) -> None:
        super().__init__()

        if classes != 2:
            raise ValueError(
                "Error"
            )

        self.network = PIDNet(
            m=2,
            n=3,
            num_classes=classes,
            planes=32,
            ppm_planes=96,
            head_planes=128,
            augment=True,
            use_cls_head=use_cls_head,
            cls_topk_ratio=cls_topk_ratio,
            use_gkm_fusion=True,
            cls_guidance_floor=cls_guidance_floor,
        )

        if pretrained_path is not None:
            load_pidnet_imagenet_weights(
                self.network,
                pretrained_path,
            )

    @staticmethod
    def _resize_map(
        tensor: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        if tensor.shape[-2:] == size:
            return tensor
        return F.interpolate(
            tensor,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_size = tuple(images.shape[-2:])
        outputs = self.network(images)

        if not isinstance(outputs, dict):
            raise RuntimeError(
                "Error"
            )

        required = {
            "aux",
            "out",
            "edge",
            "cls",
            "guide_cls",
            "blur_map",
            "reliability_map",
            "fusion_gate",
        }
        missing = required - set(outputs)
        if missing:
            raise KeyError(
                f"模型输出缺少字段：{sorted(missing)}"
            )

        result = dict(outputs)

        for key in (
            "aux",
            "out",
            "edge",
            "blur_map",
            "reliability_map",
            "fusion_gate",
            "gate_p",
            "gate_i",
            "gate_d",
        ):
            if key in result:
                result[key] = self._resize_map(
                    result[key],
                    input_size,
                )


        if "restoration_delta" in result:
            delta = result["restoration_delta"]
            delta_energy = torch.mean(
                delta.abs(),
                dim=1,
                keepdim=True,
            )
            result["restoration_energy"] = (
                self._resize_map(
                    delta_energy,
                    input_size,
                )
            )
            del result["restoration_delta"]

        for key in ("cls", "guide_cls"):
            value = result[key]
            if value.ndim != 2 or value.shape[1] != 1:
                raise RuntimeError(
                    f"{key}输出应为[B,1]，"
                    f"实际={tuple(value.shape)}"
                )

        return result


def build_pidnet_s(
    classes: int = 2,
    pretrained_path: Path | None = None,
    use_cls_head: bool = True,
    cls_topk_ratio: float = 0.05,
    cls_guidance_floor: float = 0.65,
) -> nn.Module:
    return BFRSegAdapter(
        classes=classes,
        pretrained_path=pretrained_path,
        use_cls_head=use_cls_head,
        cls_topk_ratio=cls_topk_ratio,
        cls_guidance_floor=cls_guidance_floor,
    )
