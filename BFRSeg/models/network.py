# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    BasicBlock,
    Bottleneck,
    segmenthead,
    DAPPM,
    PAPPM,
    PagFM,
    Bag,
    Light_Bag,
)

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in (max_groups, 4, 2, 1):
        if groups <= channels and channels % groups == 0:
            return groups
    return 1


def _make_gaussian_kernel(
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    if kernel_size % 2 != 1:
        raise ValueError("Gaussian kernel_size 必须为奇数。")
    if sigma <= 0:
        raise ValueError("Gaussian sigma 必须大于0。")

    radius = kernel_size // 2
    coordinates = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.float32,
    )
    yy, xx = torch.meshgrid(
        coordinates,
        coordinates,
        indexing="ij",
    )
    kernel = torch.exp(
        -(xx.square() + yy.square())
        / (2.0 * sigma * sigma)
    )
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, kernel_size, kernel_size)


class DefectExistenceHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        topk_ratio: float = 0.05,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(in_channels // 2, 32)
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError(
                f"topk_ratio必须位于(0,1]，当前为{topk_ratio}"
            )

        self.topk_ratio = float(topk_ratio)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mil_conv = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
            bias=True,
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_channels + 1, 1)

    def forward(
        self,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature.ndim != 4:
            raise ValueError(
                "分类头输入必须为[B,C,H,W]，"
                f"实际为{tuple(feature.shape)}"
            )

        batch_size = feature.shape[0]
        gap_feature = self.gap(feature).flatten(1)
        gmp_feature = self.gmp(feature).flatten(1)
        global_feature = torch.cat(
            [gap_feature, gmp_feature],
            dim=1,
        )
        global_feature = self.global_mlp(global_feature)

        mil_map = self.mil_conv(feature)
        mil_flatten = mil_map.flatten(1)
        location_count = mil_flatten.shape[1]
        k = max(
            1,
            int(location_count * self.topk_ratio),
        )
        topk_logit = torch.topk(
            mil_flatten,
            k=k,
            dim=1,
            largest=True,
            sorted=False,
        ).values.mean(dim=1, keepdim=True)

        cls_feature = torch.cat(
            [global_feature, topk_logit],
            dim=1,
        )
        cls_logit = self.classifier(cls_feature)

        if cls_logit.shape != (batch_size, 1):
            raise RuntimeError(
                "分类头输出尺寸错误："
                f"期望={(batch_size, 1)}，"
                f"实际={tuple(cls_logit.shape)}"
            )
        return cls_logit, mil_map


class GaussianKernelMixtureFeatureRestoration(nn.Module):
    
    def __init__(
        self,
        channels: int,
        sigmas: tuple[float, ...] = (0.8, 1.4, 2.2),
        kernel_size: int = 9,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()

        kernel_size = int(kernel_size)
        if hidden_channels is None:
            hidden_channels = max(channels, 32)
        if not sigmas:
            raise ValueError("至少需要一个 Gaussian sigma。")

        self.channels = int(channels)
        self.sigmas = tuple(float(s) for s in sigmas)
        self.kernel_size = kernel_size
        self.num_kernels = len(self.sigmas) + 1  # identity + Gaussian candidates

        kernel_bank = torch.cat(
            [
                _make_gaussian_kernel(
                    kernel_size,
                    sigma,
                )
                for sigma in self.sigmas
            ],
            dim=0,
        )
        self.register_buffer(
            "gaussian_kernels",
            kernel_bank,
            persistent=True,
        )

        groups = _group_count(
            hidden_channels,
            max_groups=8,
        )
        self.context_encoder = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.kernel_weight_head = nn.Conv2d(
            hidden_channels,
            self.num_kernels,
            kernel_size=1,
            bias=True,
        )

        initial_gain = math.log(math.exp(0.10) - 1.0)
        self.inverse_gain_logits = nn.Parameter(
            torch.full(
                (len(self.sigmas),),
                initial_gain,
                dtype=torch.float32,
            )
        )

        refine_in_channels = channels * 3 + 1
        refine_groups = _group_count(
            channels,
            max_groups=8,
        )
        self.delta_refine = nn.Sequential(
            nn.Conv2d(
                refine_in_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(refine_groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(refine_groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True,
            ),
        )

        reliability_in_channels = channels * 3 + 1
        reliability_hidden = max(channels // 2, 16)
        reliability_groups = _group_count(
            reliability_hidden,
            max_groups=8,
        )
        self.reliability_head = nn.Sequential(
            nn.Conv2d(
                reliability_in_channels,
                reliability_hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                reliability_groups,
                reliability_hidden,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                reliability_hidden,
                1,
                kernel_size=1,
                bias=True,
            ),
        )


        maximum_sigma = max(self.sigmas)
        blur_levels = [0.0] + [
            sigma / maximum_sigma
            for sigma in self.sigmas
        ]
        self.register_buffer(
            "blur_levels",
            torch.tensor(
                blur_levels,
                dtype=torch.float32,
            ).view(1, self.num_kernels, 1, 1),
            persistent=True,
        )

    def _gaussian_filter(
        self,
        feature: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        channels = feature.shape[1]
        weight = kernel.expand(
            channels,
            1,
            self.kernel_size,
            self.kernel_size,
        )
        return F.conv2d(
            feature,
            weight,
            bias=None,
            stride=1,
            padding=self.kernel_size // 2,
            groups=channels,
        )

    def forward(
        self,
        feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if feature.ndim != 4:
            raise ValueError(
                "恢复模块输入必须为[B,C,H,W]，"
                f"实际={tuple(feature.shape)}"
            )
        if feature.shape[1] != self.channels:
            raise ValueError(
                f"恢复模块期望通道={self.channels}，"
                f"实际={feature.shape[1]}"
            )

        context = self.context_encoder(feature)
        kernel_logits = self.kernel_weight_head(context)
        kernel_weights = torch.softmax(
            kernel_logits,
            dim=1,
        )

        gains = F.softplus(
            self.inverse_gain_logits
        )
        candidates = [feature]
        for index, kernel in enumerate(
            self.gaussian_kernels
        ):
            blurred = self._gaussian_filter(
                feature,
                kernel.unsqueeze(0),
            )
            detail = feature - blurred
            candidate = (
                feature
                + gains[index].view(1, 1, 1, 1)
                * detail
            )
            candidates.append(candidate)

        candidate_stack = torch.stack(
            candidates,
            dim=1,
        )  # [B,K,C,H,W]
        mixed_feature = (
            kernel_weights.unsqueeze(2)
            * candidate_stack
        ).sum(dim=1)

        raw_delta = mixed_feature - feature
        blur_map = (
            kernel_weights
            * self.blur_levels.to(
                dtype=kernel_weights.dtype
            )
        ).sum(dim=1, keepdim=True)

        refine_input = torch.cat(
            [
                feature,
                mixed_feature,
                raw_delta.abs(),
                blur_map,
            ],
            dim=1,
        )
        restoration_delta = self.delta_refine(
            refine_input
        )
        restored_feature = (
            feature + restoration_delta
        )

        reliability_input = torch.cat(
            [
                feature,
                restored_feature,
                restoration_delta.abs(),
                blur_map,
            ],
            dim=1,
        )
        reliability_map = torch.sigmoid(
            self.reliability_head(
                reliability_input
            )
        )

        return {
            "restored_feature": restored_feature,
            "restoration_delta": restoration_delta,
            "blur_map": blur_map,
            "reliability_map": reliability_map,
            "kernel_weights": kernel_weights,
        }


class BranchRestorationInjector(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_type: str,
    ) -> None:
        super().__init__()
        if branch_type not in {"p", "i", "d"}:
            raise ValueError(
                f"未知 branch_type：{branch_type}"
            )
        self.branch_type = branch_type

        groups = _group_count(
            out_channels,
            max_groups=8,
        )
        self.project = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
        )
        gate_hidden = max(out_channels // 4, 16)
        gate_groups = _group_count(
            gate_hidden,
            max_groups=8,
        )
        self.branch_gate_head = nn.Sequential(
            nn.Conv2d(
                out_channels + 2,
                gate_hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(gate_groups, gate_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                gate_hidden,
                1,
                kernel_size=1,
                bias=True,
            ),
        )

        
        self.inject_scale = nn.Parameter(
            torch.full((1,), 0.01)
        )

    def _branch_transform(
        self,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        if self.branch_type == "p":
            return delta
        if self.branch_type == "i":
            return F.avg_pool2d(
                delta,
                kernel_size=5,
                stride=1,
                padding=2,
            )

        local_mean = F.avg_pool2d(
            delta,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        return delta - local_mean

    def forward(
        self,
        branch_feature: torch.Tensor,
        restoration_delta: torch.Tensor,
        blur_map: torch.Tensor,
        reliability_map: torch.Tensor,
        cls_soft_factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_size = branch_feature.shape[-2:]

        delta = self._branch_transform(
            restoration_delta
        )
        delta = F.interpolate(
            delta,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        correction = self.project(delta)

        blur = F.interpolate(
            blur_map,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        reliability = F.interpolate(
            reliability_map,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        learned_gate = torch.sigmoid(
            self.branch_gate_head(
                torch.cat(
                    [
                        correction,
                        blur,
                        reliability,
                    ],
                    dim=1,
                )
            )
        )
        blur_factor = 0.05 + 0.95 * blur
        reliability_factor = (
            0.25 + 0.75 * reliability
        )
        learned_factor = (
            0.25 + 0.75 * learned_gate
        )
        gate = (
            blur_factor
            * reliability_factor
            * learned_factor
            * cls_soft_factor
        )

        output = (
            branch_feature
            + torch.tanh(self.inject_scale)
            * gate
            * correction
        )
        return output, gate


class BFRSeg(nn.Module):

    def __init__(
        self,
        m: int = 2,
        n: int = 3,
        num_classes: int = 19,
        planes: int = 64,
        ppm_planes: int = 96,
        head_planes: int = 128,
        augment: bool = True,
        use_cls_head: bool = False,
        cls_topk_ratio: float = 0.05,
        use_gkm_fusion: bool = True,
        cls_guidance_floor: float = 0.65,
    ) -> None:
        super().__init__()

        if not 0.0 <= cls_guidance_floor <= 1.0:
            raise ValueError(
                "cls_guidance_floor必须位于[0,1]。"
            )

        self.augment = augment
        self.use_cls_head = use_cls_head
        self.use_gkm_fusion = use_gkm_fusion
        self.cls_guidance_floor = float(
            cls_guidance_floor
        )

        # I Branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3,
                planes,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            BatchNorm2d(
                planes,
                momentum=bn_mom,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            BatchNorm2d(
                planes,
                momentum=bn_mom,
            ),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(
            BasicBlock,
            planes,
            planes,
            m,
        )
        self.layer2 = self._make_layer(
            BasicBlock,
            planes,
            planes * 2,
            m,
            stride=2,
        )
        self.layer3 = self._make_layer(
            BasicBlock,
            planes * 2,
            planes * 4,
            n,
            stride=2,
        )
        self.layer4 = self._make_layer(
            BasicBlock,
            planes * 4,
            planes * 8,
            n,
            stride=2,
        )
        self.layer5 = self._make_layer(
            Bottleneck,
            planes * 8,
            planes * 8,
            2,
            stride=2,
        )

        # P Branch
        self.compression3 = nn.Sequential(
            nn.Conv2d(
                planes * 4,
                planes * 2,
                kernel_size=1,
                bias=False,
            ),
            BatchNorm2d(
                planes * 2,
                momentum=bn_mom,
            ),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(
                planes * 8,
                planes * 2,
                kernel_size=1,
                bias=False,
            ),
            BatchNorm2d(
                planes * 2,
                momentum=bn_mom,
            ),
        )
        self.pag3 = PagFM(
            planes * 2,
            planes,
        )
        self.pag4 = PagFM(
            planes * 2,
            planes,
        )
        self.layer3_ = self._make_layer(
            BasicBlock,
            planes * 2,
            planes * 2,
            m,
        )
        self.layer4_ = self._make_layer(
            BasicBlock,
            planes * 2,
            planes * 2,
            m,
        )
        self.layer5_ = self._make_layer(
            Bottleneck,
            planes * 2,
            planes * 2,
            1,
        )

        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(
                BasicBlock,
                planes * 2,
                planes,
            )
            self.layer4_d = self._make_layer(
                Bottleneck,
                planes,
                planes,
                1,
            )
            self.diff3 = nn.Sequential(
                nn.Conv2d(
                    planes * 4,
                    planes,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                BatchNorm2d(
                    planes,
                    momentum=bn_mom,
                ),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(
                    planes * 8,
                    planes * 2,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                BatchNorm2d(
                    planes * 2,
                    momentum=bn_mom,
                ),
            )
            self.spp = PAPPM(
                planes * 16,
                ppm_planes,
                planes * 4,
            )
            self.dfm = Light_Bag(
                planes * 4,
                planes * 4,
            )
        else:
            self.layer3_d = self._make_single_layer(
                BasicBlock,
                planes * 2,
                planes * 2,
            )
            self.layer4_d = self._make_single_layer(
                BasicBlock,
                planes * 2,
                planes * 2,
            )
            self.diff3 = nn.Sequential(
                nn.Conv2d(
                    planes * 4,
                    planes * 2,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                BatchNorm2d(
                    planes * 2,
                    momentum=bn_mom,
                ),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(
                    planes * 8,
                    planes * 2,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                BatchNorm2d(
                    planes * 2,
                    momentum=bn_mom,
                ),
            )
            self.spp = DAPPM(
                planes * 16,
                ppm_planes,
                planes * 4,
            )
            self.dfm = Bag(
                planes * 4,
                planes * 4,
            )

        self.layer5_d = self._make_layer(
            Bottleneck,
            planes * 2,
            planes * 2,
            1,
        )

        if self.use_gkm_fusion:
            branch_channels = planes * 4
            self.gkm_restoration = (
                GaussianKernelMixtureFeatureRestoration(
                    channels=planes,
                    sigmas=(0.8, 1.4, 2.2),
                    kernel_size=9,
                    hidden_channels=planes,
                )
            )
            guide_groups = _group_count(
                branch_channels,
                max_groups=8,
            )
            self.guide_fusion = nn.Sequential(
                nn.Conv2d(
                    branch_channels * 3,
                    branch_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(
                    guide_groups,
                    branch_channels,
                ),
                nn.SiLU(inplace=True),
            )
            self.guide_cls_head = DefectExistenceHead(
                in_channels=branch_channels,
                hidden_channels=max(
                    head_planes // 2,
                    32,
                ),
                topk_ratio=cls_topk_ratio,
                dropout=0.1,
            )
            self.inject_p = BranchRestorationInjector(
                in_channels=planes,
                out_channels=branch_channels,
                branch_type="p",
            )
            self.inject_i = BranchRestorationInjector(
                in_channels=planes,
                out_channels=branch_channels,
                branch_type="i",
            )
            self.inject_d = BranchRestorationInjector(
                in_channels=planes,
                out_channels=branch_channels,
                branch_type="d",
            )
        else:
            self.gkm_restoration = None
            self.guide_fusion = None
            self.guide_cls_head = None
            self.inject_p = None
            self.inject_i = None
            self.inject_d = None

        # Prediction heads
        if self.augment:
            self.seghead_p = segmenthead(
                planes * 2,
                head_planes,
                num_classes,
            )
            self.seghead_d = segmenthead(
                planes * 2,
                planes,
                1,
            )

        self.final_layer = segmenthead(
            planes * 4,
            head_planes,
            num_classes,
        )

        if self.use_cls_head:
            self.cls_head = DefectExistenceHead(
                in_channels=planes * 4,
                hidden_channels=head_planes,
                topk_ratio=cls_topk_ratio,
                dropout=0.1,
            )
        else:
            self.cls_head = None

        self._initialize_weights()
        self._initialize_restoration_branch()

    def _initialize_restoration_branch(self) -> None:
        if self.gkm_restoration is None:
            return

        kernel_bias = (
            self.gkm_restoration
            .kernel_weight_head
            .bias
        )
        if kernel_bias is not None:
            nn.init.zeros_(kernel_bias)
            with torch.no_grad():
                kernel_bias[0] = 1.5

        reliability_last = (
            self.gkm_restoration
            .reliability_head[-1]
        )
        if reliability_last.bias is not None:
            nn.init.constant_(
                reliability_last.bias,
                1.0,
            )

        for injector in (
            self.inject_p,
            self.inject_i,
            self.inject_d,
        ):
            gate_last = (
                injector.branch_gate_head[-1]
            )
            if gate_last.bias is not None:
                nn.init.zeros_(
                    gate_last.bias
                )

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(
        self,
        block,
        inplanes,
        planes,
        blocks,
        stride=1,
    ):
        downsample = None
        if (
            stride != 1
            or inplanes != planes * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    planes * block.expansion,
                    momentum=bn_mom,
                ),
            )

        layers = [
            block(
                inplanes,
                planes,
                stride,
                downsample,
            )
        ]
        inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    stride=1,
                    no_relu=(index == blocks - 1),
                )
            )
        return nn.Sequential(*layers)

    def _make_single_layer(
        self,
        block,
        inplanes,
        planes,
        stride=1,
    ):
        downsample = None
        if (
            stride != 1
            or inplanes != planes * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    planes * block.expansion,
                    momentum=bn_mom,
                ),
            )
        return block(
            inplanes,
            planes,
            stride,
            downsample,
            no_relu=True,
        )

    def forward(
        self,
        input_image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        width_output = input_image.shape[-1] // 8
        height_output = input_image.shape[-2] // 8

        shallow = self.conv1(input_image)
        shallow = self.layer1(shallow)

        restoration_outputs = None
        if self.gkm_restoration is not None:
            restoration_outputs = self.gkm_restoration(
                shallow
            )

        x = self.relu(
            self.layer2(
                self.relu(shallow)
            )
        )
        x_p = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_p = self.pag3(
            x_p,
            self.compression3(x),
        )
        x_d = x_d + F.interpolate(
            self.diff3(x),
            size=[height_output, width_output],
            mode="bilinear",
            align_corners=algc,
        )
        temp_p = x_p if self.augment else None

        x = self.relu(self.layer4(x))
        x_p = self.layer4_(self.relu(x_p))
        x_d = self.layer4_d(self.relu(x_d))

        x_p = self.pag4(
            x_p,
            self.compression4(x),
        )
        x_d = x_d + F.interpolate(
            self.diff4(x),
            size=[height_output, width_output],
            mode="bilinear",
            align_corners=algc,
        )
        temp_d = x_d if self.augment else None

        x_p = self.layer5_(self.relu(x_p))
        x_d = self.layer5_d(self.relu(x_d))
        x_i = F.interpolate(
            self.spp(self.layer5(x)),
            size=[height_output, width_output],
            mode="bilinear",
            align_corners=algc,
        )

        guide_cls_logit = None
        guide_mil_map = None
        gate_p = None
        gate_i = None
        gate_d = None

        if restoration_outputs is not None:
            guide_feature = self.guide_fusion(
                torch.cat(
                    [x_p, x_i, x_d],
                    dim=1,
                )
            )
            (
                guide_cls_logit,
                guide_mil_map,
            ) = self.guide_cls_head(
                guide_feature
            )
            cls_probability = torch.sigmoid(
                guide_cls_logit
            ).view(-1, 1, 1, 1)
            cls_soft_factor = (
                self.cls_guidance_floor
                + (
                    1.0
                    - self.cls_guidance_floor
                )
                * cls_probability
            )

            x_p, gate_p = self.inject_p(
                branch_feature=x_p,
                restoration_delta=restoration_outputs[
                    "restoration_delta"
                ],
                blur_map=restoration_outputs[
                    "blur_map"
                ],
                reliability_map=restoration_outputs[
                    "reliability_map"
                ],
                cls_soft_factor=cls_soft_factor,
            )
            x_i, gate_i = self.inject_i(
                branch_feature=x_i,
                restoration_delta=restoration_outputs[
                    "restoration_delta"
                ],
                blur_map=restoration_outputs[
                    "blur_map"
                ],
                reliability_map=restoration_outputs[
                    "reliability_map"
                ],
                cls_soft_factor=cls_soft_factor,
            )
            x_d, gate_d = self.inject_d(
                branch_feature=x_d,
                restoration_delta=restoration_outputs[
                    "restoration_delta"
                ],
                blur_map=restoration_outputs[
                    "blur_map"
                ],
                reliability_map=restoration_outputs[
                    "reliability_map"
                ],
                cls_soft_factor=cls_soft_factor,
            )

        final_feature = self.dfm(
            x_p,
            x_i,
            x_d,
        )
        main_logits = self.final_layer(
            final_feature
        )

        cls_logit = None
        mil_map = None
        if self.cls_head is not None:
            cls_logit, mil_map = self.cls_head(
                final_feature
            )

        result: dict[str, torch.Tensor] = {
            "out": main_logits,
        }

        if self.augment:
            result["aux"] = self.seghead_p(
                temp_p
            )
            result["edge"] = self.seghead_d(
                temp_d
            )

        if cls_logit is not None:
            result["cls"] = cls_logit
            result["mil"] = mil_map

        if restoration_outputs is not None:
            result.update(
                {
                    "guide_cls": guide_cls_logit,
                    "guide_mil": guide_mil_map,
                    "blur_map": restoration_outputs[
                        "blur_map"
                    ],
                    "reliability_map": (
                        restoration_outputs[
                            "reliability_map"
                        ]
                    ),
                    "kernel_weights": (
                        restoration_outputs[
                            "kernel_weights"
                        ]
                    ),
                    "restoration_delta": (
                        restoration_outputs[
                            "restoration_delta"
                        ]
                    ),
                    "gate_p": gate_p,
                    "gate_i": gate_i,
                    "gate_d": gate_d,
                    "fusion_gate": (
                        gate_p + gate_i + gate_d
                    ) / 3.0,
                }
            )

        return result
