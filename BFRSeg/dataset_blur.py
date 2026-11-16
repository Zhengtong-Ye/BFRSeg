from __future__ import annotations

from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff", ".webp",
}

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)
IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


class BinarySegDataset(Dataset):
    """
    自建金属缺陷二值分割数据集。

    新目录结构：

        data_root/
        ├── train/
        │   ├── sharp/
        │   │   ├── images/
        │   │   └── masks/
        │   └── real_blur/
        │       ├── images/
        │       └── masks/
        ├── val/
        └── test/

    推荐调用：

        BinarySegDataset(
            data_root=...,
            split="train",
            domain="sharp",
            image_size=768,
            training=True,
        )

    domain 可取：
        "sharp"、"real_blur" 或 None。

    当 domain=None 时：
        1. 如果存在旧目录 data_root/images/<split>，读取旧目录；
        2. 否则合并读取 <split>/sharp 和 <split>/real_blur。

    返回：
        target_format="ce":
            image_tensor: [3,H,W] float32
            mask_tensor:  [H,W] int64，0/1

        target_format="bce":
            image_tensor: [3,H,W] float32
            mask_tensor:  [1,H,W] float32，0/1

        第三个返回值为原图绝对路径字符串。

    注意：
        三路径训练已经单独生成物理失焦图，因此默认不再在 Dataset
        中随机加入普通高斯/运动模糊，避免退化来源混乱。
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        domain: str | None = None,
        image_size: int = 1024,
        training: bool = False,
        target_format: str = "ce",
        enable_blur_augmentation: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = str(split)
        self.domain = None if domain is None else str(domain)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.target_format = target_format.lower()
        self.enable_blur_augmentation = bool(enable_blur_augmentation)

        if self.image_size <= 0:
            raise ValueError(
                f"image_size必须大于0，当前为：{self.image_size}"
            )

        if self.target_format not in {"ce", "bce"}:
            raise ValueError(
                "target_format只能是'ce'或'bce'，"
                f"当前为：{target_format}"
            )

        if self.domain not in {None, "sharp", "real_blur"}:
            raise ValueError(
                "domain只能为None、'sharp'或'real_blur'，"
                f"当前为：{self.domain}"
            )

        self.samples: list[tuple[Path, Path]] = []

        for image_dir, mask_dir in self._resolve_domain_directories():
            self._append_samples(image_dir, mask_dir)

        if not self.samples:
            raise RuntimeError(
                f"split={self.split}, domain={self.domain}中没有找到有效样本。"
            )

    def _resolve_domain_directories(
        self,
    ) -> list[tuple[Path, Path]]:
        """解析新旧两种数据目录。"""
        if self.domain is not None:
            image_dir = (
                self.data_root
                / self.split
                / self.domain
                / "images"
            )
            mask_dir = (
                self.data_root
                / self.split
                / self.domain
                / "masks"
            )

            self._check_directories(image_dir, mask_dir)
            return [(image_dir, mask_dir)]

        # 兼容旧结构：data_root/images/train、data_root/masks/train。
        legacy_image_dir = self.data_root / "images" / self.split
        legacy_mask_dir = self.data_root / "masks" / self.split

        if legacy_image_dir.is_dir() and legacy_mask_dir.is_dir():
            return [(legacy_image_dir, legacy_mask_dir)]

        # 新结构且domain=None：合并sharp与real_blur，便于旧验证/测试脚本继续使用。
        result: list[tuple[Path, Path]] = []

        for domain_name in ("sharp", "real_blur"):
            image_dir = (
                self.data_root
                / self.split
                / domain_name
                / "images"
            )
            mask_dir = (
                self.data_root
                / self.split
                / domain_name
                / "masks"
            )

            if image_dir.is_dir() and mask_dir.is_dir():
                result.append((image_dir, mask_dir))

        if not result:
            raise NotADirectoryError(
                "没有找到可用数据目录。\n"
                f"旧结构应为：{legacy_image_dir} 与 {legacy_mask_dir}\n"
                "新结构应为："
                f"{self.data_root / self.split / 'sharp' / 'images'} "
                "以及 real_blur 对应目录。"
            )

        return result

    @staticmethod
    def _check_directories(
        image_dir: Path,
        mask_dir: Path,
    ) -> None:
        if not image_dir.is_dir():
            raise NotADirectoryError(
                f"图片目录不存在：{image_dir}"
            )

        if not mask_dir.is_dir():
            raise NotADirectoryError(
                f"掩膜目录不存在：{mask_dir}"
            )

    def _append_samples(
        self,
        image_dir: Path,
        mask_dir: Path,
    ) -> None:
        self._check_directories(image_dir, mask_dir)

        for image_path in sorted(image_dir.rglob("*")):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in IMAGE_SUFFIXES
            ):
                continue

            relative_path = image_path.relative_to(image_dir)

            # 切块脚本输出的掩膜通常统一为PNG。
            mask_path = mask_dir / relative_path.with_suffix(".png")

            # 如果掩膜保持与图像相同后缀，也兼容读取。
            if not mask_path.exists():
                same_suffix_mask = mask_dir / relative_path
                if same_suffix_mask.exists():
                    mask_path = same_suffix_mask
                else:
                    raise FileNotFoundError(
                        f"缺少对应掩膜：\n"
                        f"图片：{image_path}\n"
                        f"期望掩膜：{mask_path}"
                    )

            self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def read_image(path: Path) -> np.ndarray:
        """支持中文路径读取彩色图像，返回BGR uint8。"""
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"无法读取图片：{path}")

        return image

    @staticmethod
    def read_mask(path: Path) -> np.ndarray:
        """支持中文路径读取灰度掩膜，返回uint8。"""
        data = np.fromfile(str(path), dtype=np.uint8)
        mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"无法读取掩膜：{path}")

        return mask

    @staticmethod
    def _random_brightness_contrast(
        image: np.ndarray,
    ) -> np.ndarray:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-20.0, 20.0)

        adjusted = image.astype(np.float32) * alpha + beta
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def _random_gamma(image: np.ndarray) -> np.ndarray:
        gamma = random.uniform(0.80, 1.25)
        table = np.array(
            [((i / 255.0) ** gamma) * 255.0 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image, table)

    @staticmethod
    def _clahe_on_l_channel(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        l_channel = clahe.apply(l_channel)

        lab = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _random_noise(image: np.ndarray) -> np.ndarray:
        sigma = random.uniform(2.0, 6.0)
        noise = np.random.normal(
            loc=0.0,
            scale=sigma,
            size=image.shape,
        ).astype(np.float32)

        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def _random_defocus_baseline(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        仅为兼容旧实验保留的轻度普通模糊。
        三路径物理失焦训练时应保持enable_blur_augmentation=False。
        """
        kernel_size = random.choice([3, 5])
        return cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            0,
        )

    def augment(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 几何增强同时作用于图像和掩膜。
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        if random.random() < 0.3:
            rotate_code = random.choice(
                [
                    cv2.ROTATE_90_CLOCKWISE,
                    cv2.ROTATE_90_COUNTERCLOCKWISE,
                    cv2.ROTATE_180,
                ]
            )
            image = cv2.rotate(image, rotate_code)
            mask = cv2.rotate(mask, rotate_code)

        # 三路径训练默认关闭此项，失焦由defocus_degradation.py统一产生。
        if (
            self.enable_blur_augmentation
            and random.random() < 0.20
        ):
            image = self._random_defocus_baseline(image)

        if random.random() < 0.30:
            photometric_type = random.choice(
                ["brightness_contrast", "gamma", "clahe"]
            )

            if photometric_type == "brightness_contrast":
                image = self._random_brightness_contrast(image)
            elif photometric_type == "gamma":
                image = self._random_gamma(image)
            else:
                image = self._clahe_on_l_channel(image)

        if random.random() < 0.10:
            image = self._random_noise(image)

        return image, mask

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]

        image = self.read_image(image_path)
        mask = self.read_mask(mask_path)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                "图片和掩膜尺寸不一致："
                f"\n图片：{image_path}"
                f"\n掩膜：{mask_path}"
                f"\nimage shape={image.shape[:2]}"
                f"\nmask shape={mask.shape[:2]}"
            )

        if self.training:
            image, mask = self.augment(image, mask)

        if image.shape[:2] != (
            self.image_size,
            self.image_size,
        ):
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            mask = cv2.resize(
                mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        mask = (mask > 127).astype(np.uint8)

        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image_tensor = torch.from_numpy(image).float()

        if self.target_format == "bce":
            mask_array = np.ascontiguousarray(
                mask[None, ...].astype(np.float32)
            )
            mask_tensor = torch.from_numpy(mask_array).float()
        else:
            mask_array = np.ascontiguousarray(mask.astype(np.int64))
            mask_tensor = torch.from_numpy(mask_array).long()

        return image_tensor, mask_tensor, str(image_path.resolve())
