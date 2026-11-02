#!/usr/bin/env python3
"""PyTorch 版 PMRID 训练数据集与噪声增广。

由 dataset/training.py (MegEngine 版) 改写，适配 train.py 的调用方式：
  - aug_opts = DataAugOptions.parse_file(json_path)
  - full_dataset = CleanRawImages(data_dir=..., output_shape=..., repeats=...)
  - DataLoader 迭代得到 (img: float32 Tensor [4, h, w] in [0,1], g_mean: float)
  - noisy, gt, cvt_k, cvt_b = DataAug.transform(batch_img01, batch_g_mean)

数据目录结构（data_dir）:
  index.json  # RawImageItem 列表: path/width/height/black_level/white_level/bayer_pattern/g_mean_01
  *.raw       # uint16 原始 raw 图
"""
import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BayerPattern(str, Enum):
    RGGB = "RGGB"
    BGGR = "BGGR"
    GRBG = "GRBG"
    GBRG = "GBRG"


# ---------------------------------------------------------------------------
# 配置结构（等价于原 pydantic 模型，免依赖）
# ---------------------------------------------------------------------------
@dataclass
class RawImageItem:
    path: str
    width: int
    height: int
    black_level: int
    white_level: int = 65535
    bayer_pattern: BayerPattern = BayerPattern.BGGR
    g_mean_01: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "RawImageItem":
        return cls(
            path=d["path"],
            width=int(d["width"]),
            height=int(d["height"]),
            black_level=int(d["black_level"]),
            white_level=int(d.get("white_level", 65535)),
            bayer_pattern=BayerPattern(d.get("bayer_pattern", "BGGR")),
            g_mean_01=float(d.get("g_mean_01", 0.0)),
        )


@dataclass
class NoiseProfile:
    K: Tuple[float, float]           # k(iso) 拟合多项式系数（高次在前）
    B: Tuple[float, float, float]    # sigma^2(iso) 拟合多项式系数
    value_scale: float = 959.0


@dataclass
class DataAugOptions:
    noise_profile: NoiseProfile
    camera_value_scale: float = 959.0
    iso_range: Tuple[float, float] = (100.0, 3200.0)
    anchor_iso: float = 1600.0
    output_shape: Tuple[int, int] = (512, 512)          # raw 域裁剪尺寸 (H, W)
    target_brighness_range: Tuple[float, float] = (0.02, 0.5)
    output_scale: float = 1.0  # 输出归一化: k-sigma 域 /camera_value_scale 后 clamp(0,1) 再乘 output_scale

    @classmethod
    def from_dict(cls, d: dict) -> "DataAugOptions":
        p = d["noise_profile"]
        return cls(
            noise_profile=NoiseProfile(
                K=tuple(p["K"]),
                B=tuple(p["B"]),
                value_scale=float(p.get("value_scale", 959.0)),
            ),
            camera_value_scale=float(d.get("camera_value_scale", 959.0)),
            iso_range=tuple(d.get("iso_range", (100.0, 6400.0))),
            anchor_iso=float(d.get("anchor_iso", 1600.0)),
            output_shape=tuple(d.get("output_shape", (512, 512))),
            target_brighness_range=tuple(d.get("target_brighness_range", (0.02, 0.5))),
            output_scale=float(d.get("output_scale", 1.0)),
        )

    @classmethod
    def parse_file(cls, path) -> "DataAugOptions":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# 干净 raw 数据集
# ---------------------------------------------------------------------------
class CleanRawImages(Dataset):

    def __init__(self, *, data_dir: Optional[str] = None, index_file: Optional[str] = None,
                 output_shape: Tuple[int, int] = (512, 512), repeats: int = 1):
        """
        Args:
            - data_dir:    包含 index.json 和 raw 图像的目录
            - index_file:  index.json 的绝对路径（与 data_dir 二选一）
            - output_shape: (H, W) raw 域随机裁剪尺寸，需为 2 的倍数
            - repeats:     数据重复次数（等效扩大一个 epoch 的样本数）
        """
        super().__init__()
        if index_file is not None:
            index_path = Path(index_file)
            data_dir = index_path.parent
        else:
            assert data_dir is not None
            data_dir = Path(data_dir)
            index_path = data_dir / "index.json"

        self.output_shape = tuple(output_shape)
        self.filelist: List[RawImageItem] = []
        with open(index_path, "r") as f:
            for x in json.load(f):
                item = RawImageItem.from_dict(x)
                if not os.path.isabs(item.path):
                    item.path = str(data_dir / item.path)
                self.filelist.append(item)

        assert len(self.filelist) > 0, f"index.json 中没有数据: {index_path}"
        if repeats > 1:
            self.filelist = self.filelist * repeats

    def __len__(self):
        return len(self.filelist)

    def random_flip_and_crop(self, img: np.ndarray, src_pattern: BayerPattern) -> np.ndarray:
        """随机翻转 + Bayer 对齐裁剪，并把 bayer pattern 归一化为 RGGB。"""
        flip_ud = np.random.rand() > 0.5
        flip_lr = np.random.rand() > 0.5

        if src_pattern == BayerPattern.RGGB:
            crop_x_offset, crop_y_offset = 0, 0
        elif src_pattern == BayerPattern.GBRG:
            crop_x_offset, crop_y_offset = 0, 1
        elif src_pattern == BayerPattern.GRBG:
            crop_x_offset, crop_y_offset = 1, 0
        else:  # BGGR
            crop_x_offset, crop_y_offset = 1, 1

        if flip_lr:
            crop_x_offset = (crop_x_offset + 1) % 2
        if flip_ud:
            crop_y_offset = (crop_y_offset + 1) % 2

        H0, W0 = img.shape
        tH, tW = self.output_shape
        assert H0 >= tH and W0 >= tW, \
            f"图像 {W0}x{H0} 小于裁剪尺寸 {tW}x{tH}"

        x0 = np.random.randint(0, W0 - tW + 1)
        y0 = np.random.randint(0, H0 - tH + 1)
        x0 = x0 // 2 * 2 + crop_x_offset
        y0 = y0 // 2 * 2 + crop_y_offset
        x0 = min(x0, W0 - tW)
        y0 = min(y0, H0 - tH)

        img_crop = img[y0:y0 + tH, x0:x0 + tW]
        if flip_lr:
            img_crop = np.flip(img_crop, axis=1)
        if flip_ud:
            img_crop = np.flip(img_crop, axis=0)

        return img_crop

    def __getitem__(self, index: int):
        item = self.filelist[index]
        rawimg = np.fromfile(item.path, dtype=np.uint16).reshape((item.height, item.width))

        # 防退化：裁剪区域可能落在全黑/全白平坦区，重试若干次取有动态范围的裁剪
        crop = None
        for _ in range(4):
            crop = self.random_flip_and_crop(rawimg, item.bayer_pattern).astype(np.float32)
            if crop.max() - crop.min() > 1.0:   # raw 域至少 1 个灰阶的动态范围
                break

        raw01 = (crop - item.black_level) / (item.white_level - item.black_level)

        H, W = raw01.shape
        # pixel shuffle 打包为 4 通道 RGGB: (H//2, W//2, 4)
        rggb01 = raw01.reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3).reshape(H // 2, W // 2, 4)

        # [h, w, 4] -> [4, h, w] float32 tensor
        img_t = torch.from_numpy(np.ascontiguousarray(rggb01.transpose(2, 0, 1)))
        return img_t, float(item.g_mean_01)


# ---------------------------------------------------------------------------
# 噪声模型与增广
# ---------------------------------------------------------------------------
class NoiseProfileFunc:

    def __init__(self, noise_profile: NoiseProfile):
        self.polyK = np.poly1d(noise_profile.K)
        self.polyB = np.poly1d(noise_profile.B)
        self.value_scale = noise_profile.value_scale

    def __call__(self, iso, value_scale=959.0):
        """返回该 ISO 下的 k(增益) 和 b(读出噪声方差)。iso 可为标量或数组。"""
        r = value_scale / self.value_scale
        k = self.polyK(iso) * r
        b = self.polyB(iso) * r * r
        # 钳位防退化: k->0 会让 cvt_k 爆炸成 inf; 负读出方差无物理意义
        k = np.maximum(k, 1e-6)
        b = np.maximum(b, 0.0)
        return k, b


class DataAug:

    def __init__(self, opts: DataAugOptions):
        self.opts = opts
        self.noise_func = NoiseProfileFunc(opts.noise_profile)

    def transform(self, batch_img01: torch.Tensor, batch_g_mean, max_retry: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            - batch_img01: [N, 4, h, w] float32 tensor, 值域 [0, 1]
            - batch_g_mean: 长度 N 的 g 均值（tensor / list / ndarray 均可）
            - max_retry: 退化样本（全0/全1/常数图）亮度重采次数

        Returns:
            - batch_noisy: k-sigma 归一化后的噪声图 [N, 4, h, w]
            - batch_gt:    k-sigma 归一化后的干净图 [N, 4, h, w]
            - cvt_k:       每个样本的归一化增益 [N, 1, 1, 1]
            - cvt_b:       每个样本的归一化偏置 [N, 1, 1, 1]
        """
        batch_gt = batch_img01 * self.opts.camera_value_scale
        batch_gt = self.brightness_aug(batch_gt, batch_g_mean)

        # 防退化：亮度增广后出现全0/全1/常数图（极端暗目标 + 暗裁剪叠加），
        # 对退化样本用安全亮度范围（避开区间最暗的 40%）重采
        for _ in range(max_retry):
            mask = self._degenerate_mask(batch_gt)
            if not mask.any():
                break
            batch_gt[mask] = self._safe_brightness(
                batch_img01[mask] * self.opts.camera_value_scale)

        n_bad = int(self._degenerate_mask(batch_gt).sum())
        if n_bad > 0:
            print(f"[DataAug][WARN] {n_bad}/{batch_gt.shape[0]} 个样本亮度重采 {max_retry} 次后仍为常数图"
                  "（裁剪区域本身全黑，建议检查数据 black_level / g_mean_01）")

        batch_noisy, isos = self.add_noise(batch_gt)
        cvt_k, cvt_b = self.k_sigma(isos)

        batch_noisy = batch_noisy * cvt_k + cvt_b
        batch_gt = batch_gt * cvt_k + cvt_b

        # 归一化到 [0, output_scale]: k-sigma 域 -> /camera_value_scale -> clamp(0,1) -> ×output_scale
        batch_noisy = batch_noisy / self.opts.camera_value_scale
        batch_gt = batch_gt / self.opts.camera_value_scale

        batch_noisy = batch_noisy.clamp(0, 1.0) * self.opts.output_scale
        batch_gt = batch_gt.clamp(0, 1.0) * self.opts.output_scale


        return batch_noisy, batch_gt, cvt_k, cvt_b

    @staticmethod
    def _degenerate_mask(t: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """检测每张图是否为常数图（全0/全1/任意常数）或含非有限值。"""
        N = t.shape[0]
        flat = t.reshape(N, -1)
        rng = flat.max(dim=1)[0] - flat.min(dim=1)[0]
        is_const = rng < eps
        is_bad = ~torch.isfinite(flat).all(dim=1)
        return is_const | is_bad

    def _safe_brightness(self, img: torch.Tensor) -> torch.Tensor:
        """重采亮度：目标值取自亮度区间的中后段（避开最暗 40%），按当前图实际均值缩放。"""
        low, high = self.opts.target_brighness_range
        tlow = low + (high - low) * 0.4
        N = img.shape[0]
        btarget = np.exp(np.random.uniform(np.log(tlow), np.log(high), size=(N, ))).astype(np.float32)
        mean01 = (img.reshape(N, -1).mean(dim=1) / self.opts.camera_value_scale).cpu().numpy().astype(np.float32)
        s = np.clip(btarget / np.maximum(mean01, 1e-6), 0.01, 1.0).astype(np.float32)
        return img * torch.from_numpy(s).view(-1, 1, 1, 1)

    def k_sigma(self, iso) -> Tuple[torch.Tensor, torch.Tensor]:
        """把任意 ISO 的 (k, sigma) 映射到 anchor ISO 空间。"""
        k, sigma = self.noise_func(iso, value_scale=self.opts.camera_value_scale)
        k_a, sigma_a = self.noise_func(self.opts.anchor_iso, value_scale=self.opts.camera_value_scale)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        device = "cpu"
        cvt_k = torch.as_tensor(np.asarray(cvt_k, dtype=np.float32).reshape(-1, 1, 1, 1), device=device)
        cvt_b = torch.as_tensor(np.asarray(cvt_b, dtype=np.float32).reshape(-1, 1, 1, 1), device=device)
        return cvt_k, cvt_b

    def brightness_aug(self, img: torch.Tensor, orig_gmean) -> torch.Tensor:
        """把每张图的亮度随机调整到 [low, high]（log 均匀采样）。"""
        low, high = self.opts.target_brighness_range
        g = np.asarray(np.clip(np.asarray(orig_gmean, dtype=np.float64).reshape(-1), 1e-6, None))
        N = g.shape[0]
        btarget = np.exp(np.random.uniform(np.log(low), np.log(high), size=(N, )))
        s = np.clip(btarget / g, 0.01, 1.0).astype(np.float32)
        return img * torch.from_numpy(s).view(-1, 1, 1, 1)

    def add_noise(self, img: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Args:
            - img: [N, 4, h, w]，值域 [0, camera_value_scale]

        Returns:
            - noisy: 加噪后的图（含量化 round）
            - isos:  每个 sample 采样的 ISO
        """
        N = img.shape[0]
        isos = np.random.uniform(*self.opts.iso_range, size=(N, ))
        k, b = self.noise_func(isos, value_scale=self.opts.camera_value_scale)

        k_t = torch.as_tensor(np.asarray(k, dtype=np.float32)).view(-1, 1, 1, 1)
        b_t = torch.as_tensor(np.asarray(b, dtype=np.float32)).view(-1, 1, 1, 1)

        # Poisson-Gaussian: lambda = img/k (光子计数, 任意正数, 不做 clip),
        # lambda 钳 0 防止 black_level 减除后残留的负值导致 NaN
        shot_noisy = torch.poisson((img / k_t).clamp_min(0.0)) * k_t
        read_noisy = torch.randn_like(img) * torch.sqrt(b_t)
        noisy = torch.round(shot_noisy + read_noisy)
        return noisy, isos


def rggb_to_rgb01(img4: np.ndarray, gamma: float = 1 / 2.2) -> np.ndarray:
    """RGGB 4 通道 -> 伪彩 RGB [h, w, 3]，用于显示。输入 [h, w, 4] 或 [4, h, w]。"""
    x = np.asarray(img4, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 4 and x.shape[-1] != 4:
        x = x.transpose(1, 2, 0)
    r, g1, g2, b = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    rgb = np.stack([r, (g1 + g2) / 2.0, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0) ** gamma


def visualize_noise(noisy, gt, cvt_k=None, cvt_b=None, save_path="noise_check.png",
                    camera_value_scale=959.0, max_samples=8, target_brightness: float = 0.5):
    """验证噪声添加是否正确：把 k-sigma 空间的 noisy/gt 反归一化回 [0,1]，
    按干净图的均值把每个样本"提亮"到 target_brightness（仅影响显示，不影响训练数据），
    保存"左噪声 | 右干净"的并排对比 PNG，并打印每个样本的统计信息。

    Args:
        - noisy/gt/cvt_k/cvt_b: 直接传 DataAug.transform() 的返回值
          （[N,4,h,w] tensor / [N,1,1,1] tensor，单个样本去掉 batch 维也可以）
        - target_brightness: 显示目标平均亮度 (0, 1]，暗图会被提亮便于观察噪声结构
    Returns:
        - save_path
    """
    def _batch01(t, k, b):
        if torch.is_tensor(t):
            t = t.detach().cpu().numpy()
        t = np.asarray(t, dtype=np.float32)
        if t.ndim == 3:
            t = t[None]
        if k is not None:
            if torch.is_tensor(k):
                k = k.detach().cpu().numpy()
            if torch.is_tensor(b):
                b = b.detach().cpu().numpy()
            k = np.asarray(k, np.float32).reshape(-1, 1, 1, 1)
            b = np.asarray(b, np.float32).reshape(-1, 1, 1, 1)
            t = (t - b) / np.maximum(k, 1e-6)
        return np.clip(t / camera_value_scale, 0.0, 1.0)

    noisy01 = _batch01(noisy, cvt_k, cvt_b)
    gt01 = _batch01(gt, cvt_k, cvt_b)
    N = min(noisy01.shape[0], max_samples)

    # 显示增益: 以干净图均值为基准, 每样本独立提亮到 target_brightness（只提亮不压暗）
    gains = []
    for i in range(N):
        m = float(gt01[i].mean())
        gains.append(float(np.clip(target_brightness / max(m, 1e-6), 1.0, 1e4)))

    rows = []
    for i in range(N):
        rgb_noisy = rggb_to_rgb01(np.clip(noisy01[i] * gains[i], 0.0, 1.0))
        rgb_gt = rggb_to_rgb01(np.clip(gt01[i] * gains[i], 0.0, 1.0))
        gap = np.ones((rgb_gt.shape[0], 6, 3), dtype=np.float32)
        rows.append(np.concatenate([rgb_noisy, gap, rgb_gt, gap], axis=1))
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(save_path), (grid * 255).astype(np.uint8)[..., ::-1])

    print(f"已保存噪声对比图: {save_path}  (每行: 噪声图 | 干净图, 共 {N} 个样本, 显示目标亮度 {target_brightness})")
    for i in range(N):
        n, g = noisy01[i], gt01[i]
        noise_std = float((n - g).std())
        flags = []
        if n.max() - n.min() < 1e-4:
            flags.append("noisy为常数图")
        if g.max() - g.min() < 1e-4:
            flags.append("gt为常数图")
        flag_str = ("  [WARN] " + ", ".join(flags)) if flags else ""
        print(f"  样本{i}: 原始gt均值={g.mean():.4f} 显示增益x{gains[i]:.1f} | "
              f"noisy[min={n.min():.4f} max={n.max():.4f} mean={n.mean():.4f}] "
              f"gt[min={g.min():.4f} max={g.max():.4f}] "
              f"噪声std={noise_std:.4f}{flag_str}")
    return save_path


if __name__ == "__main__":
    # 自检: python dataset.py —— 含一张全黑图，验证防退化逻辑和可视化
    import tempfile
    h, w = 256, 256
    d = tempfile.mkdtemp(prefix="pmrid_ds_")
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 300 + 200 * np.sin(xx / 57.0) + 150 * np.cos(yy / 43.0) + rng.normal(0, 12, (h, w))
    raw1 = np.clip(base, 64, 1023).astype(np.uint16)
    raw2 = np.full((h, w), 64, dtype=np.uint16)   # 全黑图, 故意触发防退化路径
    open(os.path.join(d, "img0.raw"), "wb").write(raw1.tobytes())
    open(os.path.join(d, "img1.raw"), "wb").write(raw2.tobytes())
    json.dump([
        {"path": "img0.raw", "width": w, "height": h, "black_level": 64, "white_level": 1023,
         "bayer_pattern": "BGGR", "g_mean_01": 0.2},
        {"path": "img1.raw", "width": w, "height": h, "black_level": 64, "white_level": 1023,
         "bayer_pattern": "BGGR", "g_mean_01": 0.0},
    ], open(os.path.join(d, "index.json"), "w"))

    ds = CleanRawImages(data_dir=d, output_shape=(128, 128))
    img, gmean = ds[0]
    print("img:", tuple(img.shape), img.dtype, "| g_mean:", gmean)

    opts = DataAugOptions(
        # 噪声曲线示例 (a7SII 风格, V=959 约定): k(iso)=polyK(iso), sigma^2(iso)=polyB(iso)
        # 正式训练时用 SID 短/长曝对拟合的真实参数替换
        noise_profile=NoiseProfile(K=(0.0005995267, 0.00868861),
                                   B=(7.11772e-7, 6.514934e-4, 0.11492713)),
        iso_range=(100.0, 6400.0),
        # target_brighness_range 保持默认 (0.02, 0.5) 真实暗光分布;
        # 可视化由 visualize_noise 的显示增益(target_brightness)负责提亮
    )
    aug = DataAug(opts)
    batch = torch.stack([ds[0][0], ds[1][0]])
    gm = torch.tensor([ds[0][1], ds[1][1]])
    noisy, gt, k, b = aug.transform(batch, gm)
    visualize_noise(noisy, gt, k, b, save_path="/tmp/pmrid_noise_check.png",
                    camera_value_scale=opts.camera_value_scale)
    print("dataset.py self-test OK")
