#!/usr/bin/env python3
"""批量生成每张训练图的加噪前后对比图。

对 index.json 里每张图: 读裸 raw -> k-sigma 域合成噪声 (aug.json) ->
保存 "左加噪 | 右干净" 的全分辨率对比 PNG 到 <out>/。

用法:
  python gen_noise_preview.py                          # 默认 dataset/sid_clean
  python gen_noise_preview.py --data-dir dataset/sid_clean --target-brightness 0.3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import DataAug, DataAugOptions, visualize_noise


def pack_rggb(img01: np.ndarray) -> np.ndarray:
    """单通道 raw [H,W] -> RGGB 4通道 [H//2, W//2, 4] (与 CleanRawImages 一致)"""
    H, W = img01.shape
    return img01[: H // 2 * 2, : W // 2 * 2].reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3).reshape(H // 2, W // 2, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/sid_clean", help="含 index.json + aug.json 的训练数据目录")
    ap.add_argument("--out", default=None, help="输出目录, 默认 <data-dir>/noise_preview")
    ap.add_argument("--target-brightness", type=float, default=0.5, help="显示目标亮度")
    ap.add_argument("--seed", type=int, default=0, help="随机种子 (ISO/亮度采样可复现)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out) if args.out else data_dir / "noise_preview"
    out.mkdir(parents=True, exist_ok=True)

    items = json.load(open(data_dir / "index.json"))
    opts = DataAugOptions.parse_file(data_dir / "aug.json")
    aug = DataAug(opts)
    print(f"共 {len(items)} 张, 噪声配置: {data_dir/'aug.json'}, 输出: {out}/")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for i, it in enumerate(items):
        stem = Path(it["path"]).stem
        dst = out / f"{stem}.png"
        if dst.exists():
            print(f"[{i+1}/{len(items)}] 已存在, 跳过 {dst.name}")
            continue
        raw = np.fromfile(it["path"], dtype=np.uint16).reshape((it["height"], it["width"]))
        raw01 = (raw.astype(np.float32) - it["black_level"]) / max(it["white_level"] - it["black_level"], 1)
        img4 = torch.from_numpy(np.ascontiguousarray(pack_rggb(raw01).transpose(2, 0, 1))).unsqueeze(0)
        noisy, gt, k, b = aug.transform(img4, [float(it["g_mean_01"])])
        visualize_noise(noisy, gt, k, b, save_path=str(dst),
                        camera_value_scale=opts.camera_value_scale,
                        max_samples=1, target_brightness=args.target_brightness)
        print(f"[{i+1}/{len(items)}] {dst.name}")

    print(f"\n完成: {len(list(out.glob('*.png')))} 张对比图 -> {out}/")


if __name__ == "__main__":
    main()
