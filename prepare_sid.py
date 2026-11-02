#!/usr/bin/env python3
"""SID 长曝干净图筛选脚本（论文流程复现）。

论文: "we choose the 10s and 30s long-exposure subset captured by the Sony
α7s II camera, and manually take out those with visible noise, leaving
214 high quality RAW images"

三步工作流:
  1. 抽取:   python prepare_sid.py --src dataset/Sony2025.zip --out dataset/sid_clean --extract
             (从 zip 只解压 *_10s.ARW / *_30s.ARW 到 <out>/Sony/long/)
  2. 预览:   python prepare_sid.py --src dataset/sid_clean/Sony/long --out dataset/sid_clean --preview
             (每张图生成提亮中心裁剪预览 + 拼图 montage + stats.csv, 供人工剔除)
  3. 建索引: python prepare_sid.py --src dataset/sid_clean/Sony/long --out dataset/sid_clean --index \
                 --exclude dataset/sid_clean/exclude.txt
             (人工看过预览后, 把要剔除的文件名一行一个写进 exclude.txt, 生成 index.json)

输出 index.json 供 dataset.py 的 CleanRawImages 使用。
"""
import argparse
import csv
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

from dataset import rggb_to_rgb01

EXPOSURE_RE = re.compile(r"_(\d+(?:\.\d+)?)s\.ARW$", re.IGNORECASE)
PREVIEW_TILE = 256          # 单个预览 tile 尺寸
MONTAGE_COLS = 5            # montage 每行图数


def select_long_exposure(names, exposures=("10", "30")):
    """从文件名筛出指定曝光时长的长曝图, 返回 [(name, exposure_sec), ...]"""
    out = []
    for n in names:
        base = os.path.basename(n)
        m = EXPOSURE_RE.search(base)
        if not m:
            continue
        if str(int(float(m.group(1)))) in exposures:
            out.append((n, float(m.group(1))))
    return sorted(out)


def cmd_extract(args):
    """从 zip / 目录抽取指定曝光的长曝 ARW 到 <out>/Sony/long/"""
    src, out = Path(args.src), Path(args.out)
    dst_dir = out / "Sony" / "long"
    dst_dir.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        names = [str(p) for p in src.rglob("*.ARW")]
        selected = select_long_exposure(names, args.exposures)
        n_done = 0
        for name, _sec in selected:
            dst = dst_dir / os.path.basename(name)
            if dst.exists():
                continue
            src_path = src / os.path.basename(name) if (src / os.path.basename(name)).exists() else Path(name)
            n_done += 1
            print(f"链接/复制: {src_path} -> {dst}")
            try:
                os.link(src_path, dst)      # 同盘硬链接, 零拷贝
            except OSError:
                import shutil
                shutil.copy2(src_path, dst)
    elif src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as zf:
            selected = select_long_exposure(zf.namelist(), args.exposures)
            print(f"zip 中共 {len(selected)} 张 {args.exposures}s 长曝图")
            n_done = 0
            for i, (name, _sec) in enumerate(selected):
                dst = dst_dir / os.path.basename(name)
                if dst.exists():
                    continue
                n_done += 1
                print(f"[{i+1}/{len(selected)}] 解压: {name}")
                with zf.open(name) as fsrc, open(dst, "wb") as fdst:
                    fdst.write(fsrc.read())
    else:
        sys.exit(f"--src 需要是 zip 文件或 Sony/long 目录: {src}")

    kept = len(list(dst_dir.glob("*.ARW")))
    print(f"完成, 本次处理 {n_done} 张, 目录现有 {kept} 张 ARW -> {dst_dir}")


def decode_arw(path):
    """rawpy 解码 ARW, 返回 (visible_raw uint16 [H,W], meta dict)"""
    import rawpy
    with rawpy.imread(str(path)) as raw:
        vis = raw.raw_image_visible.copy()
        pattern = np.asarray(raw.raw_pattern).tolist()
        code2ch = {0: "R", 1: "G", 2: "B", 3: "G"}   # libraw 4值CFA: 3 是第二个绿
        bayer = "".join(code2ch[c] for row in pattern for c in row)     # 如 "RGGB"
        black = int(np.round(np.mean(raw.black_level_per_channel)))
        white = int(raw.white_level)
        H, W = vis.shape
        meta = dict(width=W, height=H, black_level=black, white_level=white,
                    bayer_pattern=bayer)
    return vis, meta


def g_mean_01(vis: np.ndarray, meta: dict) -> float:
    """绿色通道平均亮度 [0,1] (作为该图 g_mean)"""
    bayer = meta["bayer_pattern"]
    blk, wht = meta["black_level"], meta["white_level"]
    img01 = (vis.astype(np.float32) - blk) / max(wht - blk, 1)
    H, W = img01.shape
    # (H, W) -> (H//2, W//2, 2, 2): 前两维是 2x2 block 索引, 后两维是 block 内偏移
    m = img01[: H // 2 * 2, : W // 2 * 2].reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3)
    gmask = np.array([[c == "G" for c in bayer[i * 2:(i + 1) * 2]] for i in range(2)])
    return float(m[..., gmask].mean())


def read_iso(path) -> str:
    """读 EXIF ISO (供后续噪声参数拟合用), 失败返回空"""
    try:
        import exifread
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False, stop_tag="ISOSpeedRatings")
        for key in ("EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity"):
            if key in tags:
                return str(tags[key])
    except Exception:
        pass
    return ""


def iso_in_range(iso_str, iso_min: float, iso_max: float) -> bool:
    """ISO 是否在 [iso_min, iso_max] 内; EXIF 读取失败不误杀, 交给人工/exclude.txt"""
    if not iso_str:
        return True
    try:
        v = float(iso_str)
    except ValueError:
        return True
    return iso_min <= v <= iso_max


def make_preview(vis: np.ndarray, meta: dict, target_brightness=0.5):
    """中心裁剪 + 提亮到目标亮度, 返回 BGR uint8 tile (raw 裁 2*PREVIEW_TILE, 打包后 RGB 为 PREVIEW_TILE)"""
    H, W = vis.shape
    t = min(PREVIEW_TILE * 2, H, W)
    y0, x0 = (H - t) // 2, (W - t) // 2
    crop = vis[y0:y0 + t, x0:x0 + t]
    img01 = (crop.astype(np.float32) - meta["black_level"]) / max(meta["white_level"] - meta["black_level"], 1)
    gain = float(np.clip(target_brightness / max(img01.mean(), 1e-6), 1.0, 1e4))
    # 单通道 raw 打包为 RGGB 4 通道 (与 dataset.py 的 CleanRawImages 一致)
    H, W = img01.shape
    img4 = img01[: H // 2 * 2, : W // 2 * 2].reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3).reshape(H // 2, W // 2, 4)
    rgb = rggb_to_rgb01(np.clip(img4 * gain, 0.0, 1.0))
    return (rgb[..., ::-1] * 255).astype(np.uint8)


def scan_stats(src: Path, args, prev_dir: Path = None):
    """扫描目录: 读 ISO 过滤 + rawpy 解码 + g_mean 统计。
    prev_dir 非空时顺带生成预览 tile。返回 (rows, tiles)。"""
    files = sorted(src.glob("*.ARW"))
    if not files:
        sys.exit(f"目录中没有 ARW: {src}")
    rows, tiles = [], []
    for i, p in enumerate(files):
        # 先读 ISO (廉价), 不在范围内直接跳过, 免去 rawpy 解码
        iso = read_iso(p)
        if not iso_in_range(iso, args.iso_min, args.iso_max):
            print(f"[{i+1}/{len(files)}] {p.name} ISO={iso} 超出 [{args.iso_min:g}, {args.iso_max:g}], 跳过")
            continue
        try:
            vis, meta = decode_arw(p)
        except Exception as e:
            print(f"[{i+1}/{len(files)}] 解码失败 {p.name}: {e}")
            rows.append(dict(name=p.name, iso=iso, error=str(e)[:80]))
            continue
        gm = g_mean_01(vis, meta)
        rows.append(dict(name=p.name, iso=iso, g_mean_01=round(gm, 6), **meta))
        png = None
        if prev_dir is not None:
            png = prev_dir / (p.stem + ".png")
            if not png.exists():
                cv2.imwrite(str(png), make_preview(vis, meta))
            tiles.append((p.name, png))
        print(f"[{i+1}/{len(files)}] {p.name} ISO={iso} g_mean={gm:.4f} {meta['width']}x{meta['height']} "
              f"black={meta['black_level']} pattern={meta['bayer_pattern']}")
        if args.max_images and i + 1 >= args.max_images:
            break
    return rows, tiles


def write_stats_csv(out: Path, rows):
    with open(out / "stats.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def cmd_preview(args):
    """生成 stats.csv + 每图预览 tile + montage 拼图, 供人工剔除可见噪声图"""
    src, out = Path(args.src), Path(args.out)
    prev_dir = out / "preview"
    prev_dir.mkdir(parents=True, exist_ok=True)

    rows, tiles = scan_stats(src, args, prev_dir)
    if not rows:
        sys.exit(f"没有图片通过筛选 (ISO 范围 [{args.iso_min:g}, {args.iso_max:g}]), 请放宽 --iso-min/--iso-max")
    write_stats_csv(out, rows)

    # montage 拼图 (每 MONTAGE_COLS*MONTAGE_COLS 张一张, 文件名写在图上)
    per = MONTAGE_COLS * MONTAGE_COLS
    gi = 0
    for gi in range(0, len(tiles), per):
        group = tiles[gi:gi + per]
        grid = np.zeros((MONTAGE_COLS * (PREVIEW_TILE + 18), MONTAGE_COLS * PREVIEW_TILE, 3), np.uint8)
        for j, (name, png) in enumerate(group):
            r, c = divmod(j, MONTAGE_COLS)
            y, x = r * (PREVIEW_TILE + 18), c * PREVIEW_TILE
            tile = cv2.imread(str(png))
            grid[y:y + PREVIEW_TILE, x:x + PREVIEW_TILE] = tile
            cv2.putText(grid, name.replace(".ARW", ""), (x + 4, y + PREVIEW_TILE + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.imwrite(str(prev_dir / f"montage_{gi//per:02d}.png"), grid)

    print(f"\n预览完成: {len(tiles)} 张 -> {prev_dir}/ (montage_00.png 从小到大浏览)")
    print(f"统计: {out/'stats.csv'}")
    print("人工剔除: 把可见噪声/坏图文件名写进 exclude.txt (一行一个, 含 .ARW), 然后运行 --index")


def cmd_index(args):
    """应用 exclude.txt, 生成 index.json"""
    src, out = Path(args.src), Path(args.out)
    exclude = set()
    if args.exclude and Path(args.exclude).exists():
        exclude = {l.strip() for l in open(args.exclude) if l.strip()}

    # 优先复用 stats.csv 避免重复解码; 但文件集或 ISO 范围与当前参数不一致时自动重扫
    stats_path = out / "stats.csv"
    rows = None
    if stats_path.exists():
        rows = list(csv.DictReader(open(stats_path)))
        cached_names = {r["name"] for r in rows if r.get("name")}
        want_names = {p.name for p in src.glob("*.ARW")
                      if iso_in_range(read_iso(p), args.iso_min, args.iso_max)}
        if cached_names != want_names:
            print(f"stats.csv 与当前文件集/ISO范围 [{args.iso_min:g}, {args.iso_max:g}] 不一致, 重新扫描...")
            rows = None
    if rows is None:
        rows, _ = scan_stats(src, args)
        if not rows:
            sys.exit(f"没有图片通过筛选 (ISO 范围 [{args.iso_min:g}, {args.iso_max:g}]), 请放宽 --iso-min/--iso-max")
        write_stats_csv(out, rows)

    items, skipped = [], []
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = row["name"]
        if "error" in row and row.get("error"):
            skipped.append(name)
            continue
        if not iso_in_range(row.get("iso", ""), args.iso_min, args.iso_max):
            skipped.append(name)
            continue
        if name in exclude:
            skipped.append(name)
            continue
        p = src / name
        if not p.exists():
            skipped.append(name)
            continue
        # ARW 容器 -> 裸 uint16 raw 落盘 (CleanRawImages 直接读 raw)
        raw_path = raw_dir / (p.stem + ".raw")
        if not raw_path.exists():
            vis, _meta = decode_arw(p)
            vis.tofile(raw_path)
        items.append(dict(
            path=str(raw_path.resolve()),
            width=int(row["width"]), height=int(row["height"]),
            black_level=int(row["black_level"]), white_level=int(row["white_level"]),
            bayer_pattern=row["bayer_pattern"], g_mean_01=float(row["g_mean_01"]),
        ))

    idx_path = out / "index.json"
    json.dump(items, open(idx_path, "w"), indent=1)
    print(f"index.json 写入 {len(items)} 张 (剔除 {len(set(skipped))} 张) -> {idx_path}")
    print("下一步: 配好 aug json 后即可训练")
    print(f"  python train.py --data-aug-config <aug.json> --data-dir {out} ...")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Sony2025.zip 或解压后的 Sony/long 目录")
    ap.add_argument("--out", required=True, help="输出目录")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--extract", action="store_true", help="步骤1: 抽取指定曝光的 ARW")
    mode.add_argument("--preview", action="store_true", help="步骤2: 生成预览和统计")
    mode.add_argument("--index", action="store_true", help="步骤3: 生成 index.json")
    ap.add_argument("--exposures", nargs="+", default=["10", "30"], help="曝光时长(秒), 默认 10 30")
    ap.add_argument("--iso-min", type=float, default=0.0, help="ISO 下限 (0=不限)")
    ap.add_argument("--iso-max", type=float, default=float("inf"), help="ISO 上限 (inf=不限), 如 --iso-max 800")
    ap.add_argument("--exclude", default=None, help="exclude.txt 路径")
    ap.add_argument("--max-images", type=int, default=0, help="preview 模式最多处理 N 张 (0=全部)")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.extract:
        cmd_extract(args)
    elif args.preview:
        cmd_preview(args)
    elif args.index:
        cmd_index(args)


if __name__ == "__main__":
    main()
