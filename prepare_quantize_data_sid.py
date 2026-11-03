#!/usr/bin/env python3
import os
import pickle
from pathlib import Path
import numpy as np
import exifread
import rawpy
from models.net_mge import Network
from utils import RawUtils
import random
import argparse  
import cv2


def get_iso_from_arw(arw_file):
    with open(arw_file, 'rb') as f:
        tags = exifread.process_file(f, stop_tag='ISOSpeedRatings')
        iso_tag = 'EXIF ISOSpeedRatings'
        if iso_tag in tags:
            return int(str(tags[iso_tag]))
        else:
            return None


def get_raw_arw(arw_file):
    try:
        with rawpy.imread(arw_file) as raw:
            raw_image = raw.raw_image_visible.astype(np.float32)
            black_level = raw.black_level_per_channel[0]
            white_level = raw.white_level

            raw_image = (raw_image - black_level) / (white_level - black_level)
            raw_image = np.clip(raw_image, 0, 1)

            def get_bayer_pattern(raw):
                try:
                    pattern = raw.raw_pattern  # shape: (2, 2)，值为 0-3
                    colors = raw.color_desc.decode('ascii')  # 通常是 'RGBG'
                    bayer_chars = [colors[i] for i in pattern.flatten()]
                    return ''.join(bayer_chars)
                except Exception as e:
                    print(f"获取 Bayer 格式失败：{e}")
                    return 'Unknown'

            bayer_pattern = get_bayer_pattern(raw)

        iso = get_iso_from_arw(arw_file)
        return raw_image, iso, bayer_pattern

    except rawpy.LibRawError as e:
        print(f"[{arw_file}] 发生错误: {e}")
        return None, None, None


class KSigma:
    def __init__(self, K_coeff, B_coeff, anchor, V: float = 959.0):
        self.K = np.poly1d(K_coeff)
        self.Sigma = np.poly1d(B_coeff)
        self.anchor = anchor
        self.V = V

    def __call__(self, img_01, iso: float, inverse=False):
        k, sigma = self.K(iso), self.Sigma(iso)
        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        img = img_01 * self.V
        if not inverse:
            img = img * cvt_k + cvt_b
        else:
            img = (img - cvt_b) / cvt_k

        return img / self.V


class Denoiser:
    def __init__(self, ksigma: KSigma, block_h: int, block_w: int, inp_scale=255.0):  # 🔥 传入参数


        self.ksigma = ksigma
        self.inp_scale = inp_scale
        self.total_time = 0
        self.index = 0
        self.block_h = block_h
        self.block_w = block_w

    def pre_process(self, bayer_01: np.ndarray):
        rggb = RawUtils.bayer2rggb(bayer_01)
        rggb = rggb.clip(0, 1)

        H, W = rggb.shape[:2]
        ph = (32 - H % 32) % 32
        pw = (32 - W % 32) % 32

        ph_top = ph // 2
        ph_bottom = ph - ph_top
        pw_left = pw // 2
        pw_right = pw - pw_left

        rggb = np.pad(rggb,
                      [(ph_top, ph_bottom), (pw_left, pw_right), (0, 0)],
                      mode='constant')
        inp_rggb = rggb.transpose(2, 0, 1)[np.newaxis]
        self.ph, self.pw = ph, pw
        return inp_rggb

    def run(self, bayer_01: np.ndarray, iso: float):
        inp_rggb_01 = self.pre_process(bayer_01)
        inp_rggb = self.ksigma(inp_rggb_01, iso) * self.inp_scale
        inp = np.ascontiguousarray(inp_rggb)
        inp = np.clip(inp, 0, 255)
        print(f"inp_rggb min: {inp_rggb.min()}, max: {inp_rggb.max()}")
        print(f"inp min: {inp.min()}, max: {inp.max()}, mean: {inp.mean()}")

        num_to_save = 16
        block_h, block_w = self.block_h, self.block_w  
        B, C, H, W = inp.shape
        num_blocks_h = H // block_h
        num_blocks_w = W // block_w
        # print(f"num_blocks_w*num_blocks_h:{num_blocks_w*num_blocks_h}")

        all_positions = [(i, j) for i in range(num_blocks_h) for j in range(num_blocks_w)]
        selected_positions = random.sample(all_positions, min(num_to_save, len(all_positions)))

        result_dir = f"calib_data_{2*block_h}x{2*block_w}/"
        os.makedirs(result_dir, exist_ok=True)
        for i, (h_idx, w_idx) in enumerate(selected_positions):
            h_start = h_idx * block_h
            w_start = w_idx * block_w
            block = inp[:, :, h_start:h_start + block_h, w_start:w_start + block_w]
            np.save(f'{result_dir}/img_{self.index}.npy', block.astype(np.float32))
            self.index += 1

        return 0


def process_arw_folder_recursive(root_folder, block_h, block_w):  


    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=1600,
    )
    denoiser = Denoiser(ksigma, block_h, block_w)  #
    processed_iso_counts = {}

    for root, _, files in os.walk(root_folder):
        for name in sorted(files):
            if name.lower().endswith('.arw'):
                arw_path = os.path.join(root, name)
                raw_img, iso, bayer_pattern = get_raw_arw(arw_path)

                if raw_img is None or iso is None or bayer_pattern is None:
                    continue

                max_count = 4
                current_count = processed_iso_counts.get(iso, 0)
                if current_count >= max_count:
                    continue

                print(f"[{arw_path}] 正在处理，ISO={iso}, Bayer={bayer_pattern}")

                try:
                    if bayer_pattern == 'BGGR':
                        raw_img = RawUtils.bggr2rggb(raw_img)
                    elif bayer_pattern == 'RGGB':
                        pass
                    else:
                        print(f"[{arw_path}] 未支持的 Bayer 格式：{bayer_pattern}，跳过。")
                        continue

                    denoiser.run(raw_img, iso=iso)
                    processed_iso_counts[iso] = current_count + 1

                except Exception as e:
                    print(f"[{arw_path}] 处理失败，错误：{e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_h", type=int, required=True, help="块高度")
    parser.add_argument("--block_w", type=int, required=True, help="块宽度")
    parser.add_argument("--data_dir", type=str, default="./dataset/Sony/short", help="ARW 文件根目录")
    args = parser.parse_args()

    print(f"\n========= 正在处理文件夹：{args.data_dir} =========")
    process_arw_folder_recursive(args.data_dir, args.block_h, args.block_w)
