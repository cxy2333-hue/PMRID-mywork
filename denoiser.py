#!/usr/bin/env python3
import argparse
from typing import Tuple
import cv2
import numpy as np
from utils import RawUtils

import tflite_runtime.interpreter as tflite
import os

import re



def parse_tflite_model_info(model_path: str):
    # 尺寸部分：匹配 model_32_sid 或 model_32_32 这样的格式
    size_match = re.search(r"model_(\d+)(?:_(\d+))?", model_path)

    # 精度部分：匹配 float32 / int8
    dtype_match = re.search(r"(float\d+|int\d+)", model_path)

    sizes = []
    if size_match:
        sizes.append(int(size_match.group(1)))
        if size_match.group(2):
            sizes.append(int(size_match.group(2)))

    dtype = dtype_match.group(1) if dtype_match else None

    return sizes, dtype

class KSigma:

    def __init__(self, K_coeff: Tuple[float, float], B_coeff: Tuple[float, float, float], anchor: float, V: float = 959.0):
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

    def __init__(self, model_path, ksigma: KSigma, inp_scale=256.0):
    
        
        self.net = tflite.Interpreter(model_path,num_threads=1)
        self.ksigma = ksigma
        self.inp_scale = inp_scale
        self.total_time=0
        self.input_details = self.net.get_input_details()
        self.output_details = self.net.get_output_details()

    def pre_process(self, bayer_01: np.ndarray):
        rggb = RawUtils.bayer2rggb(bayer_01)
        rggb = rggb.clip(0, 1)

        H, W = rggb.shape[:2]

        ph = (32 - H % 32) % 32
        pw = (32 - W % 32) % 32

        # 分配 padding 到上下/左右
        ph_top = ph // 2
        ph_bottom = ph - ph_top
        pw_left = pw // 2
        pw_right = pw - pw_left

        # 执行 padding（只对 H 和 W 做 padding，不动通道数）
        rggb = np.pad(rggb, 
                            [(ph_top, ph_bottom), (pw_left, pw_right), (0, 0)], 
                            mode='constant')

        print(f"rggb.shape:{rggb.shape}")

        inp_rggb = rggb.transpose(2, 0, 1)[np.newaxis]
        self.ph, self.pw = ph, pw
        return inp_rggb

    def sliding_window_inference(self, inp, block_size=32, overlap=4):
        """
        inp: BHWC 输入
        block_size: 模型输入块大小 (H=W=block_size)
        overlap: 重叠大小
        """

        B, H, W, C = inp.shape
        stride = block_size - overlap

        pred = np.zeros_like(inp, dtype=np.float32)
        weight_map = np.zeros_like(inp, dtype=np.float32)

        # 构造权重：边缘小，中心大
        yy, xx = np.meshgrid(np.linspace(-1, 1, block_size), np.linspace(-1, 1, block_size), indexing="ij")
        weight = (1 - np.abs(xx)) * (1 - np.abs(yy))  # 双线性权重
        weight = np.clip(weight, 1e-6, None)
        weight = weight[None, :, :, None]             # (1,H,W,1)，方便广播

        # 遍历滑窗
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                i_end = i + block_size
                j_end = j + block_size

                # 实际覆盖的范围
                h_slice = slice(i, min(i_end, H))
                w_slice = slice(j, min(j_end, W))
                block = inp[:, h_slice, w_slice, :]

                # 判断是否需要 pad
                pad_bottom = max(0, i_end - H)
                pad_right = max(0, j_end - W)

                if pad_bottom > 0 or pad_right > 0:
                    block = np.pad(
                        block,
                        pad_width=((0,0),(0,pad_bottom),(0,pad_right),(0,0)),
                        mode="reflect"
                    )

                # 模型推理
                self.net.allocate_tensors()
                self.net.set_tensor(self.input_details[0]['index'], block)

                # start_time = time.time()

                self.net.invoke()
                # self.total_time += time.time() - start_time
                out_block = self.net.get_tensor(self.output_details[0]['index'])

                # 去掉 pad 的部分
                out_block = out_block[:, :block.shape[1]-pad_bottom, :block.shape[2]-pad_right, :]

                # 权重也要裁剪
                w_crop = weight[:, :out_block.shape[1], :out_block.shape[2], :]

                # 加权融合
                pred[:, h_slice, w_slice, :] += out_block * w_crop
                weight_map[:, h_slice, w_slice, :] += w_crop

        # 避免除0
        pred /= np.maximum(weight_map, 1e-6)
        return pred.astype(inp.dtype)


    def run(self, bayer_01: np.ndarray, iso: float,size):
        inp_rggb_01 = self.pre_process(bayer_01)

        inp_rggb = self.ksigma(inp_rggb_01, iso) * self.inp_scale
        print(f"inp_rggb min: {inp_rggb.min()}, max: {inp_rggb.max()}")


        # 获取输入信息

        input_scale, input_zero_point = self.input_details[0]['quantization']

        if self.input_details[0]['dtype'] == np.dtype('int8'):
            
            inp_rggb = (inp_rggb / input_scale + input_zero_point).round()
            
            # inp_rggb=np.clip(inp_rggb, 0,255)
            inp = np.ascontiguousarray(inp_rggb).astype(np.int8)
  
        else:
            inp = np.ascontiguousarray(inp_rggb).astype(np.float32)

        inp = inp.transpose(0, 2, 3, 1)
        print(f"inp.shape:{inp.shape}")
        pred=self.sliding_window_inference(inp, block_size=size // 2, overlap=size // 4)

        # print(f"infer time:{self.total_time}")
        # output

        output_scale, output_zero_point = self.output_details[0]['quantization']

        if self.output_details[0]['dtype'] == np.dtype('int8'):
            pred = np.clip(pred,-128,127)
            pred = (pred - output_zero_point) * output_scale
            
            
        
        print("Input scale:", input_scale, "zero_point:", input_zero_point)
        print("Output scale:", output_scale, "zero_point:", output_zero_point)

        print(f"np.max(pred):{np.max(pred)}")
        pred = pred/self.inp_scale

        # import ipdb; ipdb.set_trace()
        pred = np.transpose(pred[0],[0,1,2])
        print(f"pred.shape:{pred.shape}")
        pred = self.ksigma(pred, iso, inverse=True)

        ph, pw = self.ph, self.pw
        # pred = pred[ph:-ph, pw:-pw]
        h0 = ph // 2
        h1 = pred.shape[0] - (ph - ph // 2)
        w0 = pw // 2
        w1 = pred.shape[1] - (pw - pw // 2)
        pred = pred[h0:h1, w0:w1,:]


        return RawUtils.rggb2bayer(pred)


if __name__ == "__main__":
    main()


