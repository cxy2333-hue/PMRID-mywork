#!/usr/bin/env python3
import argparse
from typing import Tuple
import time
import cv2

import numpy as np
import skimage.metrics
import onnxruntime as ort

from utils import RawUtils

import tflite_runtime.interpreter as tflite


import os
from scipy.ndimage import convolve
import math
from run_test_tflite_single_block import KSigma,Denoiser, parse_tflite_model_info
from run_test_onnx_single_bak import white_balance_gain, white_balance


def run_test(model_path):
    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=1600,
    )

    sizes, dtype =parse_tflite_model_info(model_path)
    size=sizes[0]
    print(size, dtype)

    denoiser = Denoiser(model_path, ksigma)
    W, H = 2048, 1536
    
    file_name="black2light_20250902_1" # window black2light

    folder_path = f'ov5640/{file_name}'
    result_dir = f'im_result/{file_name}'
    os.makedirs(result_dir, exist_ok=True)
    light = 60
    iso = 6400
    frame = 140
    for file in sorted(os.listdir(folder_path)):
        if not file.endswith('.raw'):
            continue
        file_path = os.path.join(folder_path, file)
        file_id = os.path.splitext(file)[0]  # 获取编号部分，例如 '123' from '123.raw'
        if int(file_id) != frame:
            continue
        print(f"frame:{file_id}")
        with open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        input_bayer = data.reshape(H, W)
        raw_demosaicing = cv2.demosaicing(input_bayer, cv2.COLOR_BayerRG2BGR)

        gray = cv2.cvtColor(raw_demosaicing, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        print(f"brightness raw:{brightness}")

        # 转 RGGB 格式 + 降噪
        input_bayer = RawUtils.bggr2rggb(input_bayer)
        pred_bayer = denoiser.run(input_bayer / 255.0, iso=iso,size=size)
        pred_bayer = RawUtils.rggb2bggr(pred_bayer)
        pred_bayer = np.clip(pred_bayer, 0, 1)

        
        # # Demosaic 预测图
        pred_bayer = (pred_bayer * 255.0).astype(np.uint8)
        outraw=os.path.join(result_dir, f"{file_id}_denoised_block{size}_{dtype}.raw")
        pred_bayer.tofile(outraw)


        # 图像亮度调高
       
        pred_bayer_uint8 = np.clip(pred_bayer, 0, 255).astype(np.uint8)

        # Demosaicing，得到3通道图像
        pred_demosaicing = cv2.demosaicing(pred_bayer_uint8, cv2.COLOR_BayerRG2BGR)

        gray = cv2.cvtColor(pred_demosaicing, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        print(f"brightness pred:{brightness}")
        cv_rate=light/brightness

        out_path = os.path.join(result_dir, f"{file_id}_denoised_{iso}_tflite_block{size}_{dtype}.png")
        pred_demosaicing=np.clip(pred_demosaicing*cv_rate, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path, pred_demosaicing)

        print(f"[{file}] -> saved {out_path}")


        # # 拼接左图（原始）和右图（降噪后）
        # vis_combined = np.hstack([input_demosaicing, pred_demosaicing])

        # # 保存拼接图像
        # combined_vis_path = os.path.join(result_dir, f"{file_id}_compare_{dtype}.png")
        # cv2.imwrite(combined_vis_path, vis_combined)

        # print(f"[{file}] -> saved {combined_vis_path}")



    # # 拼接原始raw的demosaic图像视频

    # print("开始拼接原始raw的demosaic视频...")
    # video_path = os.path.join(result_dir, f"{file_name}_undenoised.mp4")
    # image_files = sorted([f for f in os.listdir(result_dir) if f.endswith(f"raw_demosaicing.png")])

    # if len(image_files) == 0:
    #     print("没有找到任何图像用于生成视频")
    #     return

    # first_img_path = os.path.join(result_dir, image_files[0])
    # first_frame = cv2.imread(first_img_path)
    # height, width, _ = first_frame.shape

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))  # FPS=10

    # for img_file in image_files:
    #     img_path = os.path.join(result_dir, img_file)
    #     frame = cv2.imread(img_path)
    #     if frame.shape[:2] != (height, width):
    #         frame = cv2.resize(frame, (width, height))
    #     out.write(frame)

    # out.release()
    # print(f"已保存视频至: {video_path}")

    # # 拼接降噪后图像视频

    # print("开始拼接降噪后图像视频...")
    # video_path = os.path.join(result_dir, f"{file_name}_denoised_result.mp4")
    # image_files = sorted([f for f in os.listdir(result_dir) if f.endswith(f"pmird_demosaic_{iso}_tflite.png")])

    # if len(image_files) == 0:
    #     print("没有找到任何图像用于生成视频")
    #     return

    # first_img_path = os.path.join(result_dir, image_files[0])
    # first_frame = cv2.imread(first_img_path)
    # height, width, _ = first_frame.shape

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))  # FPS=10

    # for img_file in image_files:
    #     img_path = os.path.join(result_dir, img_file)
    #     frame = cv2.imread(img_path)
    #     if frame.shape[:2] != (height, width):
    #         frame = cv2.resize(frame, (width, height))
    #     out.write(frame)

    # out.release()
    # print(f"已保存视频至: {video_path}")


    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)

    args = parser.parse_args()

    run_test(args.model)



if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
