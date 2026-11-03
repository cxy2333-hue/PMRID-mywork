#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from utils import RawUtils
import tflite_runtime.interpreter as tflite
import time
import os
from denoiser import KSigma,Denoiser, parse_tflite_model_info


def run_test(model_path,frame):

    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=400,
    )

    sizes, dtype =parse_tflite_model_info(model_path)
    size=sizes[0]
    print(size, dtype)

    denoiser = Denoiser(model_path, ksigma)
    W, H = 2048, 1536      # 测试图像的宽高（完整的单帧数据）


    folder_path = f'imtest'
    result_dir = f'imresult'
    os.makedirs(result_dir, exist_ok=True)

    light = 60     # 想图像显示的亮度（提亮）， 方便观察噪声
    iso = 6400     # 固定参数（低光照条件下）


    for file in sorted(os.listdir(folder_path)):
        if not file.endswith('.raw'):
            continue
        file_path = os.path.join(folder_path, file)
        file_id = os.path.splitext(file)[0]  # 获取编号部分，例如 '123' from '123.raw'

        if int(file_id) != frame:       # 单帧测试
            continue

        print(f"frame:{file_id}")
        with open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        input_bayer = data.reshape(H, W)
        raw_demosaicing = cv2.demosaicing(input_bayer, cv2.COLOR_BayerRG2BGR)      

        # 计算图像的亮度
        gray = cv2.cvtColor(raw_demosaicing, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)    

        # 计算调到目标亮度需要缩放的倍数
        cv_rate = light/brightness 

        
        out_path = os.path.join(result_dir, f"{file_id}_raw.png")       # 可视化带噪原始图像
        raw_demosaicing=np.clip(raw_demosaicing*cv_rate, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path, raw_demosaicing)

        # 转 RGGB 格式 + 降噪
        input_bayer = RawUtils.bggr2rggb(input_bayer)

        # 推理结果
        pred_bayer = denoiser.run(input_bayer / 255.0, iso=iso,size=size) 

        pred_bayer = RawUtils.rggb2bggr(pred_bayer)
        pred_bayer = np.clip(pred_bayer, 0, 1)

        
        # # Demosaic 推理图
        pred_bayer = (pred_bayer * 255.0).astype(np.uint8)
        # outraw=os.path.join(result_dir, f"{file_id}_denoised_block{size}_{dtype}.raw")
        # pred_bayer.tofile(outraw)       # 保存为二进制文件
      
        pred_bayer_uint8 = np.clip(pred_bayer, 0, 255).astype(np.uint8)

        # opencv 的 Demosaicing，用于可视化
        pred_demosaicing = cv2.demosaicing(pred_bayer_uint8, cv2.COLOR_BayerRG2BGR)

        # 计算图像亮度
        gray = cv2.cvtColor(pred_demosaicing, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        # print(f"brightness pred:{brightness}")

        # 计算调到目标亮度需要缩放的倍数
        cv_rate=light/brightness

        out_path = os.path.join(result_dir, f"{file_id}_denoised_{iso}_tflite_block{size}_{dtype}.png")

        # 图像亮度调高 可视化
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
    parser.add_argument('--model', type=str, default="./model/model_64_64_int16x8.tflite")
    parser.add_argument('--frame', type=int, default=130)
    args = parser.parse_args()

    run_test(args.model,  args.frame)

    
## 用法： python run_test_tflite_single_block_video.py --model ./model/model_64_64_int16x8.tflite --frame 150 
##  也可以直接改 parser.add_argument('--frame', type=int, default=130)里的default值 进行单帧测试 python run_test_tflite_single_block_video.py即可
if __name__ == "__main__":
    main()

