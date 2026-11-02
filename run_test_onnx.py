#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from typing import Tuple
import time
import cv2

import numpy as np
import skimage.metrics
from tqdm import tqdm

from utils import RawUtils
import onnxruntime as ort

import os
from scipy.ndimage import convolve

from skimage.util import view_as_windows

from numba import njit, prange

import numpy as np
from numba import njit

@njit
def is_bad_pixel(center, neighbors, threshold, max_val=1023):
    diffs = neighbors - center
    all_positive = np.all(diffs > 0)
    all_negative = np.all(diffs < 0)

    if not (all_positive or all_negative):
        return False

    if np.all(np.abs(diffs) > threshold):
        return True
    return False

@njit
def correct_channel(raw, corrected, mask_coords, neighbors_offsets, threshold, H, W):
    
    for idx in range(mask_coords.shape[0]):
        y, x = mask_coords[idx]

        center = raw[y, x]
        neighbors = np.empty(len(neighbors_offsets))

        for i in range(len(neighbors_offsets)):
            dy, dx = neighbors_offsets[i]
            ny = min(max(y + dy, 0), H - 1)
            nx = min(max(x + dx, 0), W - 1)
            neighbors[i] = raw[ny, nx]
        # print("neighbors:")
        # for i in range(len(neighbors)):
        #     print(neighbors[i])  # 逐个打印值
        if is_bad_pixel(center, neighbors, threshold):
            corrected[y, x] = np.median(neighbors)
            

@njit
def get_mask_coords(H, W, offset_y, offset_x, step_y=2, step_x=2):
    count = ((H - offset_y + step_y - 1) // step_y) * ((W - offset_x + step_x - 1) // step_x)
    coords = np.empty((count, 2), dtype=np.int32)
    idx = 0
    for y in range(offset_y, H, step_y):
        for x in range(offset_x, W, step_x):
            coords[idx, 0] = y
            coords[idx, 1] = x
            idx += 1
    return coords

def bad_pixel_correction_bggr_no_G_split(raw, threshold):
    H, W = raw.shape
    corrected = raw.copy()

    # 定义通道像素坐标
    mask_B_coords = get_mask_coords(H, W, 0, 0)
    mask_G1_coords = get_mask_coords(H, W, 0, 1)
    mask_G2_coords = get_mask_coords(H, W, 1, 0)
    mask_G_coords = np.vstack((mask_G1_coords, mask_G2_coords))
    mask_R_coords = get_mask_coords(H, W, 1, 1)


    BR_neighbors_offsets = np.array([
        (-2, -2), (-2, 0), (-2, 2),
        (0, -2),          (0, 2),
        (2, -2),  (2, 0), (2, 2)
    ], dtype=np.int32)

    G_neighbors_offsets = np.array([
        (-1, -1), (-1, 1),
        (0, -2),          (0, 2),
        (1, -1),  (1, 1),
        (-2, 0),          (2, 0)
    ], dtype=np.int32)

    # 分别矫正每个通道
    correct_channel(raw, corrected, mask_B_coords, BR_neighbors_offsets, threshold, H, W)
    correct_channel(raw, corrected, mask_G_coords, G_neighbors_offsets, threshold, H, W)
    correct_channel(raw, corrected, mask_R_coords, BR_neighbors_offsets, threshold, H, W)

    return corrected


def downsample_bayer_to_rgb(bggr):
    """
    将 BGGR Bayer 格式图像下采样为 RGB 图像
    """
    h, w = bggr.shape
    h2, w2 = h // 2, w // 2

    rgb = np.zeros((h2, w2, 3), dtype=bggr.dtype)
    # print(f"bggr dtype: {bggr.dtype}, min: {bggr.min()}, max: {bggr.max()}")



    rgb[..., 0] = bggr[0::2, 0::2]                  # B
    rgb[..., 1] = (bggr[0::2, 1::2] + bggr[1::2, 0::2]) / 2  # G

    rgb[..., 2] = bggr[1::2, 1::2]                  # R

    return rgb[:,:,::-1]


def feature_select(img_tmp, thresh_dark=0.02, thresh_saturation=0.98):
    img_tmp = img_tmp.reshape(-1, 3)
    mask = np.all((img_tmp > thresh_dark) & (img_tmp < thresh_saturation), axis=1)
    # print(f"np.max(img_tmp):{np.max(img_tmp)}")
    if not np.any(mask):
        print("not mask")
        feature_data = np.tile(img_tmp.mean(axis=0), (4, 1))
    else:
        img_filtered = img_tmp[mask]
        bright_v = img_filtered[np.argmax(img_filtered.sum(axis=1))]
        
        max_wp = img_filtered.max(axis=0)
        mean_v = img_filtered.mean(axis=0)
        dark_v = img_filtered[np.argmin(img_filtered.sum(axis=1))]

        feature_data = np.vstack([bright_v, max_wp, mean_v, dark_v]) # 4*3
        

    feature_data /= (feature_data.sum(axis=1, keepdims=True) + 1e-7)
    return feature_data[:, :2]


# def white_balance_gain(raw) :

#     rgb=downsample_bayer_to_rgb(raw)
#     # print(rgb[:5])
#     feature=feature_select(rgb).astype(np.float32) # rgb

#     feature=np.expand_dims(feature,axis=0)
#     # print(f"input feature: {feature}")

#     model=ort.InferenceSession("/home/tjy/project_cxy/open-source/image_denoising/PMRID/AWB_TEST/Color-Constancy-PCC/pretrain_models/fold_1.onnx")
#     input_name = model.get_inputs()[0].name
#     output_name = model.get_outputs()[0].name

    
#     pred = model.run([output_name], {input_name: feature})[0][0]

#     k_r,k_g=pred

#     k_b=(1-k_r-k_g)

#     gain=np.array([k_g/k_r,1,k_g/k_b])

#     # gain=np.array([k_r,k_g,k_b])

#     # gain=gain/np.linalg.norm(gain)
#     # print(f"illum:{gain}")
#     return gain


def estimate_cct_from_image(img):
    # 1. 取 RGB 平均值（或灰卡区域）
    r_mean = np.mean(img[:, :, 0])
    g_mean = np.mean(img[:, :, 1])
    b_mean = np.mean(img[:, :, 2])

    # 2. 线性 RGB → XYZ（ITU Rec.709 D65矩阵为例）
    R, G, B = r_mean, g_mean, b_mean
    X = 0.4124*R + 0.3576*G + 0.1805*B
    Y = 0.2126*R + 0.7152*G + 0.0722*B
    Z = 0.0193*R + 0.1192*G + 0.9505*B

    # 3. 计算 xy 色度
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)

    # 4. 使用 McCamy 拟合公式估计 CCT（粗略）
    n = (x - 0.3320) / (y - 0.1858)
    cct = 437 * n**3 + 3601 * n**2 + 6861 * n + 5517

    return cct



# 色温 -> 增益表
color_temp_table = {
    2500: (2.0, 1.0, 1.5),
    3500: (1.6, 1.0, 1.7),
    5000: (1.2, 1.0, 2.0),
    6500: (1.0, 1.0, 1.0),
    7500: (0.9, 1.0, 0.95),
    9000: (0.8, 1.0, 0.9),
}

def get_gain_from_cct(cct):
    keys = sorted(color_temp_table.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= cct <= keys[i + 1]:
            t1, t2 = keys[i], keys[i + 1]
            ratio = (cct - t1) / (t2 - t1)
            r1, g1, b1 = color_temp_table[t1]
            r2, g2, b2 = color_temp_table[t2]
            return (
                r1 + (r2 - r1) * ratio,
                g1 + (g2 - g1) * ratio,
                b1 + (b2 - b1) * ratio,
            )
    return color_temp_table[keys[-1]]  # 超出范围用最大值


def rgb_to_ycbcr(img):
    """将RGB图像转换为YCbCr"""
    matrix = np.array([[ 0.299,  0.587,  0.114],
                       [-0.169, -0.331,  0.500],
                       [ 0.500, -0.419, -0.081]])
    shift = np.array([0, 128, 128])
    ycbcr = img @ matrix.T + shift
    return ycbcr.astype(np.uint8)

def is_color_cast(ycbcr_img, threshold=5):
    """判断是否存在颜色偏移"""
    cb = ycbcr_img[:, :, 1].astype(np.int32)
    cr = ycbcr_img[:, :, 2].astype(np.int32)
    cb_mean = np.mean(cb)
    cr_mean = np.mean(cr)
    return abs(cb_mean - 128) > threshold or abs(cr_mean - 128) > threshold

def auto_level(channel, cut_limit=0.01):
    """通道直方图拉伸（自动色阶）"""
    hist, bins = np.histogram(channel.flatten(), 256, [0,256])
    total = np.sum(hist)

    # 去除两端cut_limit百分比的像素
    cut = total * cut_limit
    cumulative = np.cumsum(hist)
    low = np.searchsorted(cumulative, cut)
    high = np.searchsorted(cumulative, total - cut)

    # 拉伸
    channel = np.clip(channel, low, high)
    stretched = ((channel - low) / (high - low)) * 255
    return stretched.astype(np.uint8)



def white_balance(image, wb_gain=np.array([1,1,1]), method="pcc") :

    balanced_image = np.zeros_like(image, dtype=np.float32)

    B,G,R=image[:, :, 0],image[:, :, 1],image[:, :, 2]

    if method=="ycbcr":
        img = image.astype(np.uint8)
        ycbcr = rgb_to_ycbcr(img[:,:,::-1])
        if not is_color_cast(ycbcr):
            return img  # 没有偏色，不做处理

        balanced_image = np.zeros_like(img)
        for i in range(3):  # 对 R/G/B 各通道做拉伸
            balanced_image[:, :, i] = auto_level(img[:, :, i], cut_limit=0.01)

        return balanced_image


    if method=="cct":
        cct = estimate_cct_from_image(image[:,:,::-1])
        print(f"估算色温：{int(cct)}K")
        gains = get_gain_from_cct(cct)
        k_b, k_g, k_r=gains
        print(f"使用增益：R={gains[0]:.2f}, G={gains[1]:.2f}, B={gains[2]:.2f}")


    if method=="pcc":

        # # 目标白光色度（D65）
        # r_tgt, g_tgt = 0.3127, 0.3290
        # b_tgt = 1.0 - r_tgt - g_tgt
        # target = np.array([r_tgt, g_tgt, b_tgt], dtype=np.float32)

        # # 计算增益
        # gains = (target / wb_gain)
        k_b, k_g, k_r = wb_gain[::-1]
 

    if method=="grayworld":
        mean_b=np.mean(B).astype(np.float32)
        mean_g=np.mean(G).astype(np.float32)
        mean_r=np.mean(R).astype(np.float32)
        print(f"mean_b,mean_g,mean_r:{mean_b},{mean_g},{mean_r}")
        K=(mean_b + mean_g + mean_r) / 3
        # K=mean_g
        k_b = K / mean_b if mean_b != 0 else 1
        k_g = K / mean_g if mean_g != 0 else 1
        k_r = K / mean_r if mean_r != 0 else 1
        

    if method=="max":
        # 计算每个通道的最大值
        max_b = np.max(B)
        max_g = np.max(G)
        max_r = np.max(R)
        print(f"max_b,max_g,max_r:{max_b},{max_g},{max_r}")
        # max_value = max(max_b,max_g,max_r)
        max_value=255

        k_b = max_value / max_b if max_b != 0 else 1
        k_g = max_value / max_g if max_g != 0 else 1
        k_r = max_value / max_r if max_r != 0 else 1

    if method=="graypixel":
        mask=(np.abs(R-G)<10) & (np.abs(R-B)<10) & (np.abs(B-G)<10)

        mean_b=np.mean(B[mask]).astype(np.float32)
        mean_g=np.mean(G[mask]).astype(np.float32)
        mean_r=np.mean(R[mask]).astype(np.float32)

        print(f"mean_b,mean_g,mean_r:{mean_b},{mean_g},{mean_r}")

        ref = mean_g
        k_b = ref / mean_b
        k_g = 1
        k_r = ref / mean_r


        return balanced_image.astype(np.uint8)

    
    balanced_image[:, :, 0] = np.clip(B * k_b, 0, 255)
    balanced_image[:, :, 1] = np.clip(G * k_g, 0, 255)
    balanced_image[:, :, 2] = np.clip(R * k_r, 0, 255)

    print(f"k_b,k_g,k_r",k_b,k_g,k_r)

    return balanced_image.astype(np.uint8)


def bayer2rgb_bilinear(bayer: np.ndarray, pattern: str = 'BGGR') -> np.ndarray:
    """
    Bayer to RGB conversion using bilinear interpolation with convolution.
    Supports BGGR pattern only.
    """
    bayer = bayer.astype(np.float32)

    # Define interpolation kernels
    K = {
        'G': np.array([[0, 1, 0],
                       [1, 4, 1],
                       [0, 1, 0]], dtype=np.float32) / 4.0,
        'R_or_B': np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], dtype=np.float32) / 4.0
    }

    # Masks for each color channel (BGGR)
    mask_R = np.zeros_like(bayer)
    mask_G = np.zeros_like(bayer)
    mask_B = np.zeros_like(bayer)

    # BGGR pattern
    mask_B[0::2, 0::2] = 1  # Blue
    mask_G[0::2, 1::2] = 1  # Green on blue row
    mask_G[1::2, 0::2] = 1  # Green on red row
    mask_R[1::2, 1::2] = 1  # Red

    # print(f"np.max(bayer * mask_R):{np.max(bayer * mask_R)}")
    R = convolve(bayer * mask_R, K['R_or_B'])  # default mode reflect
    G = convolve(bayer * mask_G, K['G'])
    B = convolve(bayer * mask_B, K['R_or_B'])

    # Restore original values at known positions
    R[mask_R == 1] = bayer[mask_R == 1]
    G[mask_G == 1] = bayer[mask_G == 1]
    B[mask_B == 1] = bayer[mask_B == 1]

    bgr = np.stack([B, G, R], axis=-1)

    return bgr




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

    def __init__(self, model_path: Path, ksigma: KSigma, inp_scale=255.0, min=0, max=1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1  
        so.inter_op_num_threads = 1
        self.net = ort.InferenceSession(model_path,so)
        self.ksigma = ksigma
        self.inp_scale = inp_scale
        self.totol_time=0
        self.min=min
        self.max=max

    def pre_process(self, bayer_01: np.ndarray):
        rggb = RawUtils.bayer2rggb(bayer_01)
        rggb = rggb.clip(0, 1)
        
        H, W = rggb.shape[:2]

        # # 192*256
        ph = (32 - H % 32) % 32
        pw = (32 - W % 32) % 32

        # 分配 padding 到上下/左右
        ph_top = ph // 2
        ph_bottom = ph - ph_top
        pw_left = pw // 2
        pw_right = pw - pw_left

        # 执行 padding（只对 H 和 W 做 padding，不动通道数）
        rggb_padded = np.pad(rggb, 
                            [(ph_top, ph_bottom), (pw_left, pw_right), (0, 0)], 
                            mode='constant')

        print(f"rggb_padded.shape:{rggb_padded.shape}")

        inp_rggb = rggb.transpose(2, 0, 1)[np.newaxis]
        

        self.ph, self.pw = ph, pw
        # print(ph // 2, ph - ph // 2)
        # print(pw // 2, pw - pw // 2)
        return inp_rggb

    def run(self, bayer_01: np.ndarray, iso: float):
        inp_rggb_01 = self.pre_process(bayer_01)

        inp_rggb = self.ksigma(inp_rggb_01, iso) * self.inp_scale

        

        inp = np.ascontiguousarray(inp_rggb).astype(np.float32)
        print(f"inp.shape:{inp.shape}")
        print(f"np.min(inp),{np.min(inp)}")
        print(f"np.max(inp),{np.max(inp)}")
        print(f"np.mean(inp),{np.mean(inp)}")
        print(f"np.std(inp),{np.std(inp)}")
        # print(f"inp[:1]:{inp[:1]}")
        inp = inp
        input_name = self.net.get_inputs()[0].name
        output_name = self.net.get_outputs()[0].name

        it=1
        for i in range(it):
            start_time=time.time()
            pred = (self.net.run([output_name], {input_name: inp})[0][0])/ self.inp_scale
            self.totol_time+=time.time()-start_time
        
        print(f"infer time:{self.totol_time/it}")

        pred=pred*(self.max-self.min)+self.min


        # import ipdb; ipdb.set_trace()
        pred = np.transpose(pred,[1, 2, 0])
        print(f"pred.shape:{pred.shape}")
        pred = self.ksigma(pred, iso, inverse=True)

        # ph, pw = self.ph, self.pw

        # pred = pred[ph:-ph, pw:-pw]

        ## 648*486
        ph, pw = self.ph, self.pw
        
        h0 = ph // 2
        h1 = pred.shape[0] - (ph - ph // 2)
        w0 = pw // 2
        w1 = pred.shape[1] - (pw - pw // 2)
        pred = pred[h0:h1, w0:w1,:]
        print(pred.shape)
        ## 648*486

        return RawUtils.rggb2bayer(pred)


def run_test(model_path):

    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=300,
        V=959
    )
    denoiser = Denoiser(model_path, ksigma) 
    # denoiser = Denoiser(model_path, ksigma, 1, 0, 0.0623)
    W, H = 2048, 1536  
    iso = 6400

    # file_name="black2light_20250902_1"
    frame=130

    light=60
    # 读取单帧数据（示例）
    with open(f"imtest/{frame}.raw", 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    # data=data[89*W*H:90*W*H]

    # 重塑为BGGR排列的二维矩阵
    input_bayer = data.reshape(H, W)


    # input_bayer = input_bayer[:H, :W]
    print(f"input_bayer.shape:{input_bayer.shape}")
    print(f"np.mean(input_bayer):{np.mean(input_bayer)}")
    print(f"np.max(input_bayer):{np.max(input_bayer)}")
    # 提取绿色分量
    g1 = input_bayer[0::2, 1::2]  # 偶数行, 奇数列
    g2 = input_bayer[1::2, 0::2]  # 奇数行, 偶数列
    green = np.concatenate([g1.flatten(), g2.flatten()])

    print(f"绿色均值: {green.mean()}")

    visual_rate=light/green.mean()
    # visual_rate=1

    raw_demosaicing=cv2.demosaicing(np.clip(input_bayer*visual_rate, 0, 255).astype(np.uint8), cv2.COLOR_BayerRG2BGR)


    result_dir='im_result'

    cv2.imwrite(f'{result_dir}/{frame}_raw.png', raw_demosaicing,[cv2.IMWRITE_PNG_COMPRESSION, 0])



    input_bayer = RawUtils.bggr2rggb(input_bayer)
    # print(f"input_bayer shape:{input_bayer.shape}")
    print(f"input_bayer.shape:{input_bayer.shape}")
    

    pred_bayer = denoiser.run(input_bayer/255, iso=iso)
    pred_bayer=RawUtils.rggb2bggr(pred_bayer)
    pred_bayer=np.clip(pred_bayer,0,1)

    # wb_gain=white_balance_gain(pred_bayer)

    print(pred_bayer[0,:10])
    # pred_bayer=pred_bayer[:243,:]   # 还原为243*324
    print(f"pred_bayer.shape:{pred_bayer.shape}")

    pred_bayer = pred_bayer * 255.0  # （0,1）-> （0,255）

    # demosaic
    bgr_img = cv2.demosaicing(pred_bayer.astype(np.uint8), cv2.COLOR_BayerRG2BGR).astype(np.float32)

    # 然后调亮度
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(pred_bayer)
    print(f"brightness pred:{brightness}")
    cv_rate=light/brightness

    bgr_img = np.clip(bgr_img * cv_rate, 0, 255).astype(np.uint8)

    im_path=f'{result_dir}/{frame}_denoised_{iso}.png'
    cv2.imwrite(im_path, bgr_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(f"img saved in: {im_path}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)

    args = parser.parse_args()

    run_test(args.model)



if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
