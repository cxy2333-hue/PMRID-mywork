#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

def align4(x):  # 对齐到4的倍数，硬件友好
    return max(4, int((x + 3) // 4 * 4))

def Conv2D(in_channels, out_channels, kernel_size, stride, padding,
           is_seperable=False, has_relu=False):
    modules = OrderedDict()
    if is_seperable:
        modules['depthwise'] = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        modules['pointwise'] = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, bias=True
        )
    else:
        modules['conv'] = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=True
        )
    if has_relu:
        modules['relu'] = nn.ReLU()
    return nn.Sequential(modules)

class LearnablePool(nn.Module):
    """可学习的下采样卷积"""
    def __init__(self, in_ch):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_ch, 1, 2, 2) / 4)  # 每个通道独立权重

    def forward(self, x):
        # 分组卷积，每个通道独立卷积
        return F.conv2d(x, self.weight, stride=2, padding=0, groups=x.shape[1])


class ChannelAttention(nn.Module):
    """轻量通道注意力"""
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class LightweightFusion(nn.Module):
    """带通道注意力的特征融合模块"""
    def __init__(self, in_ch, mid=16, reduction=8):
        super().__init__()
        # 像素级 gating
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 1, 1, 0)
        )
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, spatial_feat, freq_feat):
        x = torch.cat([spatial_feat, freq_feat], dim=1)
        weight = torch.sigmoid(self.gate(x))       # 像素级融合
        fused = spatial_feat * (1 - weight) + freq_feat * weight
        ca = self.channel_att(fused)               # 通道注意力增强
        fused = fused * ca
        return fused

# -------------------------------------
# Encoder Block
# -------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        # 5x5 -> 3x3，padding 相应改为 1
        self.conv1 = Conv2D(in_channels, mid_channels, 3, stride, 1, is_seperable=True, has_relu=True)
        self.conv2 = Conv2D(mid_channels, out_channels, 3, 1, 1, is_seperable=True, has_relu=False)
        self.proj = (
            nn.Identity() if stride==1 and in_channels==out_channels
            else Conv2D(in_channels, out_channels, 3, stride, 1, is_seperable=True, has_relu=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        proj = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + proj
        return self.relu(x)

def EncoderStage(in_channels, out_channels, num_blocks):
    blocks = [EncoderBlock(in_channels, out_channels//4, out_channels, stride=2)]
    for _ in range(num_blocks-1):
        blocks.append(EncoderBlock(out_channels, out_channels//4, out_channels, stride=1))
    return nn.Sequential(*blocks)



# -------------------------------------
# Decoder Block
# -------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv0 = Conv2D(in_channels, out_channels, kernel_size, 1, padding, is_seperable=True, has_relu=True)
        self.conv1 = Conv2D(out_channels, out_channels, kernel_size, 1, padding, is_seperable=True, has_relu=False)

    def forward(self, x):
        inp = x
        x = self.conv0(x)
        x = self.conv1(x)
        return x + inp

class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_in_channels, out_channels):
        super().__init__()
        self.decode_conv = DecoderBlock(in_channels, in_channels)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.proj_conv = Conv2D(skip_in_channels, out_channels, 3, 1, 1, is_seperable=True, has_relu=True)
    def forward(self, inputs):
        inp, skip = inputs
        x = self.decode_conv(inp)
        x = self.upsample(x)
        y = self.proj_conv(skip)

        # --- 自动对齐 ---
        if x.shape[-2:] != y.shape[-2:]:
            min_h = min(x.shape[-2], y.shape[-2])
            min_w = min(x.shape[-1], y.shape[-1])
            x = x[..., :min_h, :min_w]
            y = y[..., :min_h, :min_w]

        return x + y




class DWT_Haar(nn.Module):
    """Haar 小波变换"""
    def __init__(self):
        super().__init__()
        s = 1.0 / math.sqrt(2.0)
        ll = torch.tensor([[ s,  s],
                           [ s,  s]])
        lh = torch.tensor([[-s, -s],
                           [ s,  s]])
        hl = torch.tensor([[-s,  s],
                           [-s,  s]])
        hh = torch.tensor([[ s, -s],
                           [-s,  s]])
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,2,2]
        self.register_buffer("filters", filters)

    def forward(self, x):
        B, C, H, W = x.shape
        filt = self.filters.to(x.device).type_as(x)
        y = F.conv2d(x, filt.repeat(C,1,1,1), stride=2, groups=C)
        # 输出: [B, 4*C, H/2, W/2]
        return y

class FreTransferLite(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.conv_pre = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pool = LearnablePool(in_ch)
        self.conv_post = nn.Conv2d(in_ch, in_ch*4, 1, groups=1)
        self.att = ChannelAttention(in_ch*4)

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.pool(x)  # 可学习下采样
        x = self.conv_post(x)  # 输出 LL, HL, LH, HH
        x = self.att(x)         # 通道注意力增强
        return x

class Frequency_Net(nn.Module):
    def __init__(self, channel_scale=1.0):
        super().__init__()

        self.c0 = align4(16 * channel_scale)
        self.c1 = align4(32 * channel_scale)
        self.c2 = align4(64 * channel_scale)
        self.cm = align4(32 * channel_scale)

        self.conv0 = Conv2D(16, self.c0, 3, 1, 1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(self.c0, self.c1, 2)
        self.enc2 = EncoderStage(self.c1, self.c2, 4)
        self.middle = Conv2D(self.c2, self.cm, 3, 1, 1, is_seperable=True, has_relu=True)

        self.se_mid = ChannelAttention(self.cm)

    def forward(self, inp):

        fre_conv0 = self.conv0(inp)
        fre_enc1 = self.enc1(fre_conv0)
        fre_enc2 = self.enc2(fre_enc1)
        freq_mid = self.middle(fre_enc2)
        freq_mid = self.se_mid(freq_mid)

        return fre_conv0, fre_enc1, fre_enc2, freq_mid



class Spatial_Net(nn.Module):
    def __init__(self, channel_scale=1.0, freq_chs=(32, 64, 32)):

        super().__init__()


        s_ch = lambda x: align4(x * channel_scale)

        self.conv0 = Conv2D(4, s_ch(16), 3, 1, 1)
        self.enc1 = EncoderStage(s_ch(16), s_ch(64), 2)
        self.enc2 = EncoderStage(s_ch(64), s_ch(128), 2)
        self.enc3 = EncoderStage(s_ch(128), s_ch(256), 4)

        self.middle = Conv2D(s_ch(256), s_ch(64), 3, 1, 1, is_seperable=True)

        fmid_ch, f2_ch, f1_ch = freq_chs

        self.dec1 = DecoderStage(s_ch(64) + fmid_ch, s_ch(128), s_ch(64))
        self.dec2 = DecoderStage(s_ch(64) + f2_ch, s_ch(64), s_ch(32))
        self.dec3 = DecoderStage(s_ch(32) + f1_ch, s_ch(16), s_ch(16))

        self.out = nn.Sequential(
            Conv2D(s_ch(16), s_ch(16), 3, 1, 1, is_seperable=True),
            nn.Conv2d(s_ch(16), 4, 3, 1, 1)
        )

    def forward(self, inp, freq_feats=None):
        """
        freq_feats: tuple(fmid, f2, f1)
        """
        conv0 = self.conv0(inp)
        enc1 = self.enc1(conv0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        mid = self.middle(enc3)

        if freq_feats is not None:
            fmid, f2, f1 = freq_feats


            fmid = F.interpolate(fmid, size=mid.shape[-2:], mode='bilinear', align_corners=False)
            up1_in = torch.cat([mid, fmid], dim=1)
            up1 = self.dec1((up1_in, enc2))


            f2 = F.interpolate(f2, size=up1.shape[-2:], mode='bilinear', align_corners=False)
            up2_in = torch.cat([up1, f2], dim=1)
            up2 = self.dec2((up2_in, enc1))


            f1 = F.interpolate(f1, size=up2.shape[-2:], mode='bilinear', align_corners=False)
            up3_in = torch.cat([up2, f1], dim=1)
            x = self.dec3((up3_in, conv0))

        else:
     
            up2 = self.dec1((mid, enc2))
            up1 = self.dec2((up2, enc1))
            x = self.dec3((up1, conv0))

        out = self.out(x)
        return out




# -------------------------------------
# 主网络
# -------------------------------------
class Network(nn.Module):
    def __init__(self, Spatial_channel_scale=0.5, Frequency_channel_scale=0.5):
        super().__init__()
        self.Frequency_Net = Frequency_Net(Frequency_channel_scale)

        f_chs = (self.Frequency_Net.c1, 
                 self.Frequency_Net.c2,
                 self.Frequency_Net.cm)

        self.Spatial_Net = Spatial_Net(Spatial_channel_scale, freq_chs=f_chs)
        self.FreTransferLite = FreTransferLite(4)
        # self.dwt = DWT_Haar()
        self.dwt_ca = ChannelAttention(4 * 4)
        self.dwt_refine = nn.Sequential(
            nn.Conv2d(4*4, 4*4, 3, padding=1, groups=4),
            nn.ReLU(inplace=True)
        )

    def forward(self, inp):
        fre_inp = self.FreTransferLite(inp)  

        fre_inp = self.dwt_ca(fre_inp)
        fre_inp = self.dwt_refine(fre_inp)
        
        #  频域子网
        f0, f1, f2, fmid = self.Frequency_Net(fre_inp)

        # 空域子网
        out = self.Spatial_Net(inp, freq_feats=(fmid, f2, f1))

        # 残差输出

        final_pred = inp + out

        return final_pred




# -------------------------------------
# 测试网络
# -------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    net = Network(0.5,0.5)
    print(f"Net params: {count_params(net)}")
    # for s in [16,32,64]:
    #     img = torch.randn(1, 4, s, s)
    #     out = net(img)
    #     print(f"Input {s}x{s} -> Output {out.shape}")


    # img = torch.randn(1, 4, 768, 1024)
    # onnx_path = "model.onnx"
    # torch.onnx.export(
    #     net,
    #     img,                      # 示例输入张量
    #     onnx_path,
    #     opset_version=11,
    #     do_constant_folding=True, # 常量折叠优化
    #     input_names=["input"],
    #     output_names=["output"],
    #     dynamic_axes={
    #         # 如果需要支持动态batch，开启下面；固定batch=1就注释
    #         # "input": {0: "batch_size"},
    #         # "output": {0: "batch_size"}
    #     }
    # )
    # print(f"ONNX exported -> {onnx_path}")

    # # 可选：校验onnx是否合法
    # import onnx
    # model_onnx = onnx.load(onnx_path)
    # onnx.checker.check_model(model_onnx)
    # print("onnx check passed!")
