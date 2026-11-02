#!/usr/bin/env python3
import os
import argparse
import importlib
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import numpy as np
from models.net_torch import Network, get_loss_l1
from dataset import CleanRawImages, DataAug, DataAugOptions

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def validate(student, val_loader, train_aug, device):
    student.eval()
    val_loss_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    with torch.no_grad():
        for imgs, g_means in val_loader:
            imgs = imgs.to(device)
            g_means = g_means.to(device)
            data = train_aug.transform(imgs, g_means)
            if data is None:
                continue
            imgs_aug, gt_aug, norm_k, norm_b = data
            preds = student(imgs_aug)
            loss = get_loss_l1(preds, gt_aug, norm_k)
            
            val_loss_sum += loss.item() * imgs.size(0)
            
            # k-sigma 域内相对指标: 逐样本用 gt 动态范围作 data_range (输出不是 [0,255])
            preds_cpu = preds.detach().cpu().numpy()
            gt_cpu = gt_aug.detach().cpu().numpy()
            for i in range(preds_cpu.shape[0]):
                pred_img = preds_cpu[i].transpose(1,2,0)
                gt_img   = gt_cpu[i].transpose(1,2,0)
                rng = max(float(gt_img.max() - gt_img.min()), 1e-6)

                psnr_sum += compare_psnr(gt_img, pred_img, data_range=rng)
                ssim_sum += compare_ssim(gt_img, pred_img, multichannel=True, data_range=rng)
            
            count += imgs.size(0)

    return {
        'loss': val_loss_sum / count,
        'psnr': psnr_sum / count,
        'ssim': ssim_sum / count
    }



def train_step(student, optimizer, imgs, gt, norm_k, alpha=0.7):

    optimizer.zero_grad()

    student_out = student(imgs)

    label_loss = get_loss_l1(student_out, gt, norm_k)

    loss = label_loss
    loss.backward()
    optimizer.step()
    return loss.item()


# def validate(student, val_loader, train_aug, device):
#     student.eval()
#     val_loss_sum = 0.0
#     count = 0
#     with torch.no_grad():
#         for imgs, g_means in val_loader:
#             imgs = imgs.to(device)
#             g_means = g_means.to(device)
#             data = train_aug.transform(imgs, g_means)
#             if data is None:
#                 continue
#             imgs_aug, gt_aug, norm_k, norm_b = data
#             preds = student(imgs_aug)
#             loss = get_loss_l1(preds, gt_aug, norm_k)
#             val_loss_sum += loss.item() * imgs.size(0)
#             count += imgs.size(0)
#     return val_loss_sum / count





def update_val_log(val_log_path, epoch, train_loss, val_loss):
    lines = []
    if os.path.exists(val_log_path):
        with open(val_log_path, "r") as f:
            lines = f.readlines()

    epoch_str = str(epoch)
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(epoch_str + ","):
            # 覆盖旧的同 epoch 记录
            lines[i] = f"{epoch},{train_loss:.6f},{val_loss:.6f}\n"
            updated = True
            break

    # 如果是新的 epoch，就追加完整行
    if not updated:
        lines.append(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")

    # 写回文件
    with open(val_log_path, "w") as f:
        f.writelines(lines)



def load_best_val_loss(filepath):
    if not os.path.exists(filepath):
        return float('inf')
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        if not line:
            return float('inf')
        parts = line.split(',')
        if len(parts) < 2:
            return float('inf')
        try:
            return float(parts[1])
        except ValueError:
            return float('inf')

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def worker_init_fn(worker_id):
    # 每个 DataLoader worker 独立播种 numpy，否则各 worker 的随机裁剪/翻转完全相同
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-aug-config', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--ckp-dir', default=Path('./checkpoints'), type=Path)
    parser.add_argument('--learning-rate', dest='lr', default=1e-3, type=float)
    parser.add_argument('--num-epoch', default=400, type=int)
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--val-split', default=0.1, type=float)
    parser.add_argument('--repeats', default=1, type=int)
    parser.add_argument('--alpha', default=0.7, type=float, help="蒸馏损失权重")
    parser.add_argument('--scale1', default=1, type=float, help="卷积层的通道数缩放比例")
    parser.add_argument('--scale2', default=1, type=float, help="卷积层的通道数缩放比例")
    parser.add_argument('--arch', default='net_test10', type=str,
                        help="models/ 下的网络模块名 (该模块需导出 Network(Spatial_scale, Frequency_scale))")
    parser.add_argument('--resume', default=None, type=str, help="路径: 从某个检查点继续训练")
    parser.add_argument('--num-workers', default=4, type=int, help="DataLoader 工作进程数")
    parser.add_argument('--save-every', default=20, type=int,
                        help="每 N 个 epoch 存一份 epoch_N.ckp（0=只存 latest/best）")

    args = parser.parse_args()

    if not args.ckp_dir.exists():
        args.ckp_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(os.cpu_count() or 8)

    aug_opts = DataAugOptions.parse_file(args.data_aug_config)
    # 先按原图数切分训练/验证, 再对训练集 repeats 扩充 (否则同一张图泄漏进验证集)
    full_dataset = CleanRawImages(data_dir=args.data_dir, output_shape=aug_opts.output_shape)
    train_aug = DataAug(aug_opts)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_sub, val_ds = random_split(full_dataset, [train_size, val_size])
    train_ds = Subset(full_dataset, list(train_sub.indices) * args.repeats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"训练集大小: {len(train_ds)}")
    print(f"验证集大小: {len(val_ds)}")

    # 网络结构由 --arch 从 models/ 下动态指定
    net_mod = importlib.import_module(f"models.{args.arch}")
    Small_net = net_mod.Network



    
    student = Small_net(args.scale1,args.scale2).to(device)
    print(f"Net params: {count_params(student)}")
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.num_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # 如果指定了 --resume，就加载之前的权重
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"从 {args.resume} 加载权重(仅模型权重，不恢复优化器/调度器)...")
        checkpoint = torch.load(args.resume, map_location=device)
        # 判断文件内容结构
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整 checkpoint 文件，只提取模型权重
            old_state_dict = checkpoint['model_state_dict']
            new_state_dict = student.state_dict()
            # 处理DDP module.前缀，如果ckpt是多卡保存
            fixed_old = {}
            for k, v in old_state_dict.items():
                if k.startswith("module."):
                    fixed_old[k[7:]] = v
                else:
                    fixed_old[k] = v

            final_dict = {}
            for k, old_tensor in fixed_old.items():
                if k not in new_state_dict:
                    print(f"skip key not in new model: {k}")
                    continue
                new_tensor = new_state_dict[k]
                old_shape = old_tensor.shape
                new_shape = new_tensor.shape
                if old_shape == new_shape:
                    final_dict[k] = old_tensor
                    continue
                print(f"resize {k}: ckpt {old_shape} → model {new_shape}")
                nt = new_tensor.clone()
                if len(old_shape) == 4:
                    # conv weight: (out, in, kh, kw)
                    old_out, old_in = old_shape[0], old_shape[1]
                    new_out, new_in = new_shape[0], new_shape[1]
                    slice_out = slice(0, new_out)
                    slice_in = slice(0, new_in)
                    nt[slice_out, slice_in, :, :] = old_tensor[slice_out, slice_in, :, :]
                elif len(old_shape) == 1:
                    # bias 一维 (out,)
                    slice_out = slice(0, new_shape[0])
                    nt[slice_out] = old_tensor[slice_out]
                else:
                    print(f"skip unknown shape {k} {old_shape}")
                    continue
                final_dict[k] = nt

            msg = student.load_state_dict(final_dict, strict=False)
            print("==== load result ====")
            print("missing_keys count:", len(msg.missing_keys))
            print("unexpected_keys count:", len(msg.unexpected_keys))
            print("模型权重已加载（允许部分不匹配）。")
            if msg.missing_keys:
                print(f"缺失权重（新模块未加载）: {msg.missing_keys}")
            if msg.unexpected_keys:
                print(f"多余权重（旧模型中存在但当前模型没有）: {msg.unexpected_keys}")

            # 只读取epoch，**不加载优化器、scheduler**
            start_epoch = checkpoint.get('epoch', -1) + 1
            if start_epoch <= 0:
                print("checkpoint未找到epoch信息，start_epoch回退到0")
                start_epoch = 0
            else:
                print(f"仅加载权重，epoch接续，从 {start_epoch} 开始训练；优化器、scheduler全部重新初始化")

        else:
            # 纯 .pth 文件，没有epoch元信息，只能从0开始
            msg = student.load_state_dict(checkpoint, strict=False)
            print("已加载纯模型权重文件 (.pth)，允许部分不匹配。")
            if msg.missing_keys:
                print(f" 缺失权重（新模块未加载）: {msg.missing_keys}")
            if msg.unexpected_keys:
                print(f" 多余权重（旧模型中存在但当前模型没有）: {msg.unexpected_keys}")
            start_epoch = 0
            print("纯pth权重无epoch元数据，start_epoch=0")




    val_log_path = args.ckp_dir / "val_loss.txt"          # 验证集loss日志文件
    best_val_log_path = args.ckp_dir / "best_val_loss.txt" # 最优模型loss记录

    best_val_loss = load_best_val_loss(best_val_log_path)
    # 主训练循环
    for epoch in range(start_epoch, args.num_epoch):
        student.train()
        epoch_loss_sum, batch_count = 0, 0

        for imgs, g_means in train_loader:
            imgs, g_means = imgs.to(device), g_means.to(device)
            data = train_aug.transform(imgs, g_means)
            if data is None:
                continue
            imgs_aug, gt_aug, norm_k, norm_b = data

            loss = train_step(student, optimizer, imgs_aug, gt_aug, norm_k, alpha=args.alpha)
            epoch_loss_sum += loss
            batch_count += 1
            scheduler.step()

        train_loss = epoch_loss_sum / batch_count
        # current_lr = optimizer.param_groups[0]['lr']  #  当前学习率
        # val_loss = validate(student, val_loader, train_aug, device)
        
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.8f}")
        
        # ---- 验证 ----
        val_metrics = validate(student, val_loader, train_aug, device)

        val_loss = val_metrics['loss']
        val_psnr = val_metrics['psnr']
        val_ssim = val_metrics['ssim']

        current_lr = optimizer.param_groups[0]['lr']

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"PSNR: {val_psnr:.4f} | "
            f"SSIM: {val_ssim:.4f} | "
            f"LR: {current_lr:.8f}")
        

        # ---- 保存 checkpoint: latest 每轮覆盖, best 按验证 loss, 每 save_every 存 epoch_N ----
        ckpt = {
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
        }
        torch.save(ckpt, args.ckp_dir / "latest.ckp")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, args.ckp_dir / "best.ckp")
            # load_best_val_loss 读取格式: epoch,val_loss,...
            with open(best_val_log_path, "w") as f:
                f.write(f"{epoch},{val_loss:.6f},{val_psnr:.6f},{val_ssim:.6f}\n")
        # 每个 epoch 存快照: 保证 val_loss.txt 中任意 epoch 的权重可回溯
        # (export_onnx_my.py 的 top-N 平均按 epoch_{epoch}.ckp 加载, 只在整点存会对不上最优 epoch)
        torch.save(ckpt, args.ckp_dir / f"epoch_{epoch}.ckp")

        update_val_log(val_log_path, epoch, train_loss, val_loss)


if __name__ == '__main__':
    main()
