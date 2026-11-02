import argparse
import importlib
import torch
from pathlib import Path
import numpy as np
# 网络结构由 --arch 从外部指定 (models/ 下模块, 需导出 Network(Spatial_scale, Frequency_scale))


# -------------------------------
# 用训练好的权重，导出自定义尺寸的onnx模型
# -------------------------------

def load_best_checkpoints(ckp_dir: Path, top_n: int, device="cpu"):
    """
    从 ckp_dir 读取 val_loss.txt，选出最小的 N 个 val_loss，
    计算对应模型权重的平均值并返回。
    """
    val_log_path = ckp_dir / "val_loss.txt"
    if not val_log_path.exists():
        raise FileNotFoundError(f"{val_log_path} 不存在")

    # ======= 读取 val_loss.txt =======
    epochs, val_losses = [], []
    with open(val_log_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                epochs.append(int(parts[0]))
                val_losses.append(float(parts[2]))

    epochs = np.array(epochs)
    val_losses = np.array(val_losses)

    # ======= top-n 最小 loss =======
    best_idx = np.argsort(val_losses)[:top_n]
    best_epochs = epochs[best_idx]
    print(f"选择 val_loss 最小的 {top_n} 个模型: {best_epochs.tolist()}")

    # ======= 加载并平均 N 个模型 =======
    avg_state_dict = None

    for epoch in best_epochs:
        ckp_path = ckp_dir / f"epoch_{epoch}.ckp"
        if not ckp_path.exists():
            raise FileNotFoundError(f"{ckp_path} 不存在")

        checkpoint = torch.load(ckp_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        # 第一次: 克隆所有张量（float / int 都保留）
        if avg_state_dict is None:
            avg_state_dict = {k: v.clone().to(device) for k, v in state_dict.items()}
        else:
            for k in avg_state_dict.keys():
                # 只累加浮点参数
                if avg_state_dict[k].dtype.is_floating_point:
                    avg_state_dict[k] += state_dict[k].to(device).float()
                # 非浮点（例如 int64）不做累加，保持第一个模型的值

    # ======= N 个模型求平均 =======
    N = float(len(best_epochs))
    for k in avg_state_dict.keys():
        if avg_state_dict[k].dtype.is_floating_point:
            avg_state_dict[k] /= N
            # 保持 dtype 一致
            avg_state_dict[k] = avg_state_dict[k].type_as(state_dict[k])

    return avg_state_dict



def export_onnx(ckp_dir, onnx_path, top_n=5, device="cpu", height=768, width=1024, scale1=1.0,scale2=1.0, arch="net_test10"):
    ckp_path = Path(ckp_dir)
    Network = importlib.import_module(f"models.{arch}").Network
    net = Network(scale1,scale2).to(device)
    # net = Network().to(device)

    # 判断是否传入的是单个 .ckp 文件
    if ckp_path.is_file() and ckp_path.suffix == ".ckp":
        print(f"检测到单个 checkpoint 文件: {ckp_path}")
        checkpoint = torch.load(ckp_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        net.load_state_dict(state_dict)
        print(f"已加载模型 {ckp_path.name}")
    else:
        # 原来的多模型平均逻辑
        avg_state_dict = load_best_checkpoints(ckp_path, top_n, device=device)
        net.load_state_dict(avg_state_dict)
        print(f"已加载并平均 {top_n} 个模型")

    # 导出 ONNX
    net.eval()
    dummy_input = torch.randn(1, 4, height, width, device=device)
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=False
    )
    print(f" ONNX 模型已保存到: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckp_dir", type=Path, required=True, help="模型目录或单个 .ckp 文件")
    parser.add_argument("--onnx_path", type=Path, required=True, help="导出的 onnx 模型路径")
    parser.add_argument("--device", default="cpu", type=str, help="导出时使用的设备")
    parser.add_argument("--topn", type=int, default=3, help="选择 val_loss 最小的 n 个模型平均")
    parser.add_argument("--height", type=int, default=768, help="dummy_input 高度")
    parser.add_argument("--width", type=int, default=1024, help="dummy_input 宽度")
    parser.add_argument("--scale1", type=float, default=1.0, help="通道缩放尺度")
    parser.add_argument("--scale2", type=float, default=1.0, help="通道缩放尺度")
    parser.add_argument("--arch", default="net_test10", type=str, help="models/ 下的网络模块名")
    args = parser.parse_args()

    export_onnx(
        args.ckp_dir,
        args.onnx_path,
        top_n=args.topn,
        device=args.device,
        height=args.height,
        width=args.width,
        scale1=args.scale1,
        scale2=args.scale2,
        arch=args.arch,
    )
