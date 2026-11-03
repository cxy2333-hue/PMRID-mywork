import tensorflow as tf
import os
import sys
import re
import numpy as np

# -------------------------------
# 量化tflite模型为int16x8 或 int8
# 
# -------------------------------

if len(sys.argv) < 2:
    print("用法: python quantize_tf.py <saved_model_dir>")
    print("示例: python quantize_tf.py ./models/pmrid_tf/model_64_64")
    sys.exit(1)

saved_model_dir = sys.argv[1].rstrip("/")

# -------------------------------
# Step 1: 从目录名解析输入尺寸、生成输出路径
# 目录名格式 model_32_32
# -------------------------------
basename = os.path.basename(saved_model_dir)
m = re.search(r"(\d+)_(\d+)", basename)
if not m:
    raise ValueError(f"无法从目录名 {basename} 解析输入尺寸，要求格式如 model_32_32")

size, _ = int(m.group(1)), int(m.group(2))
print(f"检测到输入尺寸: H={size}, W={size}\n")
print(f"basename: {basename}\n")
# 自动拼接输出路径：saved_model_dir/basename_int16x8.tflite
output_tflite_path = os.path.join(saved_model_dir, f"{basename}_int16x8.tflite")

calib_data = []
data_dir = f"calib_data_{size}x{size}"
for filename in sorted(os.listdir(data_dir)):
    if filename.endswith(".npy"):
        data = np.load(os.path.join(data_dir, filename))
        # data = data.transpose(0, 2, 3, 1)/256
        data = data.transpose(0, 2, 3, 1)
        calib_data.append(data.astype(np.float32))

def representative_dataset_gen():
    for i, data in enumerate(calib_data):
        if i % 20 == 0:
            print(f"Yielding data {i}, shape: {data.shape}")
        yield [data]

# 创建转换器
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
#   EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8  TFLITE_BUILTINS_INT8
# converter.allow_custom_ops = True
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

# 转换模型
tflite_quant_model = converter.convert()

# 保存
with open(output_tflite_path, "wb") as f:
    f.write(tflite_quant_model)

print(f"转换成功, {output_tflite_path} 已保存。")