#!/bin/bash
set -e

# 初始化 conda
source $HOME/anaconda3/etc/profile.d/conda.sh

#########################
## 用于导出将pmrid转为tflite模型，可以指定导出的尺寸，B*H*W  B=4,H通常等于W
#########################
export PATH="$HOME/.local/bin:$PATH"
## usage: ./ckp2tflite.sh 4 32 32 net_test22
B=$1
H=$2
W=$3
arch=$4

# 固定模型路径
MODEL_NAME="model"
OUTPUT_DIR="models/my_onnx"
TF_DIR="models/my_tf"
ONNX_PATH="${OUTPUT_DIR}/${MODEL_NAME}_$((2*H))_$((2*W)).onnx"
mkdir -p "$OUTPUT_DIR"

conda activate py38
python export_onnx_my.py --arch $arch  --ckp_dir "./checkpoints/$arch" --onnx_path $ONNX_PATH --height $H  --width $W

# onnx转为tensorflow
conda activate onnxtotflite
onnx2tf -i "$ONNX_PATH" -o $TF_DIR/${MODEL_NAME}_$((2*H))_$((2*W)) 

# # # 准备对应尺寸的量化数据
conda activate py38
python prepare_quantize_data_sid.py --block_h ${H} --block_w ${W}

# tensorflow INT16量化
conda activate tf2.12
python quantize_tf.py $TF_DIR/${MODEL_NAME}_$((2*H))_$((2*W))
