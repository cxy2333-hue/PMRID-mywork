#!/bin/bash
# 训练 net_test10 (scale1=scale2=1)
# 续训/迁移权重时, 取消下一行注释并改成对应 ckp 路径:
#   注意: --resume 默认接续 epoch; 若只想迁移权重从头训, 需另加 --reset-epoch
# RESUME="--resume ./checkpoints/checkpoints_student_20251107_2/epoch_350.ckp"
# net10:299之前及299是712，之后是255， net11:329之前及329是712 329之后是255

python train.py \
--arch net_test11   \
--data-aug-config ./config.json \
--data-dir ./dataset/sid_clean/ \
--batch-size 1 \
--ckp-dir ./checkpoints/net_test11 \
--learning-rate 1e-3 \
--num-epoch 350 \
--device "cpu" \
--repeats 2 \
--val-split 0.2 \
--scale1 1 \
--scale2 1 \
--alpha 0 \
--resume ./checkpoints/net_test10/epoch_328.ckp     
$RESUME
