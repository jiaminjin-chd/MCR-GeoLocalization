#!/bin/bash

# ======================================================
# 自动运行不同掩码比例的 MCR 实验，并汇总结果到 CSV
# 用法：bash run_mcr_experiments.sh
# ======================================================

# 设置实验参数
DATA_DIR="/root/autodl-tmp/DenseUAV/train"
BATCHSIZE=32
EPOCHS=120
BLOCK=4
H=384
W=384
LR=0.01
POOL="avg"
STRIDE=1
CENTER_RATIO=0.75
MASK_MODE="single"

# 掩码比例列表（从 0 到 0.95，步长 0.05）
# mask_ratios=(0 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
mask_ratios=(0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)

# 结果 CSV 文件
CSV_FILE="mcr_results.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "Dataset,Direction,MaskRatio,R@1,R@5,R@10,AP" > $CSV_FILE
fi

# 记录开始时间
echo "开始时间: $(date)"

# 遍历每个掩码比例
for ratio in "${mask_ratios[@]}"; do
    # 将小数转为字符串用于文件夹名（例如 0.05 -> 0_05）
    ratio_str=$(echo $ratio | sed 's/\./_/g')
    EXP_NAME="denseuav_lpn_mcr_${ratio_str}"
    MODEL_DIR="./model/${EXP_NAME}"
    BEST_MODEL="${MODEL_DIR}/best_model.pth"

    echo "=================================================="
    echo "处理掩码比例: $ratio (实验名: $EXP_NAME)"

    # 检查模型是否已经存在，如果存在则跳过训练
    if [ -f "$BEST_MODEL" ]; then
        echo "找到已训练的模型，跳过训练。"
    else
        echo "开始训练..."
        python train_denseuav.py \
            --data_dir $DATA_DIR \
            --batchsize $BATCHSIZE \
            --epochs $EPOCHS \
            --name $EXP_NAME \
            --LPN \
            --block $BLOCK \
            --h $H \
            --w $W \
            --lr $LR \
            --pool $POOL \
            --stride $STRIDE \
            --use_mcr \
            --mask_ratio $ratio \
            --center_ratio $CENTER_RATIO \
            --mask_mode $MASK_MODE

        # 检查训练是否成功
        if [ $? -ne 0 ]; then
            echo "训练失败，跳过此掩码比例。"
            continue
        fi
    fi

    # 评估模型
    echo "开始评估..."
    OUTPUT=$(python eval_denseuav.py --model_dir $MODEL_DIR --which_epoch best)

    # 从输出中提取指标
    R1=$(echo "$OUTPUT" | grep -oP 'R@1:\s*\K[0-9.]+')
    R5=$(echo "$OUTPUT" | grep -oP 'R@5:\s*\K[0-9.]+')
    R10=$(echo "$OUTPUT" | grep -oP 'R@10:\s*\K[0-9.]+')
    AP=$(echo "$OUTPUT" | grep -oP 'AP:\s*\K[0-9.]+')

    # 如果没有提取到，则赋值为0
    [ -z "$R1" ] && R1=0
    [ -z "$R5" ] && R5=0
    [ -z "$R10" ] && R10=0
    [ -z "$AP" ] && AP=0

    echo "结果: R@1=$R1, R@5=$R5, R@10=$R10, AP=$AP"

    # 写入 CSV
    echo "DenseUAV,Drone→Satellite,$ratio,$R1,$R5,$R10,$AP" >> $CSV_FILE

    echo "完成掩码比例 $ratio"
    echo "=================================================="
done

echo "所有实验完成！结果已保存到 $CSV_FILE"
echo "结束时间: $(date)"
shutdown -h now