#!/bin/bash

# ============================================
# 单子集单视图掩码实验脚本
# 用法: ./run_one_subset.sh <subset_name>
# 示例: ./run_one_subset.sh quarter
# ============================================

# 检查参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <subset_name>"
    echo "  subset_name: quarter, half, three_quarter"
    exit 1
fi

SUBSET=$1

# 定义路径
BASE_DATA_DIR="/root/autodl-tmp/SUES-200-512x512_small"
DATA_DIR="${BASE_DATA_DIR}/${SUBSET}"

# 检查数据集目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist. Please run split_sues200_small.py first."
    exit 1
fi

# 结果汇总文件（与全样本共用一个大表）
SUMMARY_TXT="summary_sues_single.txt"
SUMMARY_CSV="summary_sues_single.csv"

# 掩码比例列表（可根据需要减少）
MASK_RATIOS=(0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)

# 训练参数（与全样本保持一致）
VIEWS=2
DROPRATE=0.5
SHARE="--share"      # 使用权重共享
STRIDE=2
H=384
W=384
LPN="--LPN"
BLOCK=6
LR=0.01
BATCH_SIZE=8
GPU_IDS="0"

# --------------------------------------------------
# 函数：测试并提取指标
# --------------------------------------------------
run_test() {
    local exp_name=$1
    local data_dir=$2
    local output=$(python test_sues.py \
        --name "$exp_name" \
        --test_dir "/root/autodl-tmp/SUES-200-512x512" \
        --batchsize 128 \
        --gpu_ids "$GPU_IDS" 2>&1)
    # 提取 Recall@1, Recall@5, Recall@10, mAP
    r1=$(echo "$output" | grep -oP 'Recall@1:\s*\K[0-9.]+')
    r5=$(echo "$output" | grep -oP 'Recall@5:\s*\K[0-9.]+')
    r10=$(echo "$output" | grep -oP 'Recall@10:\s*\K[0-9.]+')
    mAP=$(echo "$output" | grep -oP 'AP:\s*\K[0-9.]+')
    echo "$r1 $r5 $r10 $mAP"
}

# --------------------------------------------------
# 确保汇总文件有表头
# --------------------------------------------------
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "type,mask_ratio,R1,R5,R10,mAP" > "$SUMMARY_CSV"
fi
if [ ! -f "$SUMMARY_TXT" ]; then
    echo "type | mask_ratio | R1 | R5 | R10 | mAP" > "$SUMMARY_TXT"
fi

# --------------------------------------------------
# 1. 基线（无掩码）
# --------------------------------------------------
BASELINE_EXP_NAME="sues_${SUBSET}_baseline"
echo ">>> Training baseline for $SUBSET (no mask)..."
python train_sues.py \
    --name "$BASELINE_EXP_NAME" \
    --data_dir "$DATA_DIR" \
    --views $VIEWS --droprate $DROPRATE $SHARE --stride $STRIDE \
    --h $H --w $W $LPN --block $BLOCK --lr $LR --batchsize $BATCH_SIZE \
    --gpu_ids "$GPU_IDS"

echo ">>> Testing baseline for $SUBSET..."
read r1 r5 r10 mAP <<< $(run_test "$BASELINE_EXP_NAME" "$DATA_DIR")
echo "$SUBSET | baseline | $r1 | $r5 | $r10 | $mAP" >> "$SUMMARY_TXT"
echo "$SUBSET,baseline,$r1,$r5,$r10,$mAP" >> "$SUMMARY_CSV"

# --------------------------------------------------
# 2. 循环掩码比例
# --------------------------------------------------
for mr in "${MASK_RATIOS[@]}"; do
    EXP_NAME="sues_${SUBSET}_single_mask_${mr}"
    echo ">>> Training single-view mask ratio $mr on $SUBSET..."
    python train_sues.py \
        --name "$EXP_NAME" \
        --data_dir "$DATA_DIR" \
        --views $VIEWS --droprate $DROPRATE $SHARE --stride $STRIDE \
        --h $H --w $W $LPN --block $BLOCK --lr $LR --batchsize $BATCH_SIZE \
        --gpu_ids "$GPU_IDS" \
        --use_mask --mask_ratio=$mr --patch_size=16 --keep_center --center_ratio=0.75

    echo ">>> Testing mask ratio $mr on $SUBSET..."
    read r1 r5 r10 mAP <<< $(run_test "$EXP_NAME" "$DATA_DIR")
    echo "$SUBSET | $mr | $r1 | $r5 | $r10 | $mAP" >> "$SUMMARY_TXT"
    echo "$SUBSET,$mr,$r1,$r5,$r10,$mAP" >> "$SUMMARY_CSV"
done

echo "=========================================="
echo "All experiments for $SUBSET finished."
echo "Results appended to $SUMMARY_TXT and $SUMMARY_CSV"
echo "=========================================="