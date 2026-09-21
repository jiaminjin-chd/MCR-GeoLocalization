#!/bin/bash
# run_dac_small_dual_append.sh

# 全样本结果文件（已手动添加 subset 列）
FULL_CSV="results200_dac_mixheights_dual.csv"

# 小样本转换后的根目录（DAC格式）
SMALL_DAC_BASE="/root/autodl-tmp/SUES-200-512x512_small_dac"

# 子集列表
#SUBSETS=("half" "three_quarter")
SUBSETS=("quarter" "half" "three_quarter")

# 掩码比例列表（双视图）
MASK_RATIOS=(0.0 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)

EPOCHS=1
BATCH_SIZE=24
LR=0.001
GPU_IDS="0"

# 方向固定为 drone->sat（与全样本一致）
DIRECTION="drone->sat"

# 确保全样本文件存在（防止误操作）
if [ ! -f "$FULL_CSV" ]; then
    echo "Error: $FULL_CSV not found. Please ensure it exists with header: direction,subset,mask_ratio,R1,R5,R10,Rtop1,AP"
    exit 1
fi

for subset in "${SUBSETS[@]}"; do
    TRAIN_DIR="${SMALL_DAC_BASE}/${subset}"
    echo "=========================================="
    echo "Running $subset (dual-view, 1 epoch)"
    echo "Train dir: $TRAIN_DIR"
    echo "=========================================="

    for mr in "${MASK_RATIOS[@]}"; do
        echo ">>> mask_ratio=$mr"
        CMD="python train_sues200.py \
            --data_folder $TRAIN_DIR \
            --mix_heights \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --gpu_ids $GPU_IDS"
        if [ "$mr" != "0.0" ]; then
            CMD="$CMD --use_mask --mask_ratio $mr --patch_size 16 --keep_center --center_ratio 0.75 --mask_satellite --mask_drone"
        fi
        eval $CMD

        # 提取指标
        LOG_FILE=$(ls -td ./checkpoints/sues-200/convnext_base.fb_in22k_ft_in1k_384/*/log.txt | head -1)
        if [ -f "$LOG_FILE" ]; then
            r1=$(grep -oP 'Recall@1:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
            r5=$(grep -oP 'Recall@5:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
            r10=$(grep -oP 'Recall@10:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
            rtop1=$(grep -oP 'Recall@top1:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
            ap=$(grep -oP 'AP:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
            echo "$DIRECTION,${subset},$mr,$r1,$r5,$r10,$rtop1,$ap" >> $FULL_CSV
        else
            echo "Warning: log file not found for $subset $mr"
        fi
    done
done

echo "All small-subset dual-view experiments completed. Results appended to $FULL_CSV"
echo "System will shut down in 1 minute..."
shutdown -h +1