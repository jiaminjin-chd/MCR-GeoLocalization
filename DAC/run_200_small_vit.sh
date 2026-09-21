#!/bin/bash
# run_dac_vit_small.sh
# 在 DAC 框架下使用 ViT 进行小样本双视图掩码实验（1 epoch），结果追加到全样本文件

SMALL_DAC_BASE="/root/autodl-tmp/SUES-200-512x512_small_dac"
SUBSETS=("quarter" "half" "three_quarter")
MASK_RATIOS=(0.0 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)

# 目标文件（已包含 full 数据）
OUTPUT_CSV="results_sues200_vit_complete.csv"
DIRECTION="drone->sat"

EPOCHS=1
BATCH_SIZE=8
LR=0.0001
GPU_IDS="0"
MODEL="vit_base_patch16_384"
IMG_SIZE=384

# 确保目标文件存在且表头正确
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "direction,subset,mask_ratio,R1,R5,R10,Rtop1,AP" > $OUTPUT_CSV
fi

for subset in "${SUBSETS[@]}"; do
    TRAIN_DIR="${SMALL_DAC_BASE}/${subset}"
    if [ ! -d "$TRAIN_DIR" ]; then
        echo "Warning: $TRAIN_DIR not found. Skipping $subset."
        continue
    fi
    echo "=========================================="
    echo "Running $subset (ViT, 1 epoch)"
    echo "Train dir: $TRAIN_DIR"
    echo "=========================================="

    for mr in "${MASK_RATIOS[@]}"; do
        echo ">>> mask_ratio=$mr"
        CMD="python train_sues200.py \
            --data_folder $TRAIN_DIR \
            --mix_heights \
            --model $MODEL \
            --img_size $IMG_SIZE \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --gpu_ids $GPU_IDS"
        if [ "$mr" != "0.0" ]; then
            CMD="$CMD --use_mask --mask_ratio $mr --patch_size 16 --keep_center --center_ratio 0.75 --mask_satellite --mask_drone"
        fi
        eval $CMD

        LOG_FILE=$(ls -td ./checkpoints/sues-200/vit_base_patch16_384/*/log.txt 2>/dev/null | head -1)
        if [ -f "$LOG_FILE" ]; then
            r1=$(grep -oP 'Recall@1:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
            r5=$(grep -oP 'Recall@5:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
            r10=$(grep -oP 'Recall@10:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
            rtop1=$(grep -oP 'Recall@top1:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
            ap=$(grep -oP 'AP:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
            echo "$DIRECTION,$subset,$mr,$r1,$r5,$r10,$rtop1,$ap" >> $OUTPUT_CSV
            echo "  Results for $subset $mr: R1=$r1, AP=$ap"
        else
            echo "Warning: log file not found for $subset $mr"
        fi
    done
done

echo "All small-subset experiments completed. Results appended to $OUTPUT_CSV"
# 如需自动关机，取消注释下一行
# shutdown -h +1