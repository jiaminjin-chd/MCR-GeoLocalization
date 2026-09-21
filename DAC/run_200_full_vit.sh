#!/bin/bash
# run_sues200_vit.sh
export HF_ENDPOINT=https://hf-mirror.com
DATA_FOLDER="/root/autodl-tmp/SUES-200-512x512"
MASK_RATIOS=(0.0 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)
OUTPUT_CSV="results_sues200_vit.csv"
EPOCHS=1
BATCH_SIZE=8
LR=0.0001
GPU_IDS="0"
DIRECTION="drone->sat"
MODEL="vit_base_patch16_384"
IMG_SIZE=384

echo "direction,mask_ratio,R1,R5,R10,Rtop1,AP" > $OUTPUT_CSV

for mr in "${MASK_RATIOS[@]}"; do
    echo "=========================================="
    echo "Running $MODEL mask_ratio=$mr"
    echo "=========================================="
    CMD="python train_sues200.py \
        --data_folder $DATA_FOLDER \
        --mix_heights \
        --model $MODEL \
        --img_size $IMG_SIZE \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --gpu_ids $GPU_IDS "
    if [ "$mr" != "0.0" ]; then
        CMD="$CMD --use_mask --mask_ratio $mr --patch_size 16 --keep_center --center_ratio 0.75 --mask_satellite --mask_drone"
    fi
    eval $CMD
    # 动态查找最新日志
    LOG_FILE=$(ls -td ./checkpoints/sues-200/vit_base_patch16_384/*/log.txt 2>/dev/null | head -1)
    if [ -f "$LOG_FILE" ]; then
        r1=$(grep -oP 'Recall@1:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
        r5=$(grep -oP 'Recall@5:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
        r10=$(grep -oP 'Recall@10:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
        rtop1=$(grep -oP 'Recall@top1:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
        ap=$(grep -oP 'AP:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
        echo "$DIRECTION,$mr,$r1,$r5,$r10,$rtop1,$ap" >> $OUTPUT_CSV
    else
        echo "Warning: log file not found for $mr"
    fi
done

echo "All experiments completed. Results saved to $OUTPUT_CSV"