#!/bin/bash
# run_dac_mixheights_dual_masks.sh

DATA_FOLDER="/root/autodl-tmp/SUES-200-512x512"
#MASK_RATIOS=(0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)
MASK_RATIOS=(0.0 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)
OUTPUT_CSV="results200_dac_mixheights_dual.csv"
EPOCHS=1
BATCH_SIZE=24
LR=0.001
GPU_IDS="0"
DIRECTION="drone->sat"

echo "direction,mask_ratio,R1,R5,R10,Rtop1,AP" > $OUTPUT_CSV

for mr in "${MASK_RATIOS[@]}"; do
    echo "=========================================="
    echo "Running mask_ratio=$mr (dual-view)"
    echo "=========================================="
    CMD="python train_sues200.py \
        --data_folder $DATA_FOLDER \
        --mix_heights \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --gpu_ids $GPU_IDS"
    if [ "$mr" != "0.0" ]; then
        CMD="$CMD --use_mask --mask_ratio $mr --patch_size 16 --keep_center --center_ratio 0.75 --mask_satellite --mask_drone"
    fi
    eval $CMD
    # 从最新日志中提取指标
    LOG_FILE=$(ls -td ./checkpoints/sues-200/convnext_base.fb_in22k_ft_in1k_384/*/log.txt | head -1)
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

#echo "All experiments completed. Results saved to $OUTPUT_CSV"
#echo "System will shut down in 1 minute..."
#shutdown -h +1