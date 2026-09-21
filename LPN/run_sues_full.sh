#!/bin/bash

# SUES-200 数据集路径（训练与测试共用同一个根目录）
data_dir="/root/autodl-tmp/SUES-200-512x512"

# 掩码比例列表 (0.05 到 0.95，步长 0.1)
mask_ratios=(0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)

# 结果文件
summary_file="summary_sues_single.txt"
csv_file="summary_sues_single.csv"

# 写入表头（如果文件不存在）
if [ ! -f "$csv_file" ]; then
    echo "type,mask_ratio,R1,R5,R10,mAP" > $csv_file
fi

# 1. 记录基线（无掩码）结果（已提前跑完）
echo ">>> Baseline (no mask) result (already computed):"
r1_base=80.15
r5_base=83.54
r10_base=87.31
mAP_base=81.18
echo "sues | baseline | $r1_base | $r5_base | $r10_base | $mAP_base" >> $summary_file
echo "sues,baseline,$r1_base,$r5_base,$r10_base,$mAP_base" >> $csv_file

# 2. 循环掩码比例（仅对无人机视图掩码）
for mr in "${mask_ratios[@]}"; do
    exp_name="sues_single_mask_${mr}"
    echo ">>> Running single-view mask ratio $mr on SUES-200"

    # 训练（使用 train_sues.py）
    python train_sues.py \
        --name="$exp_name" \
        --data_dir="$data_dir" \
        --views=2 --droprate=0.5 --share --stride=2 --h=384 --w=384 \
        --LPN --block=6 --lr=0.01 --batchsize=8 --gpu_ids='0' \
        --use_mask --mask_ratio=$mr --patch_size=16 --keep_center --center_ratio=0.75

    # 测试（使用 test_sues.py）
    test_output=$(python test_sues.py --name="$exp_name" --test_dir="$data_dir" --batchsize=128 --gpu_ids='0' 2>&1)
    r1=$(echo "$test_output" | grep -oP 'Recall@1:\s*\K[0-9.]+')
    r5=$(echo "$test_output" | grep -oP 'Recall@5:\s*\K[0-9.]+')
    r10=$(echo "$test_output" | grep -oP 'Recall@10:\s*\K[0-9.]+')
    mAP=$(echo "$test_output" | grep -oP 'AP:\s*\K[0-9.]+')

    # 写入结果
    echo "sues | $mr | $r1 | $r5 | $r10 | $mAP" >> $summary_file
    echo "sues,$mr,$r1,$r5,$r10,$mAP" >> $csv_file

    # 可选：清理中间模型（若空间紧张可开启）
    # python clean_models.py --name="$exp_name"
done

echo "All SUES-200 single-view masking experiments finished."
echo "Results saved to $summary_file and $csv_file"