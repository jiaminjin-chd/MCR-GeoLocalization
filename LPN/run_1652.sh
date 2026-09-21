#!/bin/bash

# 全样本数据集路径
data_dir="/root/autodl-tmp/University-Release/train"

# 掩码比例列表 (0.05 到 0.95，步长 0.1)
mask_ratios=(0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)

# 结果汇总文件（追加模式）
summary_file="summary.txt"

# 确保表头存在（如果文件为空或没有表头，则添加；但之前已有，可跳过）
# 直接追加结果

# 1. 基线（无掩码）
exp_name="LPN_full_baseline"
echo ">>> Running baseline on full dataset"
python train.py \
    --name="$exp_name" \
    --data_dir="$data_dir" \
    --views=3 --droprate=0.75 --share --stride=1 --h=256 --w=256 \
    --LPN --extra --block=4 --lr=0.001 --gpu_ids='0'

# 测试基线
test_output=$(python test.py --name="$exp_name" --test_dir='/root/autodl-tmp/University-Release/test' --batchsize=128 --gpu_ids='0' 2>&1)
r1=$(echo "$test_output" | grep -oP 'Recall@1:\s*\K[0-9.]+')
r5=$(echo "$test_output" | grep -oP 'Recall@5:\s*\K[0-9.]+')
r10=$(echo "$test_output" | grep -oP 'Recall@10:\s*\K[0-9.]+')
ap=$(echo "$test_output" | grep -oP 'AP:\s*\K[0-9.]+')
echo "full | baseline | $r1 | $r5 | $r10 | $ap" >> $summary_file

# 清理中间模型
python clean_models.py --name="$exp_name"

# 2. 循环掩码比例
for mr in "${mask_ratios[@]}"; do
    exp_name="LPN_full_center0.75_mask${mr}"
    echo ">>> Running mask ratio $mr on full dataset"
    python train.py \
        --name="$exp_name" \
        --data_dir="$data_dir" \
        --views=3 --droprate=0.75 --share --stride=1 --h=256 --w=256 \
        --LPN --extra --block=4 --lr=0.001 --gpu_ids='0' \
        --use_mask --mask_ratio=$mr --patch_size=16 --keep_center --center_ratio=0.75

    test_output=$(python test.py --name="$exp_name" --test_dir='/root/autodl-tmp/University-Release/test' --batchsize=128 --gpu_ids='0' 2>&1)
    r1=$(echo "$test_output" | grep -oP 'Recall@1:\s*\K[0-9.]+')
    r5=$(echo "$test_output" | grep -oP 'Recall@5:\s*\K[0-9.]+')
    r10=$(echo "$test_output" | grep -oP 'Recall@10:\s*\K[0-9.]+')
    ap=$(echo "$test_output" | grep -oP 'AP:\s*\K[0-9.]+')
    echo "full | $mr | $r1 | $r5 | $r10 | $ap" >> $summary_file

    python clean_models.py --name="$exp_name"
done

echo "All full-dataset experiments finished. Results appended to $summary_file"

