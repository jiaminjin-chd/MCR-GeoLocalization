#!/bin/bash

# 定义数据集路径
declare -A datasets
datasets["quarter"]="/root/autodl-tmp/University-Release/quarter"
datasets["half"]="/root/autodl-tmp/University-Release/half"
datasets["three_quarter"]="/root/autodl-tmp/University-Release/three_quarter"
datasets["full"]="/root/autodl-tmp/University-Release"

# 为每个数据集定义要跑的掩码比例（可以自由修改）
declare -A mask_lists
mask_lists["quarter"]="0.025 0.075 0.2"
mask_lists["half"]="0.075 0.125 0.2"
mask_lists["three_quarter"]="0.025 0.075"
mask_lists["full"]="0.075 0.125 0.2"

# 结果文件
summary="dac_summary.txt"

# 确保表头存在（如果文件为空或不存在则添加）
if [ ! -s "$summary" ]; then
    echo "Dataset | MaskRatio | R@1 | R@5 | R@10 | R@top1 | AP" > "$summary"
fi

# 临时日志文件
tmp_log="tmp_log.txt"

for dataset in quarter half three_quarter full; do
    data_dir=${datasets[$dataset]}
    mask_list=${mask_lists[$dataset]}

    # 如果该数据集没有定义掩码比例，则跳过
    if [ -z "$mask_list" ]; then
        echo "No masks defined for $dataset, skipping."
        continue
    fi

    for mr in $mask_list; do
        echo ">>> Running $dataset with mask ratio $mr"
        python train_university.py \
            --data_folder "$data_dir" \
            --dataset_name '' \
            --mask_ratio "$mr" \
            --keep_center \
            --center_ratio 0.75 \
            --patch_size 16 \
            --name "DAC_${dataset}_mask${mr}" 2>&1 | tee "$tmp_log"

        # 提取指标
        r1=$(grep -oP 'Recall@1:\s*\K[0-9.]+' "$tmp_log" | head -1)
        r5=$(grep -oP 'Recall@5:\s*\K[0-9.]+' "$tmp_log" | head -1)
        r10=$(grep -oP 'Recall@10:\s*\K[0-9.]+' "$tmp_log" | head -1)
        rtop1=$(grep -oP 'Recall@top1:\s*\K[0-9.]+' "$tmp_log" | head -1)
        ap=$(grep -oP 'AP:\s*\K[0-9.]+' "$tmp_log" | head -1)

        # 追加到 summary
        echo "$dataset | $mr | $r1 | $r5 | $r10 | $rtop1 | $ap" >> "$summary"
        echo "Result appended: $dataset | $mr | $r1 | $r5 | $r10 | $rtop1 | $ap"

        # 可选：删除模型文件（节省空间）
        rm -rf "./checkpoints/university/convnext_base.fb_in22k_ft_in1k_384/DAC_${dataset}_mask${mr}"
    done
done

rm -f "$tmp_log"
echo "All custom experiments finished. Results saved in $summary"