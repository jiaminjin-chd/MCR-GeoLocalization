#!/bin/bash

# ======================================================
# DenseUAV DAC ViT Mask Ablation Experiments
# ======================================================

PYTHON=python
SCRIPT=train_denseuav_v2.py

EPOCHS=70
BATCH=24
LR=1e-5


# mask ratios
RATIOS=(
0.00
0.05
0.15
0.25
0.35
0.45
0.55
0.65
0.75
0.85
0.95
)


# directions
DIRECTIONS=(
D2S
S2D
)


for direction in ${DIRECTIONS[@]}
do

    for ratio in ${RATIOS[@]}
    do

        echo "======================================"
        echo "Running:"
        echo "Direction: ${direction}"
        echo "Mask Ratio: ${ratio}"
        echo "======================================"


        ${PYTHON} ${SCRIPT} \
        --direction ${direction} \
        --mask_ratio ${ratio} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH} \
        --lr ${LR}


        echo "Finished:"
        echo "${direction} mask=${ratio}"

        echo

    done

done


echo "======================================"
echo "All DenseUAV mask experiments finished"
echo "======================================"

shutdown -h now