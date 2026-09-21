#!/bin/bash


cd /root/autodl-tmp/DAC-main


MASKS=(
0
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


for DIR in D2S S2D
do

echo "======================"
echo $DIR
echo "======================"


for MASK in ${MASKS[@]}
do

echo "Running $DIR mask=$MASK"


python train_denseuav_v2.py \
--direction $DIR \
--mask_ratio $MASK \
--epochs 30 \
--batch_size 24


done

done


python collect_denseuav_results.py


shutdown -h now