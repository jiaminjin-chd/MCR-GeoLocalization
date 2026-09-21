# Masked Collaborative Reasoning (MCR)

Official implementation of the paper:

> **Masked Collaborative Reasoning: Learning to Align Multi-Source Images from Partial Multi-View Observations**

Manuscript under review.

## Abstract

Drone-based multi-view geo-localization aims to establish reliable correspondences between drone-view and satellite-view images, yet it remains hindered by two intertwined obstacles: viewpoint discrepancies between oblique drone views and orthographic satellite imagery, and the heavy demand for annotated training pairs. This paper presents Masked Collaborative Reasoning (MCR), a training strategy that cultivates spatial reasoning in siamese matching networks without modifying their architecture or adding trainable parameters. MCR adopts center-preserving peripheral masking, retaining the central target region as a spatial anchor while randomly masking peripheral areas, so that each branch must infer missing spatial structures from visible context. Since the two branches share weights, this design enforces an implicit consistency constraint between reasoning-induced representations across views and requires no auxiliary loss. Experiments on University-1652, SUES-200, and DenseUAV show consistent gains across siamese matching baselines of varying capacities, with the largest gains on lightweight backbones. On University-1652, MCR attains 95.06% R@1 and 95.85% average precision (AP) with a single ConvNeXt-B backbone, matching multi-modal methods of substantially higher complexity. Under reduced-data regimes, MCR trained on only 50% of the data surpasses the fully supervised baseline. These results indicate that structured peripheral masking is a simple yet effective inductive bias for accurate and data-efficient cross-view visual localization.

## Contents

- `LPN/` — MCR built on the LPN framework (ResNet-50 backbone)
- `DAC/` — MCR built on the DAC framework (ConvNeXt / ViT backbones)

## Requirements

- Python 3.8
- PyTorch 2.4.1 (CUDA 11.8)
- torchvision 0.19.1
- NumPy 1.24.4
- SciPy 1.10.1
- Matplotlib 3.7.5
- Pillow 10.4.0
- PyYAML 6.0.3

All experiments are conducted on a single NVIDIA GeForce RTX 4090 GPU (24 GB memory).

## Contact

2026124063@chd.edu.cn