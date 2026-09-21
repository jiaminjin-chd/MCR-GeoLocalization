# -*- coding: utf-8 -*-
"""
生成论文级热力图（无文字版）：
- 3 行场景 x 7 列
- 行列间隙统一，用绝对 inch 控制
- 不带 colorbar
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P_LIST = [0, 5, 10, 25, 55, 95]
SCENES = [
    ('scene1', 'Simple'),
    ('scene2', 'Complex'),
    ('scene3', 'Low-texture'),
]
OUT_DIR = 'visualizations'
SIZE = (256, 256)

n_rows = len(SCENES)
n_cols = 1 + len(P_LIST)

# ========== 用绝对 inch 控制 ==========
img_inch = 1.5       # 每个子图 1.5 inch
gap_inch = 0.05      # 行/列间隙统一 0.05 inch
left_inch = 0.35     # 左侧留白，给行标签
top_inch = 0.35      # 顶部留白，给列标签
bottom_inch = 0.05
right_inch = 0.05

fig_w = left_inch + n_cols * img_inch + (n_cols - 1) * gap_inch + right_inch
fig_h = top_inch + n_rows * img_inch + (n_rows - 1) * gap_inch + bottom_inch

fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)

# 转成归一化坐标
def nx(x_inch):
    return x_inch / fig_w

def ny(y_inch):
    return y_inch / fig_h

for r in range(n_rows):
    for c in range(n_cols):
        left_inch_  = left_inch + c * (img_inch + gap_inch)
        bottom_inch_ = bottom_inch + (n_rows - 1 - r) * (img_inch + gap_inch)

        ax = fig.add_axes([
            nx(left_inch_),
            ny(bottom_inch_),
            nx(img_inch),
            ny(img_inch),
        ])

        if c == 0:
            img_path = f'samples/{SCENES[r][0]}_query.png'
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB').resize(SIZE)
                ax.imshow(np.array(img))
            else:
                ax.imshow(np.zeros((*SIZE, 3), dtype=np.uint8))
        else:
            p = P_LIST[c - 1]
            path = os.path.join(OUT_DIR, f'p{p}', f'{SCENES[r][0]}_overlay.png')
            if os.path.exists(path):
                img = Image.open(path).convert('RGB').resize(SIZE)
                ax.imshow(np.array(img))
            else:
                ax.imshow(np.zeros((*SIZE, 3), dtype=np.uint8))

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, 'heatmap_paper_blank.png')
plt.savefig(out_path, dpi=300)
print(f'Saved {out_path}')
print(f'Figure size: {fig_w:.2f} x {fig_h:.2f} inch')