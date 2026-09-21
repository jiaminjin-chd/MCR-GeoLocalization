# -*- coding: utf-8 -*-
import os
from PIL import Image

P_LIST = [0, 5, 10, 25, 55, 95]
SCENES = ['scene1', 'scene2', 'scene3']
OUT_DIR = 'visualizations'
SIZE = (256, 256)

rows = []
for scene in SCENES:
    row_imgs = []

    original = Image.open(f'samples/{scene}_query.png').convert('RGB').resize(SIZE)
    row_imgs.append(original)

    for p in P_LIST:
        img_path = os.path.join(OUT_DIR, f'p{p}', f'{scene}_overlay.png')
        if not os.path.exists(img_path):
            print(f'[Warn] {img_path} not found, use black')
            img = Image.new('RGB', SIZE, (0, 0, 0))
        else:
            img = Image.open(img_path).convert('RGB').resize(SIZE)
        row_imgs.append(img)

    # 横向拼接
    row = Image.new('RGB', (SIZE[0] * len(row_imgs), SIZE[1]))
    for i, im in enumerate(row_imgs):
        row.paste(im, (i * SIZE[0], 0))
    rows.append(row)

# 纵向拼接
total_h = SIZE[1] * len(rows)
total_w = SIZE[0] * (len(P_LIST) + 1)
long_img = Image.new('RGB', (total_w, total_h))
for i, row in enumerate(rows):
    long_img.paste(row, (0, i * SIZE[1]))

os.makedirs(OUT_DIR, exist_ok=True)
long_img.save(os.path.join(OUT_DIR, 'heatmap_long.png'))
print('Saved visualizations/heatmap_long.png')