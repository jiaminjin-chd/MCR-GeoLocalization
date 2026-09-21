# -*- coding: utf-8 -*-
"""
Grad-CAM visualization for LPN with different masking ratios.
Usage:
    python visualize_gradcam.py \
        --ckpt_root ./model \
        --samples samples/samples.txt \
        --out_dir visualizations \
        --gpu 0
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.cm as cm
import yaml
from torchvision import transforms

from model import three_view_net


# ===================== 参数 =====================
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_root', type=str, default='./model',
                    help='directory that contains all experiment folders')
parser.add_argument('--samples', type=str, default='samples/samples.txt',
                    help='list of samples: query_path sat_path scene_name')
parser.add_argument('--out_dir', type=str, default='visualizations')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--class_num', type=int, default=-1,
                    help='number of training classes; -1 means read from opts.yaml')
parser.add_argument('--block', type=int, default=4)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--alpha', type=float, default=0.5,
                    help='overlay alpha for heatmap')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.out_dir, exist_ok=True)


# ===================== 实验名与掩码比例对应 =====================
# 与 run.sh 中的 exp_name 保持一致
EXP_NAME = {
    0:  'LPN_full_baseline',
    5:  'LPN_full_center0.75_mask0.05',
    10: 'LPN_full_center0.75_mask0.1',
    25: 'LPN_full_center0.75_mask0.25',
    55: 'LPN_full_center0.75_mask0.55',
    95: 'LPN_full_center0.75_mask0.95',
}

# 正文推荐：0, 5, 10, 25, 55, 95
# 补充材料：全比例
P_LIST = [0, 5, 10, 25, 55, 95]


# ===================== 图像预处理 =====================
# 与 test.py 保持一致：Resize + ToTensor + Normalize
transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_image(path):
    """返回 (tensor[1,3,H,W], rgb_uint8[H,W,3])"""
    img = Image.open(path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    rgb = np.array(img.resize((args.img_size, args.img_size)))
    return tensor, rgb


# ===================== Grad-CAM =====================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.grads = None
        self.target_layer.register_forward_hook(self._save_feature)
        self.target_layer.register_full_backward_hook(self._save_grad)

    def _save_feature(self, module, input, output):
        self.features = output.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, drone_tensor, sat_tensor):
        self.model.zero_grad()

        # 1) satellite 分支，不反传
        with torch.no_grad():
            x_sat = self.model.model_1(sat_tensor)      # [1, 2048, block]

        # 2) drone 分支，保留计算图
        x_drone = self.model.model_3(drone_tensor)      # [1, 2048, block]

        # 3) 用 drone 与 GT satellite 特征相似度作为反传目标
        score = (x_drone.flatten(1) * x_sat.flatten(1)).sum()
        score.backward()

        # 4) Grad-CAM
        w = self.grads.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = torch.relu((w * self.features).sum(dim=1))[0]  # [H, W]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


# ===================== 上色 + 叠加 =====================
def colorize_cam(cam):
    """把 [0,1] 的 CAM 变成 JET colormap 的 RGB 图像"""
    cam_uint8 = np.uint8(255 * cam)
    cmap = cm.get_cmap('jet')
    rgba = cmap(cam_uint8 / 255.0)          # [H, W, 4], float in [0,1]
    rgb = np.uint8(rgba[:, :, :3] * 255)
    return rgb


def overlay(rgb, cam, alpha=0.5):
    """把 CAM 叠加到原图上，返回 uint8 RGB"""
    h, w = rgb.shape[:2]
    # 用 PIL 做 resize，避免依赖 cv2
    cam_img = Image.fromarray(np.uint8(cam * 255))
    cam_img = cam_img.resize((w, h), resample=Image.BILINEAR)
    cam = np.array(cam_img).astype(np.float32) / 255.0

    heat = colorize_cam(cam).astype(np.float32)
    base = rgb.astype(np.float32)
    blended = (1 - alpha) * base + alpha * heat
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


# ===================== 读取样本列表 =====================
samples = []
with open(args.samples, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            print(f'[Warn] invalid line: {line}')
            continue
        q, s, name = parts[0], parts[1], parts[2]
        samples.append((q, s, name))

print(f'Loaded {len(samples)} samples.')


# ===================== 按 p 循环 =====================
for p in P_LIST:
    if p not in EXP_NAME:
        print(f'[Skip] p={p} not in EXP_NAME')
        continue
    exp_name = EXP_NAME[p]
    ckpt_dir = os.path.join(args.ckpt_root, exp_name)
    if not os.path.isdir(ckpt_dir):
        print(f'[Skip] {ckpt_dir} not found')
        continue

    # 找最新 model_*.pth
    ckpts = [f for f in os.listdir(ckpt_dir)
             if f.startswith('net_') and f.endswith('.pth')]
    if len(ckpts) == 0:
        print(f'[Skip] no checkpoint in {ckpt_dir}')
        continue
    ckpts.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    print(f'>>> p={p}% loading {ckpt_path}')

    # 构建模型
    # 从 opts.yaml 读取该实验的 nclasses
    opts_path = os.path.join(ckpt_dir, 'opts.yaml')
    with open(opts_path, 'r') as f:
        cfg = yaml.safe_load(f)
    nclasses = cfg.get('nclasses', 689)   # 训练集 location 数
    print(f'    nclasses={nclasses}, block={cfg.get("block", 4)}, '
          f'stride={cfg.get("stride", 1)}, views={cfg.get("views", 3)}')

    # 构建模型，参数必须和训练时完全一致
    model = three_view_net(
        class_num=nclasses,
        droprate=0.75,
        stride=cfg.get('stride', 1),
        pool='avg',
        share_weight=True,
        LPN=True,
        block=cfg.get('block', 4),
    )
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    # 目标层：drone 分支 ResNet50 的 layer4[-1]
    # share_weight=True 时 model_3 就是 model_1
    target_layer = model.model_3.model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    out_p_dir = os.path.join(args.out_dir, f'p{p}')
    os.makedirs(out_p_dir, exist_ok=True)

    for query_path, sat_path, name in samples:
        if not os.path.exists(query_path):
            print(f'[Warn] query not found: {query_path}')
            continue
        if not os.path.exists(sat_path):
            print(f'[Warn] sat not found: {sat_path}')
            continue

        drone_tensor, drone_rgb = load_image(query_path)
        sat_tensor, _ = load_image(sat_path)
        drone_tensor = drone_tensor.to(device)
        sat_tensor = sat_tensor.to(device)

        cam = gradcam(drone_tensor, sat_tensor)
        overlay_img = overlay(drone_rgb, cam, alpha=args.alpha)

        save_path = os.path.join(out_p_dir, f'{name}_overlay.png')
        Image.fromarray(overlay_img).save(save_path)
        print(f'Saved {save_path}')

print('Done.')