# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
import numpy as np
import os
import time
import scipy.io
import yaml
import math
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import two_view_net

# 消除 OMP_NUM_THREADS 警告
os.environ['OMP_NUM_THREADS'] = '1'

# ----------------------------------------------------------------------
# 自定义测试数据集（增加高度提取 + 图片验证）
# ----------------------------------------------------------------------
class SUESGalleryQueryDataset(Dataset):
    def __init__(self, root_dir, view='satellite', transform=None, ids=None):
        self.root = root_dir
        self.view = view
        self.transform = transform
        self.samples = []
        if view == 'satellite':
            base_dir = os.path.join(root_dir, 'satellite-view')
        else:
            base_dir = os.path.join(root_dir, 'drone_view_512')

        all_ids = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if ids is not None:
            all_ids = [i for i in all_ids if i in ids]
        all_ids = sorted(all_ids)
        label_map = {id_: idx for idx, id_ in enumerate(all_ids)}

        for id_ in all_ids:
            label = label_map[id_]
            if view == 'satellite':
                id_dir = os.path.join(base_dir, id_)
                # 搜索所有常见图片格式
                img_files = [f for f in os.listdir(id_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not img_files:
                    print(f"Warning: No image found in {id_dir}, skipping this ID")
                    continue
                # 优先选择 '0.png' 或 '0.jpg'
                priority = ['0.png', '0.jpg']
                chosen = None
                for p in priority:
                    if p in img_files:
                        chosen = p
                        break
                if chosen is None:
                    chosen = img_files[0]
                img_path = os.path.join(id_dir, chosen)
                # 验证图片是否可读
                try:
                    with Image.open(img_path) as _:
                        pass
                except Exception as e:
                    print(f"Warning: Cannot open image {img_path}: {e}, skipping")
                    continue
                self.samples.append((img_path, label))
            else:  # drone
                drone_id_dir = os.path.join(base_dir, id_)
                if not os.path.isdir(drone_id_dir):
                    continue
                for root, dirs, files in os.walk(drone_id_dir):
                    for fname in files:
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(root, fname)
                            # 验证图片是否可读
                            try:
                                with Image.open(img_path) as _:
                                    pass
                            except Exception as e:
                                print(f"Warning: Cannot open image {img_path}: {e}, skipping")
                                continue
                            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            # 如果打开失败，打印错误并返回一个默认的黑图（防止崩溃，但建议提前发现）
            print(f"Error loading image {path}: {e}")
            # 返回一个全零张量（但会污染结果，这里仅为防止崩溃）
            img = Image.new('RGB', (384, 384), (0, 0, 0))
        if self.transform:
            img = self.transform(img)

        if self.view == 'drone':
            # 高度信息在路径的父目录名中，如 .../0001/150/xxx.jpg
            height_str = os.path.basename(os.path.dirname(path))
            try:
                height = int(height_str)
            except:
                height = -1
            return img, label, height
        else:
            return img, label

# ----------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Test on SUES-200 with height')
parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--which_epoch', default='last', type=str)
parser.add_argument('--test_dir', default='/root/autodl-tmp/SUES-200-512x512', type=str)
parser.add_argument('--name', required=True, type=str)
parser.add_argument('--pool', default='avg', type=str)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--h', default=384, type=int)
parser.add_argument('--w', default=384, type=int)
parser.add_argument('--views', default=2, type=int)
parser.add_argument('--pad', default=0, type=int)
parser.add_argument('--use_dense', action='store_true')
parser.add_argument('--LPN', action='store_true')
parser.add_argument('--multi', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--scale_test', action='store_true')
parser.add_argument('--ms', default='1', type=str)
opt = parser.parse_args()
opt.data_dir = opt.test_dir

# 补全属性
default_attrs = {
    'data_dir': None,
    'train_all': False,
    'droprate': 0.5,
    'color_jitter': False,
    'batchsize': 128,
    'h': 384,
    'w': 384,
    'share': False,
    'stride': 2,
    'LPN': False,
    'pool': 'avg',
    'erasing_p': 0,
    'lr': 0.01,
    'use_dense': False,
    'fp16': False,
    'views': 2,
    'block': 6,
    'gpu_ids': '0',
}
for k, v in default_attrs.items():
    if not hasattr(opt, k):
        setattr(opt, k, v)

# 加载训练配置
config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)

opt.fp16 = config.get('fp16', False)
opt.use_dense = config.get('use_dense', False)
opt.use_NAS = config.get('use_NAS', False)
opt.stride = config.get('stride', 2)
opt.views = config.get('views', 2)
opt.LPN = config.get('LPN', False)
opt.block = config.get('block', 6)
opt.h = config.get('h', 384)
opt.w = config.get('w', 384)

if 'nclasses' in config:
    opt.nclasses = config['nclasses']
else:
    sat_dir = os.path.join(opt.test_dir, 'satellite-view')
    if os.path.exists(sat_dir):
        opt.nclasses = len([d for d in os.listdir(sat_dir) if os.path.isdir(os.path.join(sat_dir, d))])
        print(f'[Info] nclasses inferred: {opt.nclasses}')
    else:
        opt.nclasses = 200
        print(f'[Warning] Using default nclasses {opt.nclasses}')

print(f'Using test config: h={opt.h}, w={opt.w}, block={opt.block}, LPN={opt.LPN}')

# GPU
str_ids = opt.gpu_ids.split(',')
gpu_ids = [int(id) for id in str_ids if int(id) >= 0]
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

# 多尺度
ms = [float(s) for s in opt.ms.split(',')]
if opt.scale_test:
    ms = [math.sqrt(s) for s in ms]
print(f'Scales: {ms}')

# 数据变换
data_transforms = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 构建数据集（注意 num_workers=0 便于调试）
gallery_dataset = SUESGalleryQueryDataset(opt.test_dir, view='satellite', transform=data_transforms)
query_dataset = SUESGalleryQueryDataset(opt.test_dir, view='drone', transform=data_transforms)

gallery_loader = DataLoader(gallery_dataset, batch_size=opt.batchsize, shuffle=False,
                            num_workers=0, pin_memory=True)
query_loader = DataLoader(query_dataset, batch_size=opt.batchsize, shuffle=False,
                          num_workers=0, pin_memory=True)

print(f'Gallery size: {len(gallery_dataset)}, Query size: {len(query_dataset)}')

# ----------------------------------------------------------------------
# 特征提取函数
# ----------------------------------------------------------------------
def fliplr(img):
    inv_idx = torch.arange(img.size(3)-1, -1, -1, device=img.device).long()
    return img.index_select(3, inv_idx)

def extract_feature(model, dataloader, view_type):
    features, labels, heights = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if view_type == 2:
                imgs, lbls, hts = batch
                heights.extend(hts.cpu().numpy() if torch.is_tensor(hts) else hts)
            else:
                imgs, lbls = batch

            imgs = imgs.cuda()
            n = imgs.size(0)
            if opt.LPN:
                ff = torch.zeros(n, 512, opt.block).cuda()
            else:
                ff = torch.zeros(n, 512).cuda()

            for i in range(2):
                if i == 1:
                    imgs = fliplr(imgs)
                for scale in ms:
                    if scale != 1:
                        imgs_scaled = nn.functional.interpolate(imgs, scale_factor=scale,
                                                                 mode='bilinear', align_corners=False)
                    else:
                        imgs_scaled = imgs
                    if opt.views == 2:
                        if view_type == 1:
                            out, _ = model(imgs_scaled, None)
                        else:
                            _, out = model(None, imgs_scaled)
                    ff += out

            if opt.LPN:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block)
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

            features.append(ff.data.cpu())
            labels.extend(lbls.cpu().numpy())

    features = torch.cat(features, dim=0)
    labels = np.array(labels)
    return features, labels, heights if view_type == 2 else None

# ----------------------------------------------------------------------
# 手动加载模型
# ----------------------------------------------------------------------
print('Loading model manually...')
model = two_view_net(opt.nclasses, droprate=opt.droprate, stride=opt.stride,
                     pool=opt.pool, share_weight=opt.share, LPN=opt.LPN, block=opt.block)
model = model.cuda()

model_dir = os.path.join('./model', opt.name)
if opt.which_epoch == 'last':
    pth_files = [f for f in os.listdir(model_dir) if f.startswith('net_') and f.endswith('.pth')]
    if pth_files:
        def get_num(fname):
            try:
                return int(fname.split('_')[1].split('.')[0])
            except:
                return -1
        pth_files.sort(key=get_num)
        model_path = os.path.join(model_dir, pth_files[-1])
    else:
        raise FileNotFoundError(f'No model file found in {model_dir}')
else:
    model_path = os.path.join(model_dir, f'net_{opt.which_epoch}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file {model_path} not found')

print(f'Loading model from {model_path}')
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()

# 去掉分类头（用于特征提取）
if opt.LPN:
    for i in range(opt.block):
        cls_name = 'classifier' + str(i)
        c = getattr(model, cls_name)
        c.classifier = nn.Sequential()
else:
    model.classifier.classifier = nn.Sequential()
model = model.cuda()

# ----------------------------------------------------------------------
# 提取特征并保存
# ----------------------------------------------------------------------
since = time.time()
print('Extracting gallery features...')
gallery_feature, gallery_label, _ = extract_feature(model, gallery_loader, view_type=1)
print('Extracting query features...')
query_feature, query_label, query_height = extract_feature(model, query_loader, view_type=2)
time_elapsed = time.time() - since
print(f'Feature extraction done in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')

gallery_path = [p for p, _ in gallery_dataset.samples]
query_path = [p for p, _ in query_dataset.samples]
result = {
    'gallery_f': gallery_feature.numpy(),
    'gallery_label': np.array(gallery_label),
    'gallery_path': gallery_path,
    'query_f': query_feature.numpy(),
    'query_label': np.array(query_label),
    'query_height': np.array(query_height),
    'query_path': query_path
}
mat_filename = f'pytorch_result_{opt.name}_with_height.mat'
scipy.io.savemat(mat_filename, result)
print(f'Result saved to {mat_filename}')