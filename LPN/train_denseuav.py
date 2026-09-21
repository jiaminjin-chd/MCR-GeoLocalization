# -*- coding: utf-8 -*-
"""
LPN training script for DenseUAV dataset with MCR masking.
Based on official LPN training logic (classification loss).
"""
from __future__ import print_function, division

import argparse
import os
import copy
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import numpy as np
import random

from model import two_view_net
from mask import RandomPatchMask   # 确保 mask.py 存在

# ----- 固定随机种子 -----
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----- 参数解析 -----
parser = argparse.ArgumentParser(description='Train LPN on DenseUAV with MCR')
parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/DenseUAV/train',
                    help='path to dataset root (contains satellite/ and drone/)')
parser.add_argument('--name', type=str, default='denseuav_lpn',
                    help='experiment name for saving models')
parser.add_argument('--batchsize', type=int, default=32, help='batch size')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, e.g. 0,1')
parser.add_argument('--h', type=int, default=256, help='image height')
parser.add_argument('--w', type=int, default=256, help='image width')
parser.add_argument('--stride', type=int, default=2, help='stride for ResNet')
parser.add_argument('--pool', type=str, default='avg', help='pooling type: avg|max|avg+max')
parser.add_argument('--drop_rate', type=float, default=0.5, help='dropout rate for classifier')

parser.add_argument('--LPN', action='store_true', default=True, help='use LPN')
parser.add_argument('--block', type=int, default=6, help='number of rings')

parser.add_argument('--lr', type=float, default=0.01, help='learning rate for classifier layers')
parser.add_argument('--warm_epoch', type=int, default=0, help='warm-up epochs')
parser.add_argument('--step_size', type=int, default=80, help='step size for LR decay')
parser.add_argument('--gamma', type=float, default=0.1, help='LR decay factor')
parser.add_argument('--epochs', type=int, default=120, help='total epochs')
parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume')

parser.add_argument('--share', action='store_true', default=True, help='share weights between branches')

# ----- MCR 参数 -----
parser.add_argument('--use_mcr', action='store_true', default=False,
                    help='apply MCR patch masking during training')
parser.add_argument('--mask_ratio', type=float, default=0.10,
                    help='peripheral masking ratio (0~1)')
parser.add_argument('--center_ratio', type=float, default=0.75,
                    help='center region retention ratio (side length fraction)')
parser.add_argument('--patch_size', type=int, default=16,
                    help='patch size for masking')
parser.add_argument('--mask_mode', type=str, default='single', choices=['single', 'dual'],
                    help='mask only drone branch or both drone and satellite')

opt = parser.parse_args()

# ----- GPU -----
str_ids = opt.gpu_ids.split(',')
gpu_ids = [int(id) for id in str_ids if int(id) >= 0]
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

# ----- Data transforms -----
transform_train = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_satellite = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----- Load datasets -----
print("Loading datasets...")
image_datasets = {
    'satellite': datasets.ImageFolder(os.path.join(opt.data_dir, 'satellite'), transform_satellite),
    'drone': datasets.ImageFolder(os.path.join(opt.data_dir, 'drone'), transform_train)
}
num_classes = len(image_datasets['satellite'].classes)
print(f"Number of classes: {num_classes}")

dataloaders = {
    'satellite': torch.utils.data.DataLoader(
        image_datasets['satellite'], batch_size=opt.batchsize,
        shuffle=True, num_workers=4, pin_memory=True),
    'drone': torch.utils.data.DataLoader(
        image_datasets['drone'], batch_size=opt.batchsize,
        shuffle=True, num_workers=4, pin_memory=True)
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'drone']}

# ----- Build model -----
print("Building model...")
model = two_view_net(
    class_num=num_classes,
    droprate=opt.drop_rate,
    stride=opt.stride,
    pool=opt.pool,
    share_weight=opt.share,
    LPN=opt.LPN,
    block=opt.block,
    VGG16=False,
    return_feat=False
)
model = model.cuda()

# ----- Optimizer -----
if opt.LPN:
    ignored_params = []
    for i in range(opt.block):
        cls_name = 'classifier' + str(i)
        c = getattr(model, cls_name)
        ignored_params += list(map(id, c.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optim_params = [{'params': base_params, 'lr': 0.1 * opt.lr}]
    for i in range(opt.block):
        cls_name = 'classifier' + str(i)
        c = getattr(model, cls_name)
        optim_params.append({'params': c.parameters(), 'lr': opt.lr})
    optimizer = optim.SGD(optim_params, weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    # 非 LPN 情况（保留）
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': model.classifier.parameters(), 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

# ----- Resume -----
start_epoch = 0
if opt.resume:
    if os.path.isfile(opt.resume):
        print(f"Loading checkpoint {opt.resume}")
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

# ----- Loss -----
criterion = nn.CrossEntropyLoss()

# ----- 初始化 MCR 掩码模块 -----
mask_module = None
if opt.use_mcr:
    mask_module = RandomPatchMask(
        mask_ratio=opt.mask_ratio,
        patch_size=opt.patch_size,
        fill_value=0,
        keep_center=True,
        center_ratio=opt.center_ratio
    )
    print(f"MCR enabled: mask_ratio={opt.mask_ratio}, center_ratio={opt.center_ratio}, mode={opt.mask_mode}")

# ----- Training loop -----
def lpn_loss(outputs, labels):
    loss = 0
    for part_out in outputs:
        loss += criterion(part_out, labels)
    return loss

def get_preds(outputs):
    prob = 0
    for part_out in outputs:
        prob += torch.softmax(part_out, dim=1)
    prob /= len(outputs)
    _, preds = torch.max(prob, 1)
    return preds

print("Starting training...")
os.makedirs(os.path.join('./model', opt.name), exist_ok=True)
with open(os.path.join('./model', opt.name, 'opts.yaml'), 'w') as f:
    yaml.dump(vars(opt), f, default_flow_style=False)

best_acc = 0.0
for epoch in range(start_epoch, opt.epochs):
    model.train()
    running_loss = 0.0
    running_corrects_sat = 0.0
    running_corrects_drone = 0.0

    sat_loader = dataloaders['satellite']
    drone_loader = dataloaders['drone']
    for (sat_inputs, sat_labels), (drone_inputs, drone_labels) in zip(sat_loader, drone_loader):
        if sat_inputs.size(0) < opt.batchsize:
            continue

        # ----- 应用 MCR 掩码（如果启用） -----
        if opt.use_mcr and mask_module is not None:
            # 对无人机分支应用掩码
            masked_drone = []
            for img in drone_inputs:
                masked_drone.append(mask_module(img))
            drone_inputs = torch.stack(masked_drone)
            # 如果 mask_mode == 'dual'，也对卫星分支应用掩码
            if opt.mask_mode == 'dual':
                masked_sat = []
                for img in sat_inputs:
                    masked_sat.append(mask_module(img))
                sat_inputs = torch.stack(masked_sat)

        sat_inputs = Variable(sat_inputs.cuda())
        drone_inputs = Variable(drone_inputs.cuda())
        sat_labels = Variable(sat_labels.cuda())
        drone_labels = Variable(drone_labels.cuda())

        outputs_sat, outputs_drone = model(sat_inputs, drone_inputs)

        loss = lpn_loss(outputs_sat, sat_labels) + lpn_loss(outputs_drone, drone_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sat_preds = get_preds(outputs_sat)
        drone_preds = get_preds(outputs_drone)

        running_loss += loss.item() * sat_inputs.size(0)
        running_corrects_sat += torch.sum(sat_preds == sat_labels.data).item()
        running_corrects_drone += torch.sum(drone_preds == drone_labels.data).item()

    epoch_loss = running_loss / dataset_sizes['satellite']
    epoch_acc_sat = running_corrects_sat / dataset_sizes['satellite']
    epoch_acc_drone = running_corrects_drone / dataset_sizes['drone']

    print(f'Epoch {epoch+1}/{opt.epochs} - Loss: {epoch_loss:.4f} '
          f'Sat Acc: {epoch_acc_sat:.4f} Drone Acc: {epoch_acc_drone:.4f}')
    scheduler.step()

    if (epoch + 1) % 10 == 0 or (epoch + 1) == opt.epochs:
        model_path = os.path.join('./model', opt.name, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'acc_sat': epoch_acc_sat,
            'acc_drone': epoch_acc_drone
        }, model_path)
        print(f'Checkpoint saved to {model_path}')

    if epoch_acc_sat > best_acc:
        best_acc = epoch_acc_sat
        best_path = os.path.join('./model', opt.name, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'acc_sat': epoch_acc_sat,
            'acc_drone': epoch_acc_drone
        }, best_path)
        print(f'Best model updated (Sat Acc: {best_acc:.4f})')

print("Training finished!")