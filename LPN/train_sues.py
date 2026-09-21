from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.backends.cudnn as cudnn
import matplotlib
import numpy as np
import random
import os
import time
import copy
import yaml
from shutil import copyfile

from mask import RandomPatchMask

matplotlib.use('agg')
import matplotlib.pyplot as plt

from model import two_view_net
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy
from utils import update_average, load_network, save_network
from dataset_sues import SUESDataset   # 需要自己创建的 dataset_sues.py

# ================= 固定随机种子 =================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ================================================

# fp16 支持
try:
    from apex import amp
except ImportError:
    print('Apex not installed, fp16 disabled')
    amp = None

version = torch.__version__

# ----------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Training on SUES-200')
parser.add_argument('--use_mask', action='store_true', help='apply patch mask on both views')
parser.add_argument('--mask_ratio', default=0.45, type=float, help='mask ratio for non-center area')
parser.add_argument('--patch_size', default=16, type=int, help='patch size for mask')
parser.add_argument('--keep_center', action='store_true', help='keep center area unmasked')
parser.add_argument('--center_ratio', default=0.75, type=float, help='center area ratio')

parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--name', default='sues_two_view', type=str, help='output model name')
parser.add_argument('--pool', default='avg', type=str)
parser.add_argument('--data_dir', default='/root/autodl-tmp/SUES-200-512x512', type=str)
parser.add_argument('--train_all', action='store_true', help='use all IDs for training (no val split)')
parser.add_argument('--color_jitter', action='store_true')
parser.add_argument('--batchsize', default=8, type=int)
parser.add_argument('--stride', default=2, type=int)
parser.add_argument('--pad', default=10, type=int)
parser.add_argument('--h', default=384, type=int)
parser.add_argument('--w', default=384, type=int)
parser.add_argument('--views', default=2, type=int, help='must be 2 for SUES')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability')
parser.add_argument('--use_dense', action='store_true')
parser.add_argument('--use_NAS', action='store_true')
parser.add_argument('--warm_epoch', default=0, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--moving_avg', default=1.0, type=float)
parser.add_argument('--droprate', default=0.5, type=float)
parser.add_argument('--DA', action='store_true', help='AutoAugment')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--share', action='store_true', help='share weights between views')
parser.add_argument('--extra_Google', action='store_true', help='not used, kept for compatibility')
parser.add_argument('--LPN', action='store_true', help='use LPN')
parser.add_argument('--block', default=6, type=int, help='number of blocks in LPN')
parser.add_argument('--fp16', action='store_true')
opt = parser.parse_args()
print("================== Hyperparameters ==================")
for arg in vars(opt):
    print(f"{arg}: {getattr(opt, arg)}")
print("=====================================================")

if opt.views != 2:
    print('SUES-200 only uses two views (satellite and drone). Setting --views=2')
    opt.views = 2

# GPU设置
str_ids = opt.gpu_ids.split(',')
gpu_ids = [int(id) for id in str_ids if int(id) >= 0]
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0

# ----------------------------------------------------------------------
# 数据变换
# ----------------------------------------------------------------------
# 卫星视图变换（通常不旋转，加掩码）
transform_satellite_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
]
# 无人机视图变换（加随机旋转）
transform_drone_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomAffine(90),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
]

# 添加 ToTensor
transform_satellite_list.append(transforms.ToTensor())
transform_drone_list.append(transforms.ToTensor())

if opt.use_mask:
    mask = RandomPatchMask(
        mask_ratio=opt.mask_ratio,
        patch_size=opt.patch_size,
        fill_value=0,
        keep_center=opt.keep_center,
        center_ratio=opt.center_ratio
    )
    # 对两个视图都加掩码（双视图掩码）
    # transform_satellite_list.append(mask)
    transform_drone_list.append(mask)

# 添加 Normalize
transform_satellite_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
transform_drone_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

if opt.erasing_p > 0:
    transform_satellite_list.append(RandomErasing(probability=opt.erasing_p))
    transform_drone_list.append(RandomErasing(probability=opt.erasing_p))

if opt.color_jitter:
    transform_satellite_list.insert(0, transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
    transform_drone_list.insert(0, transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))

if opt.DA:
    transform_satellite_list.insert(0, ImageNetPolicy())
    transform_drone_list.insert(0, ImageNetPolicy())

transform_satellite = transforms.Compose(transform_satellite_list)
transform_drone = transforms.Compose(transform_drone_list)

# 验证变换（无增强）
transform_val = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------------------------------------------------------
# 数据集划分（按ID 80%/20%）
# ----------------------------------------------------------------------
sat_dir = os.path.join(opt.data_dir, 'satellite-view')
all_ids = [d for d in os.listdir(sat_dir) if os.path.isdir(os.path.join(sat_dir, d))]
all_ids = sorted(all_ids)

if opt.train_all:
    train_ids = all_ids
    val_ids = []
else:
    split = int(0.8 * len(all_ids))
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

print(f'Training IDs: {len(train_ids)}, Validation IDs: {len(val_ids)}')

# 在构建数据集之前，先基于所有 ID 建立统一的标签映射
all_ids_sorted = sorted(all_ids)  # 所有 ID 的排序列表
global_label_map = {id_: idx for idx, id_ in enumerate(all_ids_sorted)}

# 训练数据集：使用 global_label_map 中的标签，但只包含 train_ids
train_dataset = SUESDataset(
    root_dir=opt.data_dir,
    transform_sat=transform_satellite,
    transform_drone=transform_drone,
    train_ids=train_ids,
    global_label_map=global_label_map  # 需要修改 dataset_sues.py 支持传入外部映射
)

# 验证数据集同理，使用相同的 global_label_map
val_dataset = SUESDataset(
    root_dir=opt.data_dir,
    transform_sat=transform_val,
    transform_drone=transform_val,
    train_ids=val_ids,
    global_label_map=global_label_map
)if val_ids else None

dataloader_train = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchsize, shuffle=True,
    num_workers=4, pin_memory=True
)
dataloader_val = torch.utils.data.DataLoader(
    val_dataset, batch_size=opt.batchsize, shuffle=False,
    num_workers=4, pin_memory=True
) if val_dataset else None

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset) if val_dataset else 0}
class_names = train_ids
print(f'Dataset sizes: {dataset_sizes}')

# ----------------------------------------------------------------------
# 训练函数
# ----------------------------------------------------------------------
y_loss = {'train': [], 'val': []}
y_err = {'train': [], 'val': []}

def one_LPN_output(outputs, labels, criterion, block):
    # 如果 outputs 是张量，转换为列表
    if isinstance(outputs, torch.Tensor):
        # 可能形状为 [batch, class, block] 或 [batch*block, class] 或 [block, batch, class]
        if outputs.dim() == 3:
            # 假设形状 [batch, class, block] -> 拆成 block 个 [batch, class]
            outputs = [outputs[:, :, i] for i in range(outputs.size(2))]
        elif outputs.dim() == 2:
            # 假设形状 [batch*block, class] -> 先 reshape 再拆分
            batch_size = labels.size(0)
            # 推断 block = outputs.size(0) // batch_size
            inferred_block = outputs.size(0) // batch_size
            outputs = outputs.view(batch_size, inferred_block, -1)  # [batch, block, class]
            outputs = [outputs[:, i, :] for i in range(inferred_block)]
        else:
            raise ValueError(f"Unsupported outputs dimension: {outputs.dim()}")
    # 此时 outputs 应为 list，长度至少为 block
    sm = nn.Softmax(dim=1)
    score = 0
    loss = 0
    for i in range(block):
        part = outputs[i]  # 期望 [batch, class]
        # 额外的安全检查
        if part.dim() == 1:
            part = part.unsqueeze(0)  # [class] -> [1, class]
        score += sm(part)
        loss += criterion(part, labels)
    _, preds = torch.max(score.data, 1)
    return preds, loss
def train_model(model, model_test, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    warm_up = 0.1
    warm_iteration = round(dataset_sizes['train'] / opt.batchsize) * opt.warm_epoch

    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
                dataloader = dataloader_train
            else:
                if dataloader_val is None:
                    continue
                model.train(False)
                dataloader = dataloader_val

            running_loss = 0.0
            running_corrects_sat = 0.0
            running_corrects_drone = 0.0

            for sat_img, drone_img, labels in dataloader:
                now_batch_size = sat_img.size(0)
                if now_batch_size < opt.batchsize:
                    continue
                if torch.cuda.is_available():
                    sat_img = sat_img.cuda()
                    drone_img = drone_img.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs_sat, outputs_drone = model(sat_img, drone_img)

                    if not opt.LPN:
                        _, preds_sat = torch.max(outputs_sat.data, 1)
                        _, preds_drone = torch.max(outputs_drone.data, 1)
                        loss = criterion(outputs_sat, labels) + criterion(outputs_drone, labels)
                    else:
                        preds_sat, loss_sat = one_LPN_output(outputs_sat, labels, criterion, opt.block)
                        preds_drone, loss_drone = one_LPN_output(outputs_drone, labels, criterion, opt.block)
                        loss = loss_sat + loss_drone

                    if phase == 'train':
                        if epoch < opt.warm_epoch:
                            warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                            loss *= warm_up
                        if opt.fp16 and amp is not None:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        optimizer.step()
                        if opt.moving_avg < 1.0:
                            update_average(model_test, model, opt.moving_avg)

                running_loss += loss.item() * now_batch_size
                running_corrects_sat += (preds_sat == labels).float().sum().item()
                running_corrects_drone += (preds_drone == labels).float().sum().item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc_sat = running_corrects_sat / dataset_sizes[phase]
            epoch_acc_drone = running_corrects_drone / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}  Sat_Acc: {epoch_acc_sat:.4f}  Drone_Acc: {epoch_acc_drone:.4f}')

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc_sat)

            if phase == 'train':
                scheduler.step()

        # 每40个epoch保存一次（节省空间，可改为20）
        if epoch % 40 == 39:
            save_network(model, opt.name, epoch)

        time_elapsed = time.time() - since
        print(f'Time {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s\n')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    return model

# ----------------------------------------------------------------------
# 模型初始化
# ----------------------------------------------------------------------
num_classes = len(global_label_map)
if opt.LPN:
    model = two_view_net(num_classes, droprate=opt.droprate, stride=opt.stride,
                         pool=opt.pool, share_weight=opt.share, LPN=True, block=opt.block)
else:
    model = two_view_net(num_classes, droprate=opt.droprate, stride=opt.stride,
                         pool=opt.pool, share_weight=opt.share)

print(model)

# 优化器
if not opt.LPN:
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': model.classifier.parameters(), 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
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
    optimizer_ft = optim.SGD(optim_params, weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)

dir_name = os.path.join('./model', opt.name)
if not opt.resume:
    os.makedirs(dir_name, exist_ok=True)  # 自动创建父目录
    copyfile('./train_sues.py', dir_name + '/train_sues.py')
    copyfile('./model.py', dir_name + '/model.py')
    with open(f'{dir_name}/opts.yaml', 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

model = model.cuda()
criterion = nn.CrossEntropyLoss()

if opt.fp16 and amp is not None:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

if opt.moving_avg < 1.0:
    model_test = copy.deepcopy(model)
    num_epochs = 140
else:
    model_test = None
    num_epochs = 120

model = train_model(model, model_test, criterion, optimizer_ft, exp_lr_scheduler, num_epochs)