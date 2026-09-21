import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.nn as nn
def train(train_config,model,dataloader,loss_functions,optimizer,epoch,train_steps_per,tensorboard=None,scheduler=None,scaler=None):
    model.train()
    losses=AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    bar=tqdm(dataloader) if train_config.verbose else dataloader
    criterion=nn.CrossEntropyLoss()
    for step,(query,reference,ids,labels) in enumerate(bar):
        query=query.to(train_config.device)
        reference=reference.to(train_config.device)
        labels=labels.to(train_config.device)
        with autocast():
            output1,output2=model(query,reference)
            features1=output1[0]
            features2=output2[0]
            logits1=output1[1]
            logits2=output2[1]
            fmap1=output1[2]
            fmap2=output2[2]
            if hasattr(model,"module"):
                scale=model.module.logit_scale.exp()
            else:
                scale=model.logit_scale.exp()
            loss_infonce=loss_functions["infoNCE"](features1,features2,scale)
            loss_cls=criterion(logits1,labels)+criterion(logits2,labels)
            if train_config.weight_dsa>0:
                fmap1=fmap1.flatten(2)
                fmap2=fmap2.flatten(2)
                loss_dsa=loss_functions["DSA_loss"](fmap1,fmap2,scale)
            else:
                loss_dsa=torch.tensor(0.0,device=train_config.device)
            loss=train_config.weight_infonce*loss_infonce+train_config.weight_cls*loss_cls+train_config.weight_dsa*loss_dsa
        scaler.scale(loss).backward() if scaler else loss.backward()
        if train_config.clip_grad:
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),train_config.clip_grad)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
        losses.update(loss.item())
        if train_config.verbose:
            bar.set_description(
                "Epoch {} Loss {:.4f}".format(epoch,losses.avg)
            )
    return losses.avg
def predict(train_config, model, dataloader):

    model.eval()

    time.sleep(0.1)

    features_list = []
    ids_list = []

    if train_config.verbose:
        bar = tqdm(dataloader)
    else:
        bar = dataloader

    with torch.no_grad():

        for imgs, ids in bar:

            imgs = imgs.to(train_config.device)

            with autocast():

                output = model(imgs)

                if isinstance(output, tuple):

                    # DenseUAV v2 model:
                    # feat_vec, logits, feat_map

                    img_feature = output[0]

                else:

                    img_feature = output


            # 保证是二维
            if img_feature.dim() == 1:
                img_feature = img_feature.unsqueeze(0)


            img_feature = F.normalize(
                img_feature,
                dim=-1
            )


            features_list.append(
                img_feature.cpu()
            )

            if isinstance(ids, torch.Tensor):
                ids_list.append(ids.cpu())
            else:
                ids_list.extend(list(ids))


    features = torch.cat(
        features_list,
        dim=0
    )

    if len(ids_list) > 0 and isinstance(ids_list[0], torch.Tensor):
        ids = torch.cat(ids_list, dim=0)
    else:
        ids = ids_list


    return features.to(train_config.device), ids


# import time
# import torch
# from tqdm import tqdm
# from .utils import AverageMeter
# from torch.cuda.amp import autocast
# import torch.nn.functional as F
# from sample4geo.loss.cal_loss import cal_kl_loss, cal_loss, cal_triplet_loss
# import torch.nn as nn
#
#
# def train(train_config, model, dataloader, loss_functions, optimizer, epoch, train_steps_per, tensorboard=None,
#           scheduler=None, scaler=None):
#     # set model train mode
#     model.train()
#
#     losses = AverageMeter()
#
#     # wait before starting progress bar
#     time.sleep(0.1)
#
#     # Zero gradients for first step
#     optimizer.zero_grad(set_to_none=True)
#
#     step = 1
#
#     if train_config.verbose:
#         bar = tqdm(dataloader, total=len(dataloader))
#     else:
#         bar = dataloader
#
#     criterion = nn.CrossEntropyLoss()
#
#     # for loop over one epoch
#     for query, reference, ids, labels in bar:
#
#         if scaler:
#             with (autocast()):  # -- 使用混合精度
#                 # data (batches) to device
#                 query = query.to(train_config.device)
#                 reference = reference.to(train_config.device)
#                 labels = labels.to(train_config.device)
#
#                 # Forward pass
#                 if train_config.handcraft_model is not True:
#                     features1, features2 = model(query, reference)
#                 else:
#                     output1, output2 = model(query, reference)
#
#                     # 添加调试打印（只打印第一个batch）
#                     if step == 1:
#                         print(f"output1[0] shape: {output1[0].shape}")  # 应为 (B, 1024)
#                         print(f"output1[1] shape: {output1[1].shape}")  # 应为 (B, 2256)
#                         print(f"output1[2] shape: {output1[2].shape}")  # 应为 (B, 1024, H, W)
#
#                     # 正确索引
#                     features1, features2 = output1[0], output2[0]  # feat_vec -> InfoNCE
#                     features_cls_1, features_cls_2 = output1[1], output2[1]  # logits -> 分类损失
#                     features_dsa_1, features_dsa_2 = output1[2], output2[2]  # feat_map -> DSA
#
#                 if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
#                     loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
#                 else:
#                     # 1. infoNCE
#                     loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())
#
#                     # 2. Classification
#                     try:
#                         loss_cls = criterion(features_cls_1, labels) + criterion(features_cls_2, labels)
#                     except (UnboundLocalError, NameError):
#                         loss_cls = torch.tensor(0.0, device=features1.device)
#
#
#                     # 3. Domian Space Alignment Loss
#                     if train_config.weight_dsa > 0:
#                         features_dsa_1 = features_dsa_1.flatten(2)  # (B, C, H*W)
#                         features_dsa_2 = features_dsa_2.flatten(2)
#                         loss_DSA = loss_functions["DSA_loss"](features_dsa_1, features_dsa_2, labels)
#                     else:
#                         loss_DSA = torch.tensor(0.0, device=features1.device)
#
#
#                 lossall = train_config.weight_infonce * loss + train_config.weight_cls * loss_cls + train_config.weight_dsa * loss_DSA
#
#                 # lossall = 1.0 * loss + 0.0 * loss_cls + 0.0 * loss_DSA
#
#                 losses.update(lossall.item())
#
#             # scaler.scale(loss).backward()  # -- 混合精度好像是这样用的
#             scaler.scale(lossall).backward()  # -- 这里才是反向传播，上面就是记录一下
#
#             # Gradient clipping
#             if train_config.clip_grad:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)
#
#                 # Update model parameters (weights)
#             scaler.step(optimizer)
#             scaler.update()
#
#             # Zero gradients for next step
#             optimizer.zero_grad()
#
#             # Scheduler
#             if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
#                 scheduler.step()
#
#         else:
#
#             # data (batches) to device
#             query = query.to(train_config.device)
#             reference = reference.to(train_config.device)
#
#             # Forward pass
#             features1, features2 = model(query, reference)
#             if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
#                 loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
#             else:
#                 loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())
#             losses.update(loss.item())
#
#             # Calculate gradient using backward pass
#             loss.backward()
#
#             # Gradient clipping
#             if train_config.clip_grad:
#                 torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)
#
#                 # Update model parameters (weights)
#             optimizer.step()
#             # Zero gradients for next step
#             optimizer.zero_grad()
#
#             # Scheduler
#             if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
#                 scheduler.step()
#
#         if train_config.verbose:
#             # tst = model.logit_scale
#             monitor = {
#                 "loss": "{:.4f}".format(loss.item()),
#                 "loss_cls": "{:.4f}".format(train_config.weight_cls * loss_cls.item()),
#                 "loss_dsa": "{:.4f}".format(train_config.weight_dsa * loss_DSA.item()),
#                 "loss_avg": "{:.4f}".format(losses.avg),
#                 "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}
#
#             bar.set_postfix(ordered_dict=monitor)
#
#             if tensorboard is not None:
#                 steps = step + (epoch - 1) * train_steps_per
#                 tensorboard.add_scalar("Loss", lossall.item(), steps)
#                 tensorboard.add_scalar("Loss_Avg", losses.avg, steps)
#                 tensorboard.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], steps)
#                 tensorboard.add_scalar("Learning_Rate_Temp", optimizer.param_groups[-1]['lr'], steps)
#                 tensorboard.add_scalar("Temperature", model.logit_scale.detach().cpu().numpy(), steps)
#
#         step += 1
#
#     if train_config.verbose:
#         bar.close()
#
#     return losses.avg
#
#
# def predict(train_config, model, dataloader):
#     model.eval()
#
#     # wait before starting progress bar
#     time.sleep(0.1)
#
#     # -- draw visualization for ablation study
#     draw_vis = False
#     if draw_vis:
#         import cv2
#         import os
#         import numpy as np
#         pic_path = "./draw_vis"
#         iterations = int(len(os.listdir(pic_path)) / 2)
#         for i in range(iterations):
#             uav_ori = cv2.imread(rf"{pic_path}/{i}_uav.jpg")
#             sat_ori = cv2.imread(rf"{pic_path}/{i}_sat.jpg")
#
#             uav_shape = uav_ori.shape[:-1]
#             sat_shape = sat_ori.shape[:-1]
#
#             uav = cv2.resize(uav_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
#             sat = cv2.resize(sat_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
#
#             uav = torch.tensor(uav).permute(2, 0, 1)
#             sat = torch.tensor(sat).permute(2, 0, 1)
#
#             # 图像标准化
#             mean = [0.485, 0.456, 0.406]
#             std = [0.229, 0.224, 0.225]
#             import torchvision.transforms as transforms
#             normalize = transforms.Normalize(mean=mean, std=std)
#             uav = normalize(uav)[None, :, :, :]
#             sat = normalize(sat)[None, :, :, :]
#
#             with torch.no_grad():
#                 with autocast():
#                     uav = uav.to(train_config.device)
#                     sat = sat.to(train_config.device)
#
#                     img_feature_uav = model(uav)[1]
#                     img_feature_uav = F.normalize(img_feature_uav, dim=1)
#
#                     img_feature_sat = model(sat)[1]
#                     img_feature_sat = F.normalize(img_feature_sat, dim=1)
#
#                     heat_map_uav = img_feature_uav[0].permute(1, 2, 0)
#                     heat_map_uav = torch.mean(heat_map_uav, dim=2).detach().cpu().numpy()
#                     heat_map_uav = (heat_map_uav - heat_map_uav.min()) / (heat_map_uav.max() - heat_map_uav.min())
#                     heat_map_uav = cv2.resize(heat_map_uav, [uav_shape[1], uav_shape[0]])
#
#                     heat_map_sat = img_feature_sat[0].permute(1, 2, 0)
#                     heat_map_sat = torch.mean(heat_map_sat, dim=2).detach().cpu().numpy()
#                     heat_map_sat = (heat_map_sat - heat_map_sat.min()) / (heat_map_sat.max() - heat_map_sat.min())
#                     heat_map_sat = cv2.resize(heat_map_sat, [sat_shape[1], sat_shape[0]])
#
#                     #  colorize
#                     colored_image_uav = cv2.applyColorMap((heat_map_uav * 255).astype(np.uint8), cv2.COLORMAP_JET)
#                     colored_image_sat = cv2.applyColorMap((heat_map_sat * 255).astype(np.uint8), cv2.COLORMAP_JET)
#
#                     # 设置半透明度（alpha值）
#                     alpha = 0.5
#                     # 将两个图像进行叠加
#                     blended_image_uav = cv2.addWeighted(uav_ori, alpha, colored_image_uav, 1 - alpha, 0)
#                     blended_image_sat = cv2.addWeighted(sat_ori, alpha, colored_image_sat, 1 - alpha, 0)
#
#
#                     out_path = r"/media/xiapanwang/主数据盘/xiapanwang/Codes/python/Cross-View_Geo-Localization/DAC/DSA_off"
#                     cv2.imwrite(rf"{out_path}/{i}_uav_vis.jpg", blended_image_uav)
#                     cv2.imwrite(rf"{out_path}/{i}_sat_vis.jpg", blended_image_sat)
#
#         return 0
#
#     if train_config.verbose:
#         bar = tqdm(dataloader, total=len(dataloader))
#     else:
#         bar = dataloader
#
#     img_features_list = []
#
#     ids_list = []
#     with torch.no_grad():
#
#         for img, ids in bar:
#
#             ids_list.append(ids)
#
#             with autocast():
#                 img = img.to(train_config.device)
#
#                 if train_config.handcraft_model is not True:
#                     img_feature = model(img)
#                 else:
#                     img_feature = model(img)[-2]
#
#                 # normalize is calculated in fp32
#                 if train_config.normalize_features:
#                     img_feature = F.normalize(img_feature, dim=-1)
#
#             # save features in fp32 for sim calculation
#             img_features_list.append(img_feature.to(torch.float32))
#
#         # keep Features on GPU
#         img_features = torch.cat(img_features_list, dim=0)
#         ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
#
#     if train_config.verbose:
#         bar.close()
#
#     return img_features, ids_list
