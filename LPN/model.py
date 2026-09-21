import argparse
import math
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

class ft_net_VGG16(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_VGG16, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)   # 保留 VGG16
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))

        #x = self.classifier(x)
        return x

# Define the VGG16-based part Model
class ft_net_VGG16_LPN(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=8, row = False):
        super(ft_net_VGG16_LPN, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        self.avgpool = nn.AdaptiveAvgPool2d((1,block))
        self.maxpool = nn.AdaptiveMaxPool2d((1,block))
        if row:  # row partition the ground view image
            self.avgpool = nn.AdaptiveAvgPool2d((block,1))
            self.maxpool = nn.AdaptiveMaxPool2d((block,1))
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.maxpool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x

# Define vgg16 based square ring partition for satellite images of cvusa/cvact
class ft_net_VGG16_LPN_R(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=4):
        super(ft_net_VGG16_LPN_R, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x
    # VGGNet's output: 8*8 part:4*4, 6*6, 8*8
    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear')
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.insert(0, avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.insert(0, avgpool)
        return torch.cat(result, dim=2)

# resnet50 backbone
class ft_net_cvusa_LPN(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=6, row=False):
        super(ft_net_cvusa_LPN, self).__init__()
        model_ft = models.resnet50(weights='IMAGENET1K_V1')
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        self.avgpool = nn.AdaptiveAvgPool2d((1,block))
        self.maxpool = nn.AdaptiveMaxPool2d((1,block))
        if row:
            self.avgpool = nn.AdaptiveAvgPool2d((block,1))
            self.maxpool = nn.AdaptiveMaxPool2d((block,1))
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.maxpool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x

class ft_net_cvusa_LPN_R(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=6):
        super(ft_net_cvusa_LPN_R, self).__init__()
        model_ft = models.resnet50(weights='IMAGENET1K_V1')
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear')
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.insert(0, avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.insert(0, avgpool)
        return torch.cat(result, dim=2)

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(weights='IMAGENET1K_V1')
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))
        #x = self.classifier(x)
        return x

# Define the ResNet50-based part Model
class ft_net_LPN(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=6):
        super(ft_net_LPN, self).__init__()
        model_ft = models.resnet50(weights='IMAGENET1K_V1')
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.shape)
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)

class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=1, pool='avg', share_weight=False, VGG16=False, LPN=False, block=2, return_feat=False):
        super(two_view_net, self).__init__()
        self.LPN = LPN
        self.block = block
        self.return_feat = return_feat
        self.sqr = True  # 保持原有逻辑
        # 以下代码不变（初始化 backbone 和 classifier）
        if VGG16:
            if LPN:
                self.model_1 = ft_net_VGG16_LPN(class_num, stride=stride, pool=pool, block=block)
                if self.sqr:
                    self.model_1 = ft_net_VGG16_LPN_R(class_num, stride=stride, pool=pool, block=block)
            else:
                self.model_1 = ft_net_VGG16(class_num, stride=stride, pool=pool)
        else:
            self.model_1 = ft_net_cvusa_LPN(class_num, stride=stride, pool=pool, block=block)
            if self.sqr:
                self.model_1 = ft_net_cvusa_LPN_R(class_num, stride=stride, pool=pool, block=block)
            self.block = self.model_1.block
        # 是否共享权重
        if share_weight:
            self.model_2 = self.model_1
        else:
            if VGG16:
                if LPN:
                    self.model_2 = ft_net_VGG16_LPN(class_num, stride=stride, pool=pool, block=block, row=self.sqr)
                else:
                    self.model_2 = ft_net_VGG16(class_num, stride=stride, pool=pool)
            else:
                self.model_2 = ft_net_cvusa_LPN(class_num, stride=stride, pool=pool, block=block, row=self.sqr)
        # 分类器
        if LPN:
            if VGG16:
                if pool == 'avg+max':
                    feat_dim = 1024
                else:
                    feat_dim = 512
            else:
                if pool == 'avg+max':
                    feat_dim = 4096
                else:
                    feat_dim = 2048
            for i in range(self.block):
                name = 'classifier' + str(i)
                setattr(self, name, ClassBlock(feat_dim, class_num, droprate))
        else:
            self.classifier = ClassBlock(2048, class_num, droprate)
            if pool == 'avg+max':
                self.classifier = ClassBlock(4096, class_num, droprate)
            if VGG16:
                self.classifier = ClassBlock(512, class_num, droprate)
                if pool == 'avg+max':
                    self.classifier = ClassBlock(1024, class_num, droprate)

    def forward(self, x1, x2):
        if self.LPN:
            # ---- 处理第一个视角 ----
            if x1 is None:
                y1 = None
                feat1 = None
            else:
                # model_1 输出形状: (batch, 2048, block)
                x1 = self.model_1(x1)
                # 在 block 维度上取平均，获得全局特征 (batch, 2048)
                feat1 = x1.mean(dim=2) if x1.dim() == 3 else x1  # 防止意外维度
                # 分类输出仍使用分块分类器
                y1 = self.part_classifier(x1)

            # ---- 处理第二个视角 ----
            if x2 is None:
                y2 = None
                feat2 = None
            else:
                x2 = self.model_2(x2)
                feat2 = x2.mean(dim=2) if x2.dim() == 3 else x2
                y2 = self.part_classifier(x2)

            # 根据 return_feat 决定返回值
            if self.return_feat:
                return y1, y2, feat1, feat2
            else:
                return y1, y2

        else:
            # ---- 非 LPN 模式（保持原样） ----
            if x1 is None:
                y1 = None
            else:
                x1 = self.model_1(x1)
                y1 = self.classifier(x1)
            if x2 is None:
                y2 = None
            else:
                x2 = self.model_2(x2)
                y2 = self.classifier(x2)
            return y1, y2

    def part_classifier(self, x):
        part = {}
        predict = {}
        scale = 200.0  # 提高至 200
        for i in range(self.block):
            part[i] = x[:, :, i].view(x.size(0), -1) * scale
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y

class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False, LPN=False, block=6):
        super(three_view_net, self).__init__()
        self.LPN = LPN
        self.block = block
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
        elif LPN:
            self.model_1 =  ft_net_LPN(class_num, stride = stride, pool = pool, block = block)
            self.model_2 =  ft_net_LPN(class_num, stride = stride, pool = pool, block = block)
            # self.block = self.model_1.block
        else: 
            self.model_1 =  ft_net(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

        if share_weight:
            self.model_3 = self.model_1
        else:
            if VGG16:
                self.model_3 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            elif LPN:
                self.model_3 =  ft_net_LPN(class_num, stride = stride, pool = pool, block = block)
            else:
                self.model_3 =  ft_net(class_num, stride = stride, pool = pool)
        if LPN:
            if pool == 'avg+max':
                for i in range(self.block):
                    name = 'classifier'+str(i)
                    setattr(self, name, ClassBlock(4096, class_num, droprate))
            else:
                for i in range(self.block):
                    name = 'classifier'+str(i)
                    setattr(self, name, ClassBlock(2048, class_num, droprate))
        else:    
            self.classifier = ClassBlock(2048, class_num, droprate)
            if pool =='avg+max':
                self.classifier = ClassBlock(4096, class_num, droprate)

    def forward(self, x1, x2, x3, x4 = None): # x4 is extra data
        if self.LPN:
            if x1 is None:
                y1 = None
            else:
                x1 = self.model_1(x1)
                y1 = self.part_classifier(x1)

            if x2 is None:
                y2 = None
            else:
                x2 = self.model_2(x2)
                y2 = self.part_classifier(x2)

            if x3 is None:
                y3 = None
            else:
                x3 = self.model_3(x3)
                y3 = self.part_classifier(x3)

            if x4 is None:
                return y1, y2, y3
            else:
                x4 = self.model_2(x4)
                y4 = self.part_classifier(x4)
                return y1, y2, y3, y4
        else:
            if x1 is None:
                y1 = None
            else:
                x1 = self.model_1(x1)
                y1 = self.classifier(x1)

            if x2 is None:
                y2 = None
            else:
                x2 = self.model_2(x2)
                y2 = self.classifier(x2)

            if x3 is None:
                y3 = None
            else:
                x3 = self.model_3(x3)
                y3 = self.classifier(x3)

            if x4 is None:
                return y1, y2, y3
            else:
                x4 = self.model_2(x4)
                y4 = self.classifier(x4)
                return y1, y2, y3, y4

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            part[i] = x[:,:,i].view(x.size(0),-1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = two_view_net(701, droprate=0.5, pool='avg', stride=1, VGG16=False, LPN=True, block=8)

    # net = three_view_net(701, droprate=0.5, pool='avg', stride=1, share_weight=True, LPN=True, block=2)
    # net.eval()

    # net = ft_net_VGG16_LPN_R(701)
    # net = ft_net_cvusa_LPN(701, stride=1)
    # net = ft_net(701)

    print(net)

    input = Variable(torch.FloatTensor(2, 3, 256, 256))
    output1,output2 = net(input,input)
    # output1,output2,output3 = net(input,input,input)
    # output1 = net(input)
    # print('net output size:')
    # print(output1.shape)
    # print(output.shape)
    for i in range(len(output1)):
        print(output1[i].shape)
    # x = torch.randn(2,512,8,8)
    # x_shape = x.shape
    # pool = AzimuthPool2d(x_shape, 8)
    # out = pool(x)
    # print(out.shape)
