import os
import torch
import torch.nn as nn
import timm
import numpy as np

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None
    print("Warning: safetensors not installed. Install with: pip install safetensors")


class ConvNeXtWithHead(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone          # timm 模型，num_classes=0
        self.num_features = backbone.num_features
        self.classifier = nn.Linear(self.num_features, num_classes)

        # 特征图输出（用于 DSA）
        self.feat_h = backbone.num_features  # 特征维度

    def forward(self, x, return_f=False):
        # 获取空间特征图（未池化）
        # timm 模型可以通过 features_only=True 获取中间层，但为了简便，我们直接修改 forward 输出
        # 这里使用 backbone.forward_features(x) 得到未池化的特征图，然后全局池化用于分类
        # 但 timm 模型可能没有直接的 forward_features，我们使用默认 forward 并调整
        # 更可靠：重新创建模型时设置 global_pool='' 并手动池化
        x = self.backbone.forward_features(x)   # 输出 (B, C, H, W) 特征图
        # 全局平均池化得到特征向量
        feat_vec = x.mean([-2, -1])              # (B, C)
        logits = self.classifier(feat_vec)
        if return_f:
            return feat_vec, logits, x          # 返回全局特征、logits、特征图
        else:
            return logits, x                    # 兼容不同调用，但为了统一，我们尽量返回元组


class two_view_net(nn.Module):
    def __init__(self, class_num, block=4, return_f=False, resnet=False):
        super(two_view_net, self).__init__()
        # 创建 timm 模型，并设置 global_pool='' 以保留空间维度
        backbone = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', pretrained=False, num_classes=0, global_pool='')
        self.model_1 = ConvNeXtWithHead(backbone, class_num)

        # DAC 额外参数
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_blocks = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.w_blocks1 = torch.nn.Parameter(torch.ones([]))
        self.w_blocks2 = torch.nn.Parameter(torch.ones([]))
        self.w_blocks3 = torch.nn.Parameter(torch.ones([]))

        self.return_f = return_f   # 用于指示是否返回特征图

    def get_config(self):
        return timm.data.resolve_model_data_config(self.model_1.backbone)

    def forward(self, x1, x2=None):
        if x2 is not None:
            # 每个输出为 (feat_vec, logits, feat_map)
            out1 = self.model_1(x1, return_f=True)
            out2 = self.model_1(x2, return_f=True)
            # 提取各分量
            feat_vec1, logits1, feat_map1 = out1
            feat_vec2, logits2, feat_map2 = out2
            # 返回 (feat_vec, logits, feat_map) 三元组
            return (feat_vec1, logits1, feat_map1), (feat_vec2, logits2, feat_map2)
        else:
            feat_vec, logits, feat_map = self.model_1(x1, return_f=True)
            return feat_vec, logits, feat_map


class three_view_net(nn.Module):
    pass


def make_model(opt):
    if opt.views == 2:
        model = two_view_net(opt.nclasses, block=opt.block, return_f=opt.triplet_loss, resnet=opt.resnet)

    # 加载本地预训练权重
    weight_path = "/root/autodl-tmp/DAC-main/model.safetensors"
    if os.path.exists(weight_path):
        if load_file is None:
            raise ImportError("safetensors not installed. Please run: pip install safetensors")
        print(f"Loading pretrained weights from {weight_path}")
        state_dict = load_file(weight_path)

        # 获取 backbone 的 state_dict
        backbone_state = model.model_1.backbone.state_dict()

        # 筛选匹配的键
        matched_keys = []
        for k in state_dict.keys():
            if k in backbone_state:
                matched_keys.append(k)

        print(f"Matched keys: {len(matched_keys)} / {len(state_dict)}")

        # 加载匹配的键
        matched_dict = {k: state_dict[k] for k in matched_keys}
        model.model_1.backbone.load_state_dict(matched_dict, strict=True)

        print(f"Successfully loaded {len(matched_keys)} keys into backbone.")
    else:
        print(f"Warning: {weight_path} not found, using random init.")

    return model