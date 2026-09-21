import os
import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None
    print("Warning: safetensors not installed.")


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(
            in_features,
            hidden_features
        )

        self.ln = nn.LayerNorm(
            hidden_features
        )

        self.act = act_layer()

        self.fc2 = nn.Linear(
            hidden_features,
            out_features
        )

        self.drop = nn.Dropout(drop)


    def forward(self,x):

        x=self.fc1(x)
        x=self.ln(x)
        x=self.act(x)
        x=self.fc2(x)

        return x



class TimmModel(nn.Module):

    def __init__(
        self,
        model_name,
        num_classes=2256,
        pretrained=True,
        img_size=383,
        local_weight_path=None
    ):

        super(TimmModel,self).__init__()


        self.img_size=img_size

        self.num_classes=num_classes


        self.is_vit = "vit" in model_name.lower()


        # ==========================
        # backbone
        # ==========================

        self.backbone=timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            global_pool=''
        )


        self.num_features=self.backbone.num_features



        # ==========================
        # classification head
        # ==========================

        self.classifier=nn.Linear(
            self.num_features,
            num_classes
        )



        # ==========================
        # load safetensors
        # ==========================

        if local_weight_path is not None:

            if os.path.exists(local_weight_path):

                if load_file is None:
                    raise ImportError(
                        "Please install safetensors"
                    )


                print(
                    "Loading pretrained weights:",
                    local_weight_path
                )


                state_dict=load_file(
                    local_weight_path
                )


                backbone_state={}

                for k,v in state_dict.items():

                    # 去掉分类头
                    if not k.startswith("head"):

                        backbone_state[k]=v



                missing,unexpected = self.backbone.load_state_dict(
                    backbone_state,
                    strict=False
                )


                print(
                    "Loaded backbone weights:",
                    len(backbone_state)
                )


                if len(missing)>0:
                    print(
                        "Missing keys:",
                        missing[:5]
                    )


                if len(unexpected)>0:
                    print(
                        "Unexpected keys:",
                        unexpected[:5]
                    )


            else:

                print(
                    "Warning: weight not found:",
                    local_weight_path
                )



        # ==========================
        # DAC temperature
        # ==========================

        self.logit_scale=torch.nn.Parameter(
            torch.ones([])*np.log(1/0.07)
        )


        self.logit_scale_blocks=torch.nn.Parameter(
            torch.ones([])*np.log(1/0.07)
        )


        self.w_blocks1=torch.nn.Parameter(
            torch.ones([])
        )

        self.w_blocks2=torch.nn.Parameter(
            torch.ones([])
        )

        self.w_blocks3=torch.nn.Parameter(
            torch.ones([])
        )



    def get_config(self):

        return timm.data.resolve_model_data_config(
            self.backbone
        )



    def set_grad_checkpointing(
        self,
        enable=True
    ):

        self.backbone.set_grad_checkpointing(
            enable
        )



    def _forward_single(self,img):


        feat_map=self.backbone.forward_features(
            img
        )


        # ==================================
        # ViT
        # output:
        # B,197,768
        # use CLS token
        # ==================================

        if self.is_vit:

            feat_vec=feat_map[:,0]


        # ==================================
        # CNN
        # output:
        # B,C,H,W
        # ==================================

        else:

            feat_vec=feat_map.mean(
                [-2,-1]
            )


        logits=self.classifier(
            feat_vec
        )


        return (
            feat_vec,
            logits,
            feat_map
        )



    def forward(
        self,
        img1,
        img2=None
    ):


        if img2 is not None:


            out1=self._forward_single(
                img1
            )


            out2=self._forward_single(
                img2
            )


            return (
                out1,
                out2
            )


        else:


            return self._forward_single(
                img1
            )