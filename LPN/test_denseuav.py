import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

from sklearn.metrics import average_precision_score

from model import two_view_net



# ======================================================
# Dataset
# ======================================================

class DenseUAVDataset(Dataset):

    def __init__(self,root,transform=None):

        self.samples=[]
        self.transform=transform


        for sid in sorted(os.listdir(root)):

            sid_path=os.path.join(root,sid)

            if not os.path.isdir(sid_path):
                continue


            for img in sorted(os.listdir(sid_path)):

                if img.lower().endswith(
                    ('.jpg','.jpeg','.png','.tif','.tiff')
                ):

                    self.samples.append(
                        (
                            os.path.join(
                                sid_path,
                                img
                            ),
                            sid
                        )
                    )


    def __len__(self):
        return len(self.samples)



    def __getitem__(self,index):

        path,sid=self.samples[index]

        img=Image.open(path).convert("RGB")

        if self.transform:
            img=self.transform(img)


        return img,sid



# ======================================================
# Feature
# ======================================================

def extract_feature(model,loader,device):

    model.eval()

    feats=[]
    ids=[]


    with torch.no_grad():

        for imgs,sid in tqdm(loader):

            imgs=imgs.to(device)


            # LPN backbone

            feat=model.model_1(imgs)


            # B,C,part

            feat=feat.permute(
                0,2,1
            )


            feat=feat.reshape(
                feat.size(0),
                -1
            )


            feat=F.normalize(
                feat,
                dim=1
            )


            feats.append(
                feat.cpu()
            )


            ids.extend(sid)



    feats=torch.cat(
        feats,
        dim=0
    )


    return feats,np.array(ids)



# ======================================================
# Official DenseUAV Evaluation
# ======================================================

def evaluate(
        q_feat,
        q_ids,
        g_feat,
        g_ids
):


    print("\nCalculating similarity...")


    sim=torch.matmul(
        q_feat,
        g_feat.T
    )


    recalls={
        1:0,
        5:0,
        10:0
    }


    AP=[]

    valid=0


    for i in range(len(q_ids)):


        gt=(g_ids==q_ids[i]).astype(
            np.int32
        )


        if gt.sum()==0:
            continue


        valid+=1


        order=torch.argsort(
            sim[i],
            descending=True
        ).numpy()



        for k in recalls:

            topk=g_ids[
                order[:k]
            ]

            if q_ids[i] in topk:

                recalls[k]+=1



        ap=average_precision_score(
            gt,
            sim[i].numpy()
        )


        AP.append(ap)



    print(
        "Valid queries:",
        valid
    )


    if valid==0:

        print("No valid query")
        return



    for k in recalls:

        print(
            f"R@{k}: "
            f"{recalls[k]/valid*100:.2f}"
        )


    print(
        f"mAP: {np.mean(AP)*100:.2f}"
    )





# ======================================================
# Main
# ======================================================


def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--model_dir",
        required=True
    )


    parser.add_argument(
        "--data_root",
        default="/root/autodl-tmp/DenseUAV"
    )


    parser.add_argument(
        "--direction",
        default="D2S",
        choices=["D2S","S2D"]
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )


    parser.add_argument(
        "--block",
        type=int,
        default=4
    )


    args=parser.parse_args()



    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print("Device:",device)



    # ==================================================
    # Model
    # ==================================================


    model=two_view_net(

        class_num=2256,

        droprate=0.5,

        stride=1,

        pool='avg',

        share_weight=True,

        LPN=True,

        block=args.block,

        return_feat=False

    )



    ckpt=torch.load(
        os.path.join(
            args.model_dir,
            "best_model.pth"
        ),
        map_location="cpu"
    )


    state=ckpt["model_state_dict"]


    new_state={}


    for k,v in state.items():

        if k.startswith("module."):

            k=k[7:]


        new_state[k]=v



    model.load_state_dict(
        new_state,
        strict=False
    )


    model=model.to(device)


    print("Loaded model")



    # ==================================================
    # Dataset
    # ==================================================


    transform=transforms.Compose([

        transforms.Resize(
            (384,384)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )

    ])




    if args.direction=="D2S":


        q_path=os.path.join(
            args.data_root,
            "test/query_drone"
        )


        g_path=os.path.join(
            args.data_root,
            "test/gallery_satellite"
        )


    else:


        q_path=os.path.join(
            args.data_root,
            "test/gallery_satellite"
        )


        g_path=os.path.join(
            args.data_root,
            "test/query_drone"
        )



    q_set=DenseUAVDataset(
        q_path,
        transform
    )


    g_set=DenseUAVDataset(
        g_path,
        transform
    )



    print("\n==============================")
    print("DenseUAV Official Evaluation")
    print("==============================")

    print(
        "Direction:",
        args.direction
    )


    print(
        "Query:",
        q_path
    )


    print(
        "Gallery:",
        g_path
    )


    print(
        "Query images:",
        len(q_set)
    )


    print(
        "Gallery images:",
        len(g_set)
    )


    print(
        "Query locations:",
        len(set([x[1] for x in q_set.samples]))
    )


    print(
        "Gallery locations:",
        len(set([x[1] for x in g_set.samples]))
    )



    q_loader=DataLoader(
        q_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )


    g_loader=DataLoader(
        g_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )



    q_feat,q_ids=extract_feature(
        model,
        q_loader,
        device
    )


    g_feat,g_ids=extract_feature(
        model,
        g_loader,
        device
    )



    print(
        "Query feature:",
        q_feat.shape
    )


    print(
        "Gallery feature:",
        g_feat.shape
    )


    evaluate(
        q_feat,
        q_ids,
        g_feat,
        g_ids
    )



if __name__=="__main__":

    main()


# import os
# import argparse
# import sys

# import torch
# import torch.nn.functional as F

# import numpy as np

# from tqdm import tqdm

# from PIL import Image

# from torch.utils.data import Dataset, DataLoader

# from torchvision import transforms

# from sklearn.metrics import average_precision_score


# sys.path.append('/root/autodl-tmp/LPN-main')

# from model import two_view_net



# # ==========================================================
# # Dataset
# # ==========================================================

# class ImageFolderWithID(Dataset):

#     def __init__(self, root, transform=None):

#         self.samples=[]

#         self.transform=transform


#         for sid in sorted(os.listdir(root)):


#             sid_path=os.path.join(
#                 root,
#                 sid
#             )


#             if not os.path.isdir(sid_path):
#                 continue


#             for img in sorted(os.listdir(sid_path)):


#                 if img.lower().endswith(
#                     (
#                         '.jpg',
#                         '.jpeg',
#                         '.png',
#                         '.tif',
#                         '.tiff'
#                     )
#                 ):


#                     self.samples.append(
#                         (
#                             os.path.join(
#                                 sid_path,
#                                 img
#                             ),
#                             sid
#                         )
#                     )


#     def __len__(self):

#         return len(self.samples)



#     def __getitem__(self,index):

#         path,sid=self.samples[index]


#         img=Image.open(path).convert('RGB')


#         if self.transform:

#             img=self.transform(img)


#         return img,sid



# # ==========================================================
# # Feature extraction
# # ==========================================================

# def extract_feature(model,loader,device):


#     model.eval()


#     features=[]

#     labels=[]


#     with torch.no_grad():


#         for imgs,ids in tqdm(loader):


#             imgs=imgs.to(device)



#             # LPN feature

#             feat=model.model_1(imgs)


#             # B,C,part

#             feat=feat.permute(
#                 0,
#                 2,
#                 1
#             )


#             feat=feat.reshape(
#                 feat.size(0),
#                 -1
#             )


#             feat=F.normalize(
#                 feat,
#                 dim=1
#             )


#             features.append(
#                 feat.cpu()
#             )


#             labels.extend(ids)



#     features=torch.cat(
#         features,
#         dim=0
#     )


#     labels=np.array(labels)


#     return features,labels



# # ==========================================================
# # Evaluation
# # ==========================================================

# def evaluate(
#     q_feat,
#     q_ids,
#     g_feat,
#     g_ids
# ):


#     print(
#         "Calculating similarity..."
#     )


#     sim=torch.matmul(
#         q_feat,
#         g_feat.T
#     )


#     recalls={

#         1:0,

#         5:0,

#         10:0

#     }


#     AP=[]



#     for i in range(
#         len(q_ids)
#     ):


#         gt=(

#             g_ids==q_ids[i]

#         ).astype(
#             np.int32
#         )


#         if gt.sum()==0:

#             continue



#         order=torch.argsort(
#             sim[i],
#             descending=True
#         ).numpy()



#         for k in recalls:


#             retrieved=g_ids[
#                 order[:k]
#             ]


#             if q_ids[i] in retrieved:

#                 recalls[k]+=1



#         ap=average_precision_score(

#             gt,

#             sim[i].numpy()

#         )


#         AP.append(ap)



#     total=len(q_ids)



#     print()

#     for k in recalls:

#         print(
#             f"R@{k}: {recalls[k]/total*100:.2f}"
#         )



#     print(
#         f"AP: {np.mean(AP)*100:.2f}"
#     )



# # ==========================================================
# # Main
# # ==========================================================

# def main():


#     parser=argparse.ArgumentParser()


#     parser.add_argument(
#         "--model_dir",
#         required=True
#     )


#     parser.add_argument(
#         "--data_root",
#         default="/root/autodl-tmp/DenseUAV"
#     )


#     parser.add_argument(
#         "--block",
#         type=int,
#         default=4
#     )


#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=64
#     )
    
#     parser.add_argument(
#         "--direction",
#         default="D2S",
#         choices=["D2S","S2D"]
#     )


#     args=parser.parse_args()



#     device=torch.device(
#         "cuda"
#         if torch.cuda.is_available()
#         else "cpu"
#     )


#     print(
#         "Device:",
#         device
#     )



#     # ======================================================
#     # Model
#     # ======================================================


#     model=two_view_net(

#         class_num=2256,

#         droprate=0.5,

#         stride=1,

#         pool='avg',

#         share_weight=True,

#         LPN=True,

#         block=args.block,

#         return_feat=False

#     )



#     ckpt_path=os.path.join(

#         args.model_dir,

#         "best_model.pth"

#     )



#     checkpoint=torch.load(

#         ckpt_path,

#         map_location="cpu"

#     )


#     state=checkpoint[
#         "model_state_dict"
#     ]



#     # 兼容保存方式

#     new_state={}


#     for k,v in state.items():

#         if k.startswith("module."):

#             k=k[7:]


#         new_state[k]=v



#     missing,unexpected=model.load_state_dict(

#         new_state,

#         strict=False

#     )


#     print(
#         "Loaded model"
#     )


#     if len(missing)>0:

#         print(
#             "Missing:",
#             len(missing)
#         )


#     if len(unexpected)>0:

#         print(
#             "Unexpected:",
#             len(unexpected)
#         )



#     model=model.to(device)



#     # ======================================================
#     # Data
#     # ======================================================


#     transform=transforms.Compose([


#         transforms.Resize(
#             (
#                 384,
#                 384
#             )
#         ),


#         transforms.ToTensor(),


#         transforms.Normalize(

#             [0.485,0.456,0.406],

#             [0.229,0.224,0.225]

#         )


#     ])
    
    
    
#     # ======================================================
#     # Direction
#     # ======================================================

#     if args.direction=="D2S":

#         query_path=os.path.join(
#             args.data_root,
#             "test/query_drone"
#         )

#         gallery_path=os.path.join(
#             args.data_root,
#             "test/gallery_satellite"
#         )


#     elif args.direction=="S2D":

#         query_path=os.path.join(
#             args.data_root,
#             "test/gallery_satellite"
#         )

#         gallery_path=os.path.join(
#             args.data_root,
#             "test/query_drone"
#         )



#     query_dataset=ImageFolderWithID(
#         query_path,
#         transform
#     )


#     gallery_dataset=ImageFolderWithID(
#         gallery_path,
#         transform
#     )


#     print(
#         "Direction:",
#         args.direction
#     )

#     print(
#         "Query path:",
#         query_path
#     )

#     print(
#         "Gallery path:",
#         gallery_path
#     )



# #     query_dataset=ImageFolderWithID(

# #         os.path.join(

# #             args.data_root,

# #             "test/query_drone"

# #         ),

# #         transform

# #     )


# #     gallery_dataset=ImageFolderWithID(

# #         os.path.join(

# #             args.data_root,

# #             "test/gallery_satellite"

# #         ),

# #         transform

# #     )



#     print(
#         "Query images:",
#         len(query_dataset)
#     )


#     print(
#         "Gallery images:",
#         len(gallery_dataset)
#     )



#     query_loader=DataLoader(

#         query_dataset,

#         batch_size=args.batch_size,

#         shuffle=False,

#         num_workers=4

#     )


#     gallery_loader=DataLoader(

#         gallery_dataset,

#         batch_size=args.batch_size,

#         shuffle=False,

#         num_workers=4

#     )



#     q_feat,q_ids=extract_feature(

#         model,

#         query_loader,

#         device

#     )


#     g_feat,g_ids=extract_feature(

#         model,

#         gallery_loader,

#         device

#     )



#     print(
#         "Query feature:",
#         q_feat.shape
#     )


#     print(
#         "Gallery feature:",
#         g_feat.shape
#     )



#     evaluate(

#         q_feat,

#         q_ids,

#         g_feat,

#         g_ids

#     )




# if __name__=="__main__":

#     main()




















# # import torch
# # import torch.nn.functional as F
# # import numpy as np
# # import os
# # import argparse
# # import sys
# # from tqdm import tqdm
# # from PIL import Image
# # from torchvision import transforms
# # from torch.utils.data import DataLoader, Dataset
# # from sklearn.metrics import average_precision_score


# # sys.path.append('/root/autodl-tmp/LPN-main')

# # from model import two_view_net


# # # ==========================================================
# # # Dataset
# # # ==========================================================

# # class ImageFolderWithID(Dataset):

# #     def __init__(self, root, transform=None):

# #         self.samples = []

# #         self.transform = transform


# #         for cls in sorted(os.listdir(root)):

# #             cls_path = os.path.join(root, cls)

# #             if os.path.isdir(cls_path):

# #                 for img in sorted(os.listdir(cls_path)):

# #                     img_path = os.path.join(cls_path, img)

# #                     if img_path.lower().endswith(
# #                         ('.jpg','.jpeg','.png','.tif','.tiff')
# #                     ):
# #                         self.samples.append(
# #                             (img_path, cls)
# #                         )


# #     def __len__(self):

# #         return len(self.samples)


# #     def __getitem__(self,index):

# #         path, label = self.samples[index]

# #         img = Image.open(path).convert('RGB')


# #         if self.transform:

# #             img = self.transform(img)


# #         return img,label



# # # ==========================================================
# # # LPN feature extraction
# # # ==========================================================

# # def extract_feature(model, loader, device):

# #     feats=[]

# #     labels=[]


# #     model.eval()


# #     with torch.no_grad():

# #         for imgs, ids in tqdm(loader):

# #             imgs=imgs.to(device)


# #             #
# #             # LPN output:
# #             # [B,C,block]
# #             #
# #             f=model.model_1(imgs)


# #             #
# #             # 保留所有part
# #             #
# #             # [B,C,6]
# #             #
# #             f=f.permute(0,2,1)


# #             #
# #             # [B,6C]
# #             #
# #             f=f.reshape(
# #                 f.size(0),
# #                 -1
# #             )


# #             #
# #             # normalize
# #             #
# #             f=F.normalize(
# #                 f,
# #                 dim=1
# #             )


# #             feats.append(
# #                 f.cpu()
# #             )

# #             labels.extend(ids)



# #     feats=torch.cat(
# #         feats,
# #         dim=0
# #     )


# #     labels=np.array(labels)


# #     return feats,labels



# # # ==========================================================
# # # Evaluation
# # # ==========================================================

# # def evaluate(model, query_loader, gallery_loader, device):


# #     q_feats,q_labels=extract_feature(
# #         model,
# #         query_loader,
# #         device
# #     )


# #     g_feats,g_labels=extract_feature(
# #         model,
# #         gallery_loader,
# #         device
# #     )


# #     print("Query feature:",q_feats.shape)
# #     print("Gallery feature:",g_feats.shape)



# #     sim=q_feats @ g_feats.T



# #     for K in [1,5,10]:

# #         topk=torch.topk(
# #             sim,
# #             K,
# #             dim=1
# #         ).indices.numpy()


# #         correct=0


# #         for i in range(len(q_labels)):

# #             retrieved=g_labels[topk[i]]

# #             if q_labels[i] in retrieved:

# #                 correct+=1



# #         recall=correct/len(q_labels)*100


# #         print(
# #             f"R@{K}: {recall:.2f}"
# #         )



# #     # AP

# #     aps=[]


# #     for i in range(len(q_labels)):


# #         gt=(
# #             g_labels==q_labels[i]
# #         ).astype(np.int32)


# #         if gt.sum()>0:


# #             ap=average_precision_score(
# #                 gt,
# #                 sim[i].numpy()
# #             )


# #             aps.append(ap)



# #     AP=np.mean(aps)*100


# #     print(
# #         f"AP: {AP:.2f}"
# #     )



# # # ==========================================================
# # # Main
# # # ==========================================================

# # def main():


# #     parser=argparse.ArgumentParser()


# #     parser.add_argument(
# #         '--model_dir',
# #         required=True
# #     )


# #     parser.add_argument(
# #         '--data_root',
# #         default='/root/autodl-tmp/DenseUAV'
# #     )


# #     parser.add_argument(
# #         '--block',
# #         type=int,
# #         default=6
# #     )


# #     parser.add_argument(
# #         '--batch_size',
# #         type=int,
# #         default=64
# #     )


# #     args=parser.parse_args()



# #     device=torch.device(
# #         'cuda'
# #         if torch.cuda.is_available()
# #         else 'cpu'
# #     )



# #     print(
# #         "Device:",
# #         device
# #     )



# #     # ==================================================
# #     # model
# #     # ==================================================


# #     model=two_view_net(

# #         class_num=2256,

# #         droprate=0.5,

# #         stride=1,

# #         pool='avg',

# #         share_weight=True,

# #         LPN=True,

# #         block=args.block,

# #         return_feat=False
# #     )



# #     ckpt_path=os.path.join(
# #         args.model_dir,
# #         'best_model.pth'
# #     )


# #     ckpt=torch.load(
# #         ckpt_path,
# #         map_location='cpu'
# #     )


# #     state=ckpt['model_state_dict']



# #     new_state={}



# #     for k,v in state.items():

# #         if k.startswith('model.'):

# #             new_state[k[6:]]=v

# #         else:

# #             new_state[k]=v



# #     model.load_state_dict(
# #         new_state,
# #         strict=False
# #     )


# #     model=model.to(device)



# #     print(
# #         "Loaded:",
# #         ckpt_path
# #     )



# #     # ==================================================
# #     # data
# #     # ==================================================


# #     transform=transforms.Compose([

# #         transforms.Resize(
# #             (384,384)
# #         ),

# #         transforms.ToTensor(),

# #         transforms.Normalize(
# #             [0.485,0.456,0.406],
# #             [0.229,0.224,0.225]
# #         )

# #     ])



# #     query_path=os.path.join(
# #         args.data_root,
# #         'test/query_drone'
# #     )


# #     gallery_path=os.path.join(
# #         args.data_root,
# #         'test/gallery_satellite'
# #     )



# #     query_dataset=ImageFolderWithID(
# #         query_path,
# #         transform
# #     )


# #     gallery_dataset=ImageFolderWithID(
# #         gallery_path,
# #         transform
# #     )



# #     print(
# #         "Query images:",
# #         len(query_dataset)
# #     )

# #     print(
# #         "Gallery images:",
# #         len(gallery_dataset)
# #     )



# #     query_loader=DataLoader(
# #         query_dataset,
# #         batch_size=args.batch_size,
# #         shuffle=False,
# #         num_workers=4
# #     )


# #     gallery_loader=DataLoader(
# #         gallery_dataset,
# #         batch_size=args.batch_size,
# #         shuffle=False,
# #         num_workers=4
# #     )



# #     evaluate(
# #         model,
# #         query_loader,
# #         gallery_loader,
# #         device
# #     )



# # if __name__=="__main__":

# #     main()