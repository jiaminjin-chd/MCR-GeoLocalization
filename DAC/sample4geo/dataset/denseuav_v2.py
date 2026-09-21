import os
import random
import cv2
import numpy as np

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
import cv2
from mask import RandomPatchMask

from torch.utils.data import Dataset


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".JPG", ".JPEG",
    ".png", ".PNG",
    ".tif", ".tiff", ".TIF", ".TIFF"
}


def list_id_dirs(root):
    """
    返回:
        {
            "000001": "/.../000001",
            ...
        }

    只读取 root 的一级 ID 文件夹。
    """
    result = {}

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"DenseUAV directory does not exist: {root}"
        )

    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)

        if os.path.isdir(path):
            result[name] = path

    return result


def list_images(folder):
    """
    返回一个 ID 文件夹下面的所有图像。
    """
    images = []

    if not os.path.isdir(folder):
        return images

    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)

        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(name)[1]

        if ext in IMAGE_EXTS:
            images.append(path)

    return images


def read_image(path):
    """
    OpenCV 读取 JPG/TIF/TIFF。
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(
            f"Failed to read image: {path}"
        )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    return img


class DenseUAVDatasetTrainV2(Dataset):
    """
    DenseUAV v2 training dataset.

    D2S:
        query   = drone
        gallery = satellite

    S2D:
        query   = satellite
        gallery = drone

    每个 sampling location 对应一个 class。

    与旧版不同：
    1. 不固定取 query 第一张图。
    2. 每次随机选择同一 location 的 query/gallery image。
    3. 一个 location 一个 label。
    4. 每个 epoch 按 location 构造 balanced pairs。
    """

    def __init__(
        self,
        query_folder,
        gallery_folder,
        transforms_query=None,
        transforms_gallery=None,
        prob_flip=0.5,
        samples_per_id=3,
        mask_ratio=0.0
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio

        if mask_ratio > 0:
            self.mask_transform = RandomPatchMask(
                mask_ratio=mask_ratio,
                patch_size=16,
                fill_value=0,
                keep_center=True,
                center_ratio=0.75
            )
        else:
            self.mask_transform = None

        self.query_folder = query_folder
        self.gallery_folder = gallery_folder

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery

        self.prob_flip = prob_flip
        self.samples_per_id = samples_per_id

        self.query_dict = list_id_dirs(query_folder)
        self.gallery_dict = list_id_dirs(gallery_folder)

        # 训练集要求两个 modality 使用相同 location ID
        self.ids = sorted(
            set(self.query_dict.keys())
            &
            set(self.gallery_dict.keys())
        )

        if len(self.ids) == 0:
            raise RuntimeError(
                "DenseUAV v2: no common IDs between "
                f"{query_folder} and {gallery_folder}"
            )

        self.label_map = {
            sid: i
            for i, sid in enumerate(self.ids)
        }

        self.query_images = {}
        self.gallery_images = {}

        valid_ids = []

        for sid in self.ids:

            qimgs = list_images(
                self.query_dict[sid]
            )

            gimgs = list_images(
                self.gallery_dict[sid]
            )

            if len(qimgs) == 0:
                continue

            if len(gimgs) == 0:
                continue

            self.query_images[sid] = qimgs
            self.gallery_images[sid] = gimgs

            valid_ids.append(sid)

        self.ids = valid_ids

        self.label_map = {
            sid: i
            for i, sid in enumerate(self.ids)
        }

        # 每个 ID 默认一个 sample。
        # 可以通过 samples_per_id 增加一个 epoch 的采样量。
        self.samples = []

        self._build_epoch_samples()

        print("=" * 70)
        print("DenseUAVDatasetTrainV2")
        print("=" * 70)
        print("Query folder  :", query_folder)
        print("Gallery folder:", gallery_folder)
        print("Valid IDs     :", len(self.ids))
        print("Samples/ID    :", samples_per_id)
        print("Epoch samples :", len(self.samples))
        print("=" * 70)

    def _build_epoch_samples(self):

        self.samples = []

        for sid in self.ids:
            for _ in range(self.samples_per_id):
                self.samples.append(sid)

        random.shuffle(self.samples)

    def shuffle(self):
        """
        DAC trainer 会调用 dataset.shuffle()。
        """
        self._build_epoch_samples()

    def __getitem__(self, index):

        sid = self.samples[index]

        qpath = random.choice(
            self.query_images[sid]
        )

        gpath = random.choice(
            self.gallery_images[sid]
        )


        query_img = read_image(qpath)
        gallery_img = read_image(gpath)


        # DAC 原始 flip
        if random.random() < self.prob_flip:

            query_img = cv2.flip(
                query_img,
                1
            )

            gallery_img = cv2.flip(
                gallery_img,
                1
            )


        # ==========================
        # resize before mask
        # ==========================

        query_img = cv2.resize(
            query_img,
            (384,384),
            interpolation=cv2.INTER_LINEAR
        )

        gallery_img = cv2.resize(
            gallery_img,
            (384,384),
            interpolation=cv2.INTER_LINEAR
        )


        # ==========================
        # Apply mask
        # ==========================

        if self.mask_transform is not None:

            query_img = self.mask_transform(
                query_img
            )

            gallery_img = self.mask_transform(
                gallery_img
            )


        # ==========================
        # DAC transform
        # ==========================

        if self.transforms_query is not None:

            query_img = self.transforms_query(
                image=query_img
            )["image"]


        if self.transforms_gallery is not None:

            gallery_img = self.transforms_gallery(
                image=gallery_img
            )["image"]


        label = self.label_map[sid]


        return (
            query_img,
            gallery_img,
            sid,
            label
        )

    def __len__(self):
        return len(self.samples)


class DenseUAVDatasetEvalV2(Dataset):
    """
    DenseUAV v2 evaluation dataset.

    注意：
    这里不再把 gallery 限制成 query 的同名 ID。

    Gallery 必须保留全部 3033 个 sampling locations。
    """

    def __init__(
        self,
        data_folder,
        transforms=None,
    ):
        super().__init__()

        self.data_folder = data_folder
        self.transforms = transforms

        self.id_dirs = list_id_dirs(
            data_folder
        )

        self.images = []
        self.sample_ids = []

        for sid in sorted(self.id_dirs.keys()):

            paths = list_images(
                self.id_dirs[sid]
            )

            for path in paths:

                self.images.append(path)
                self.sample_ids.append(sid)

        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {data_folder}"
            )

        print(
            f"DenseUAV Eval: {data_folder}"
        )
        print(
            f"  IDs    : {len(self.id_dirs)}"
        )
        print(
            f"  images : {len(self.images)}"
        )

    def __getitem__(self, index):

        path = self.images[index]
        sid = self.sample_ids[index]

        img = read_image(path)

        if self.transforms is not None:

            img = self.transforms(
                image=img
            )["image"]

        return img, str(sid)

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):

        return set(self.sample_ids)


def get_transforms(
    img_size=384,
    is_train=True
):

    mean = [
        0.485,
        0.456,
        0.406
    ]

    std = [
        0.229,
        0.224,
        0.225
    ]


    val_transform = A.Compose([

        A.Resize(
            img_size,
            img_size,
            interpolation=cv2.INTER_LINEAR_EXACT,
            p=1.0
        ),

        A.Normalize(
            mean,
            std
        ),

        ToTensorV2()

    ])


    train_transform_query = A.Compose([

        A.ImageCompression(
            quality_lower=90,
            quality_upper=100,
            p=0.5
        ),

        A.ColorJitter(
            brightness=0.15,
            contrast=0.3,
            saturation=0.3,
            hue=0.3,
            p=0.5
        ),

        A.Normalize(
            mean,
            std
        ),

        ToTensorV2()

    ])


    train_transform_gallery = A.Compose([

        A.ImageCompression(
            quality_lower=90,
            quality_upper=100,
            p=0.5
        ),

        A.ColorJitter(
            brightness=0.15,
            contrast=0.3,
            saturation=0.3,
            hue=0.3,
            p=0.5
        ),

        A.Normalize(
            mean,
            std
        ),

        ToTensorV2()

    ])


    if is_train:

        return (
            train_transform_query,
            train_transform_gallery
        )

    else:

        return val_transform