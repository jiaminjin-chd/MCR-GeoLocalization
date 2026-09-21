import os
import cv2
import copy
import random
import time

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from mask import RandomPatchMask


def get_data(path):
    """扫描路径下所有子文件夹，返回每个子文件夹下的文件列表"""
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files
    return data


class DenseUAVDatasetTrain(Dataset):
    """DenseUAV 训练数据集，与 U1652DatasetTrain 逻辑一致，但适配子文件夹结构"""

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 mask_ratio=0.0,
                 patch_size=16,
                 keep_center=False,
                 center_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.keep_center = keep_center
        self.center_ratio = center_ratio

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)

        # 取 query 和 gallery 共有的 ID
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        # 构建配对：query 取每个 ID 下的第一个文件，gallery 取该 ID 下所有文件
        self.pairs = []
        for idx in self.ids:
            query_files = self.query_dict[idx]["files"]
            if not query_files:  # 跳过空文件夹
                continue
            query_img = os.path.join(self.query_dict[idx]["path"], query_files[0])
            gallery_path = self.gallery_dict[idx]["path"]
            gallery_files = self.gallery_dict[idx]["files"]
            label = self.reverse_map_dict[idx]
            for g in gallery_files:
                self.pairs.append((idx, label, query_img, os.path.join(gallery_path, g)))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):
        idx, label, query_img_path, gallery_img_path = self.samples[index]

        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        # 掩码仅应用于 gallery（图库）图像
        # 在 __getitem__ 中
        if self.mask_ratio > 0:
            mask_fn = RandomPatchMask(
                mask_ratio=self.mask_ratio,
                patch_size=self.patch_size,
                keep_center=self.keep_center,
                center_ratio=self.center_ratio
            )
            query_img = mask_fn(query_img)  # 对query也掩码
            gallery_img = mask_fn(gallery_img)  # 对gallery也掩码

        return query_img, gallery_img, idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self):
        """自定义打乱，保证每个 batch 内 ID 不重复"""
        print("\nShuffle Dataset:")
        pair_pool = copy.deepcopy(self.pairs)
        random.shuffle(pair_pool)

        pairs_epoch = set()
        idx_batch = set()
        batches = []
        current_batch = []
        break_counter = 0
        pbar = tqdm()

        while True:
            pbar.update()
            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                idx, _, _, _ = pair
                if idx not in idx_batch and pair not in pairs_epoch:
                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)
                    break_counter = 0
                else:
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
                    break_counter += 1
                if break_counter >= 512:
                    break
            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()
        time.sleep(0.3)
        self.samples = batches
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))


class DenseUAVDatasetEval(Dataset):
    """DenseUAV 评估数据集（查询/图库）"""

    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()
        self.data_dict = get_data(data_folder)
        self.ids = list(self.data_dict.keys())
        self.transforms = transforms
        self.given_sample_ids = sample_ids
        self.images = []
        self.sample_ids = []
        self.mode = mode
        self.gallery_n = gallery_n

        # 收集所有图像路径及对应 ID
        for sample_id in self.ids:
            for file in self.data_dict[sample_id]["files"]:
                self.images.append(os.path.join(self.data_dict[sample_id]["path"], file))
                self.sample_ids.append(sample_id)

        # gallery 模式下，仅保留 sample_ids 中的 ID，并可限制数量
        if mode == 'gallery' and sample_ids is not None:
            selected = []
            for img, sid in zip(self.images, self.sample_ids):
                if sid in sample_ids:
                    selected.append((img, sid))
            if gallery_n > 0:
                selected = selected[:gallery_n]
            if selected:
                self.images, self.sample_ids = zip(*selected)
            else:
                self.images, self.sample_ids = [], []

    def __getitem__(self, index):
        img_path = self.images[index]
        sample_id = self.sample_ids[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        label = int(sample_id)
        if self.given_sample_ids is not None and sample_id not in self.given_sample_ids:
            label = -1
        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)