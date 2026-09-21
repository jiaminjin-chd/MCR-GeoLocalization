import os
import cv2
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

import copy
import random
import time

from tqdm import tqdm


# ============================================================
# DenseUAV Dataset
#
# Dataset root:
#
# /root/autodl-tmp/DenseUAV/
#
# train/
#   drone/
#       000001/
#           H100.JPG
#           H90.JPG
#           H80.JPG
#
#   satellite/
#       000001/
#           H100_old.tif
#           H90_old.tif
#           H80_old.tif
#           H100.tif
#           H90.tif
#           H80.tif
#
# test/
#   query_drone/
#   query_satellite/
#
# ============================================================


# ------------------------------------------------------------
# Supported image extensions
# ------------------------------------------------------------

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


# ------------------------------------------------------------
# Get files inside each ID directory
# ------------------------------------------------------------

def get_data(path, remove_old=True):
    """
    Read DenseUAV directory.

    Return format:

    {
        "000001": {
            "path": ".../000001",
            "files": [
                "H80.JPG",
                "H90.JPG",
                "H100.JPG"
            ]
        },
        ...
    }

    remove_old=True:
        DenseUAV satellite contains:
            H100_old.tif
            H90_old.tif
            H80_old.tif

        These old satellite images are NOT used.
    """

    if not os.path.isdir(path):
        raise FileNotFoundError(
            "\nDenseUAV directory does not exist:\n"
            f"{path}\n"
        )

    data = {}

    for sample_id in sorted(os.listdir(path)):

        sample_path = os.path.join(path, sample_id)

        if not os.path.isdir(sample_path):
            continue

        files = []

        for filename in sorted(os.listdir(sample_path)):

            file_path = os.path.join(sample_path, filename)

            if not os.path.isfile(file_path):
                continue

            lower_name = filename.lower()

            if not lower_name.endswith(IMAGE_EXTENSIONS):
                continue

            # ------------------------------------------------
            # IMPORTANT:
            # DenseUAV has H100_old.tif / H90_old.tif / H80_old.tif
            #
            # We only use the current satellite images.
            # ------------------------------------------------
            if remove_old and "_old" in lower_name:
                continue

            files.append(filename)

        if len(files) == 0:
            continue

        data[str(sample_id)] = {
            "path": sample_path,
            "files": files,
        }

    return data


# ------------------------------------------------------------
# Read image safely
# ------------------------------------------------------------

def read_image(img_path):
    """
    Read JPG / TIF / TIFF using OpenCV.

    DenseUAV satellite images are TIFF.
    """

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(
            "\nFailed to read image:\n"
            f"{img_path}\n"
            "\n"
            "Please check whether the file is corrupted "
            "or OpenCV cannot read this TIFF file."
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# ============================================================
# Training Dataset
# ============================================================

class DenseUAVDatasetTrain(Dataset):

    def __init__(
            self,
            query_folder,
            gallery_folder,
            transforms_query=None,
            transforms_gallery=None,
            prob_flip=0.5,
            shuffle_batch_size=128):

        super().__init__()

        print("\n" + "=" * 70)
        print("Loading DenseUAV Training Dataset")
        print("=" * 70)

        print("Query folder:")
        print(query_folder)

        print("Gallery folder:")
        print(gallery_folder)

        # ----------------------------------------------------
        # Load data
        # ----------------------------------------------------

        self.query_dict = get_data(
            query_folder,
            remove_old=True
        )

        self.gallery_dict = get_data(
            gallery_folder,
            remove_old=True
        )

        print(
            "Query IDs:",
            len(self.query_dict)
        )

        print(
            "Gallery IDs:",
            len(self.gallery_dict)
        )

        # ----------------------------------------------------
        # Only use IDs existing in both views
        # ----------------------------------------------------

        self.ids = list(
            set(self.query_dict.keys())
            &
            set(self.gallery_dict.keys())
        )

        self.ids.sort()

        print(
            "Matched training IDs:",
            len(self.ids)
        )

        if len(self.ids) == 0:
            raise RuntimeError(
                "\nNo matched DenseUAV IDs found!\n"
                f"query_folder = {query_folder}\n"
                f"gallery_folder = {gallery_folder}\n"
            )

        # ----------------------------------------------------
        # Map original DenseUAV ID -> classification label
        #
        # Example:
        #
        # 000001 -> 0
        # 000002 -> 1
        # ...
        # 002256 -> 2255
        #
        # This is safer than directly using int(sample_id).
        # ----------------------------------------------------

        self.map_dict = {
            sample_id: index
            for index, sample_id in enumerate(self.ids)
        }

        self.reverse_map_dict = {
            index: sample_id
            for index, sample_id in enumerate(self.ids)
        }

        # ----------------------------------------------------
        # Build training pairs
        #
        # For each DenseUAV ID:
        #
        # query:
        #   one drone image
        #
        # gallery:
        #   H80 satellite
        #   H90 satellite
        #   H100 satellite
        #
        # Therefore one ID produces 3 pairs.
        # ----------------------------------------------------

        self.pairs = []

        for sample_id in self.ids:

            query_path = self.query_dict[sample_id]["path"]

            query_files = self.query_dict[sample_id]["files"]

            gallery_path = self.gallery_dict[sample_id]["path"]

            gallery_files = self.gallery_dict[sample_id]["files"]

            # ------------------------------------------------
            # DenseUAV normally has:
            #
            # drone:
            # H100.JPG
            # H90.JPG
            # H80.JPG
            #
            # We use H100 as the query image here.
            #
            # If H100 is unavailable, fall back to first image.
            # ------------------------------------------------

            query_file = None

            for preferred_name in [
                "H100.JPG",
                "H100.jpg",
                "H90.JPG",
                "H90.jpg",
                "H80.JPG",
                "H80.jpg",
            ]:

                if preferred_name in query_files:
                    query_file = preferred_name
                    break

            if query_file is None:
                query_file = sorted(query_files)[0]

            query_img = os.path.join(
                query_path,
                query_file
            )

            label = self.map_dict[sample_id]

            # ------------------------------------------------
            # Pair query with every valid satellite image.
            #
            # Example:
            #
            # 000001:
            #
            # drone:
            #     H100.JPG
            #
            # satellite:
            #     H100.tif
            #     H90.tif
            #     H80.tif
            #
            # => 3 training pairs
            # ------------------------------------------------

            for gallery_file in gallery_files:

                gallery_img = os.path.join(
                    gallery_path,
                    gallery_file
                )

                self.pairs.append(
                    (
                        sample_id,
                        label,
                        query_img,
                        gallery_img,
                    )
                )

        # ----------------------------------------------------
        # Transforms
        # ----------------------------------------------------

        self.transforms_query = transforms_query

        self.transforms_gallery = transforms_gallery

        self.prob_flip = prob_flip

        self.shuffle_batch_size = shuffle_batch_size

        # ----------------------------------------------------
        # Current epoch samples
        # ----------------------------------------------------

        self.samples = copy.deepcopy(
            self.pairs
        )

        print(
            "Training pairs:",
            len(self.pairs)
        )

        print("=" * 70)

    # --------------------------------------------------------
    # Get item
    # --------------------------------------------------------

    def __getitem__(self, index):

        idx, label, query_img_path, gallery_img_path = \
            self.samples[index]

        # ----------------------------------------------------
        # Read images
        # ----------------------------------------------------

        query_img = read_image(
            query_img_path
        )

        gallery_img = read_image(
            gallery_img_path
        )

        # ----------------------------------------------------
        # Horizontal flip
        #
        # Same random flip for both views.
        # ----------------------------------------------------

        if np.random.random() < self.prob_flip:

            query_img = cv2.flip(
                query_img,
                1
            )

            gallery_img = cv2.flip(
                gallery_img,
                1
            )

        # ----------------------------------------------------
        # Query transform
        # ----------------------------------------------------

        if self.transforms_query is not None:

            query_img = self.transforms_query(
                image=query_img
            )["image"]

        # ----------------------------------------------------
        # Gallery transform
        # ----------------------------------------------------

        if self.transforms_gallery is not None:

            gallery_img = self.transforms_gallery(
                image=gallery_img
            )["image"]

        return (
            query_img,
            gallery_img,
            idx,
            label
        )

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    def __len__(self):

        return len(self.samples)

    # --------------------------------------------------------
    # Custom shuffle
    #
    # Same logic as DAC University dataset.
    #
    # Avoid putting the same ID repeatedly inside one batch.
    # --------------------------------------------------------

    def shuffle(self):

        print("\nShuffle DenseUAV Dataset:")

        pair_pool = copy.deepcopy(
            self.pairs
        )

        random.shuffle(
            pair_pool
        )

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

                # --------------------------------------------
                # New class in current batch
                # --------------------------------------------

                if (
                    idx not in idx_batch
                    and
                    pair not in pairs_epoch
                ):

                    idx_batch.add(idx)

                    current_batch.append(pair)

                    pairs_epoch.add(pair)

                    break_counter = 0

                else:

                    if pair not in pairs_epoch:

                        pair_pool.append(pair)

                    break_counter += 1

                # --------------------------------------------
                # Prevent infinite loop
                # --------------------------------------------

                if break_counter >= 512:

                    break

            else:

                break

            # ------------------------------------------------
            # Batch is full
            # ------------------------------------------------

            if len(current_batch) >= self.shuffle_batch_size:

                batches.extend(
                    current_batch
                )

                idx_batch = set()

                current_batch = []

        pbar.close()

        time.sleep(0.3)

        self.samples = batches

        print(
            "Original Length:",
            len(self.pairs)
        )

        print(
            "Length after Shuffle:",
            len(self.samples)
        )

        print(
            "Break Counter:",
            break_counter
        )

        print(
            "Pairs left out:",
            len(self.pairs) - len(self.samples)
        )

        if len(self.samples) > 0:

            print(
                "First ID:",
                self.samples[0][0]
            )

            print(
                "Last ID:",
                self.samples[-1][0]
            )


# ============================================================
# Evaluation Dataset
# ============================================================

class DenseUAVDatasetEval(Dataset):

    def __init__(
            self,
            data_folder,
            mode,
            transforms=None,
            sample_ids=None,
            gallery_n=-1):

        super().__init__()

        print("\nLoading DenseUAV Evaluation Dataset:")

        print(
            "Mode:",
            mode
        )

        print(
            "Folder:",
            data_folder
        )

        # ----------------------------------------------------
        # Load data
        # ----------------------------------------------------

        self.data_dict = get_data(
            data_folder,
            remove_old=True
        )

        self.ids = sorted(
            list(self.data_dict.keys())
        )

        self.transforms = transforms

        self.given_sample_ids = sample_ids

        self.images = []

        self.sample_ids = []

        self.mode = mode

        self.gallery_n = gallery_n

        # ----------------------------------------------------
        # Gallery sampling
        #
        # gallery_n = -1:
        #     use all gallery images
        #
        # Otherwise:
        #     use first N images per ID
        # ----------------------------------------------------

        for sample_id in self.ids:

            files = self.data_dict[
                sample_id
            ]["files"]

            if (
                self.gallery_n != -1
                and mode == "gallery"
            ):

                files = files[
                    :self.gallery_n
                ]

            for filename in files:

                img_path = os.path.join(
                    self.data_dict[
                        sample_id
                    ]["path"],
                    filename
                )

                self.images.append(
                    img_path
                )

                self.sample_ids.append(
                    sample_id
                )

        print(
            "IDs:",
            len(self.ids)
        )

        print(
            "Images:",
            len(self.images)
        )

    # --------------------------------------------------------
    # Get item
    # --------------------------------------------------------

    def __getitem__(self, index):

        img_path = self.images[index]

        sample_id = self.sample_ids[index]

        img = read_image(
            img_path
        )

        if self.transforms is not None:

            img = self.transforms(
                image=img
            )["image"]

        # ----------------------------------------------------
        # Important:
        #
        # DenseUAV test IDs are strings such as 002300.
        #
        # DAC evaluation only needs query/gallery IDs
        # to be equal for positive pairs.
        #
        # Therefore int(sample_id) is OK here.
        # ----------------------------------------------------

        try:

            label = int(sample_id)

        except ValueError:

            label = sample_id

        # ----------------------------------------------------
        # If gallery contains IDs that aren't in query,
        # mark them as junk = -1.
        #
        # This is exactly what DAC's evaluation code expects.
        # ----------------------------------------------------

        if self.given_sample_ids is not None:

            if sample_id not in self.given_sample_ids:

                label = -1

        return img, label

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    def __len__(self):

        return len(self.images)

    # --------------------------------------------------------
    # Return all sample IDs
    # --------------------------------------------------------

    def get_sample_ids(self):

        return set(
            self.sample_ids
        )


# ============================================================
# Image Augmentation
# ============================================================

def get_transforms(
        img_size,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]):

    # --------------------------------------------------------
    # Validation transform
    # --------------------------------------------------------

    val_transforms = A.Compose([

        A.Resize(
            img_size[0],
            img_size[1],
            interpolation=cv2.INTER_LINEAR_EXACT,
            p=1.0
        ),

        A.Normalize(
            mean,
            std
        ),

        ToTensorV2(),

    ])

    # --------------------------------------------------------
    # Satellite training transform
    # --------------------------------------------------------

    train_sat_transforms = A.Compose([

        A.ImageCompression(
            quality_lower=90,
            quality_upper=100,
            p=0.5
        ),

        A.Resize(
            img_size[0],
            img_size[1],
            interpolation=cv2.INTER_LINEAR_EXACT,
            p=1.0
        ),

        A.ColorJitter(
            brightness=0.15,
            contrast=0.3,
            saturation=0.3,
            hue=0.3,
            always_apply=False,
            p=0.5
        ),

        A.OneOf([
            A.AdvancedBlur(
                p=1.0
            ),

            A.Sharpen(
                p=1.0
            ),

        ], p=0.3),

        A.OneOf([

            A.GridDropout(
                ratio=0.4,
                p=1.0
            ),

            A.CoarseDropout(
                max_holes=25,
                max_height=int(
                    0.2 * img_size[0]
                ),
                max_width=int(
                    0.2 * img_size[1]
                ),
                min_holes=10,
                min_height=int(
                    0.1 * img_size[0]
                ),
                min_width=int(
                    0.1 * img_size[1]
                ),
                p=1.0
            ),

        ], p=0.3),

        A.RandomRotate90(
            p=1.0
        ),

        A.Normalize(
            mean,
            std
        ),

        ToTensorV2(),

    ])

    # --------------------------------------------------------
    # Drone training transform
    # --------------------------------------------------------

    train_drone_transforms = A.Compose([

        A.ImageCompression(
            quality_lower=90,
            quality_upper=100,
            p=0.5
        ),

        A.Resize(
            img_size[0],
            img_size[1],
            interpolation=cv2.INTER_LINEAR_EXACT,
            p=1.0
        ),

        A.ColorJitter(
            brightness=0.15,
            contrast=0.7,
            saturation=0.3,
            hue=0.3,
            always_apply=False,
            p=0.5
        ),

        A.OneOf([

            A.AdvancedBlur(
                p=1.0
            ),

            A.Sharpen(
                p=1.0
            ),

        ], p=0.3),

        A.OneOf([

            A.GridDropout(
                ratio=0.4,
                p=1.0
            ),

            A.CoarseDropout(
                max_holes=25,
                max_height=int(
                    0.2 * img_size[0]
                ),
                max_width=int(
                    0.2 * img_size[1]
                ),
                min_holes=10,
                min_height=int(
                    0.1 * img_size[0]
                ),
                min_width=int(
                    0.1 * img_size[1]
                ),
                p=1.0
            ),

        ], p=0.3),

        A.Normalize(
            mean,
            std
        ),

        ToTensorV2(),

    ])

    return (
        val_transforms,
        train_sat_transforms,
        train_drone_transforms
    )


# import os
# import cv2
# import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from torch.utils.data import Dataset
# import copy
# from tqdm import tqdm
# import time
# import random
# from albumentations.core.transforms_interface import ImageOnlyTransform
# import imgaug.augmenters as iaa
# from mask import RandomPatchMask
#
#
# def get_data(path):
#     data = {}
#     for root, dirs, files in os.walk(path, topdown=False):
#         for name in dirs:
#             data[name] = {"path": os.path.join(root, name)}
#             for _, _, files in os.walk(data[name]["path"], topdown=False):
#                 data[name]["files"] = files
#
#     return data
#
#
# class U1652DatasetTrain(Dataset):
#
#     def __init__(self,
#                  query_folder,
#                  gallery_folder,
#                  transforms_query=None,
#                  transforms_gallery=None,
#                  prob_flip=0.5,
#                  shuffle_batch_size=128,
#                  mask_ratio=0.0,  # 新增
#                  patch_size=16,  # 新增
#                  keep_center=False,  # 新增
#                  center_ratio=0.75):  # 新增
#         super().__init__()
#         self.mask_ratio = mask_ratio
#         self.patch_size = patch_size
#         self.keep_center = keep_center
#         self.center_ratio = center_ratio
#
#         self.query_dict = get_data(query_folder)
#         self.gallery_dict = get_data(gallery_folder)
#
#         # use only folders that exists for both gallery and query
#         self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
#         self.ids.sort()
#         self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
#         self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}
#
#         self.pairs = []
#
#         for idx in self.ids:
#
#             query_img = "{}/{}".format(self.query_dict[idx]["path"],
#                                        self.query_dict[idx]["files"][0])
#
#             gallery_path = self.gallery_dict[idx]["path"]
#             gallery_imgs = self.gallery_dict[idx]["files"]
#
#             label = self.reverse_map_dict[idx]
#
#             for g in gallery_imgs:
#                 self.pairs.append((idx, label, query_img, "{}/{}".format(gallery_path, g)))
#
#         self.transforms_query = transforms_query
#         self.transforms_gallery = transforms_gallery
#         self.prob_flip = prob_flip
#         self.shuffle_batch_size = shuffle_batch_size
#
#         self.samples = copy.deepcopy(self.pairs)
#
#     def __getitem__(self, index):
#
#         idx, label, query_img_path, gallery_img_path = self.samples[index]
#
#         # for query there is only one file in folder
#         query_img = cv2.imread(query_img_path)
#         query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
#
#         gallery_img = cv2.imread(gallery_img_path)
#         gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
#
#         if np.random.random() < self.prob_flip:
#             query_img = cv2.flip(query_img, 1)
#             gallery_img = cv2.flip(gallery_img, 1)
#
#
#             # image transforms
#         if self.transforms_query is not None:
#             query_img = self.transforms_query(image=query_img)['image']
#
#         if self.transforms_gallery is not None:
#             gallery_img = self.transforms_gallery(image=gallery_img)['image']
#
#         # 应用掩码（如果启用）
#         if self.mask_ratio > 0:
#             mask_fn = RandomPatchMask(
#                 mask_ratio=self.mask_ratio,
#                 patch_size=self.patch_size,
#                 keep_center=self.keep_center,
#                 center_ratio=self.center_ratio
#             )
#             # query_img 和 gallery_img 此时已经是 (C, H, W) 的 torch.Tensor
#             # query_img = mask_fn(query_img)  #卫星图不掩码
#             gallery_img = mask_fn(gallery_img)
#
#
#         return query_img, gallery_img, idx, label
#
#     def __len__(self):
#         return len(self.samples)
#
#     def shuffle(self, ):
#
#         '''
#         custom shuffle function for unique class_id sampling in batch
#         '''
#
#         print("\nShuffle Dataset:")
#
#         pair_pool = copy.deepcopy(self.pairs)
#
#         # Shuffle pairs order
#         random.shuffle(pair_pool)
#
#         # Lookup if already used in epoch
#         pairs_epoch = set()
#         idx_batch = set()
#
#         # buckets
#         batches = []
#         current_batch = []
#
#         # counter
#         break_counter = 0
#
#         # progressbar
#         pbar = tqdm()
#
#         while True:
#
#             pbar.update()
#
#             if len(pair_pool) > 0:
#                 pair = pair_pool.pop(0)
#
#                 idx, _, _, _ = pair
#
#                 if idx not in idx_batch and pair not in pairs_epoch:
#
#                     idx_batch.add(idx)
#                     current_batch.append(pair)
#                     pairs_epoch.add(pair)
#
#                     break_counter = 0
#
#                 else:
#                     # if pair fits not in batch and is not already used in epoch -> back to pool
#                     if pair not in pairs_epoch:
#                         pair_pool.append(pair)
#
#                     break_counter += 1
#
#                 if break_counter >= 512:
#                     break
#
#             else:
#                 break
#
#             if len(current_batch) >= self.shuffle_batch_size:
#                 # empty current_batch bucket to batches
#                 batches.extend(current_batch)
#                 idx_batch = set()
#                 current_batch = []
#
#         pbar.close()
#
#         # wait before closing progress bar
#         time.sleep(0.3)
#
#         self.samples = batches
#
#         print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
#         print("Break Counter:", break_counter)
#         print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
#         print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))
#
#
# class U1652DatasetEval(Dataset):
#
#     def __init__(self,
#                  data_folder,
#                  mode,
#                  transforms=None,
#                  sample_ids=None,
#                  gallery_n=-1):
#         super().__init__()
#
#         self.data_dict = get_data(data_folder)
#
#         # use only folders that exists for both gallery and query
#         self.ids = list(self.data_dict.keys())
#
#         self.transforms = transforms
#
#         self.given_sample_ids = sample_ids
#
#         self.images = []
#         self.sample_ids = []
#
#         self.mode = mode
#
#         self.gallery_n = gallery_n
#
#         for i, sample_id in enumerate(self.ids):
#
#             for j, file in enumerate(self.data_dict[sample_id]["files"]):
#                 self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
#                                                   file))
#
#                 self.sample_ids.append(sample_id)
#
#     def __getitem__(self, index):
#
#         img_path = self.images[index]
#         sample_id = self.sample_ids[index]
#
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         # if self.mode == "sat":
#
#         #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#         #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
#         #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)
#
#         #    img_0_90 = np.concatenate([img, img90], axis=1)
#         #    img_180_270 = np.concatenate([img180, img270], axis=1)
#
#         #    img = np.concatenate([img_0_90, img_180_270], axis=0)
#
#         # image transforms
#         if self.transforms is not None:
#             img = self.transforms(image=img)['image']
#
#         label = int(sample_id)
#         if self.given_sample_ids is not None:
#             if sample_id not in self.given_sample_ids:
#                 label = -1
#
#         return img, label
#
#     def __len__(self):
#         return len(self.images)
#
#     def get_sample_ids(self):
#         return set(self.sample_ids)
#
#
# # ********************************************** apply multi-weather setting for U1652 **********************************************************
#
# class ImgAugTransform(ImageOnlyTransform):
#     def __init__(self, aug, always_apply=False, p=1.0):
#         super(ImgAugTransform, self).__init__(always_apply, p)
#         self.aug = aug
#
#     def apply(self, img, **params):
#         return self.aug(image=img)
#
#
# # 自定义云层变换
# class CustomCloudLayer(ImgAugTransform):
#     def __init__(self, intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2,
#                  alpha_min=1.0, alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2,
#                  sparsity=0.9, density_multiplier=0.5, seed=None, always_apply=False, p=1.0):
#         aug = iaa.CloudLayer(
#             intensity_mean=intensity_mean,
#             intensity_freq_exponent=intensity_freq_exponent,
#             intensity_coarse_scale=intensity_coarse_scale,
#             alpha_min=alpha_min,
#             alpha_multiplier=alpha_multiplier,
#             alpha_size_px_max=alpha_size_px_max,
#             alpha_freq_exponent=alpha_freq_exponent,
#             sparsity=sparsity,
#             density_multiplier=density_multiplier,
#             seed=seed
#         )
#         super(CustomCloudLayer, self).__init__(aug, always_apply, p)
#
#
# # 自定义雨变换
# class CustomRain(ImgAugTransform):
#     def __init__(self, drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=None, always_apply=False, p=1.0):
#         aug = iaa.Rain(
#             drop_size=drop_size,
#             speed=speed,
#             seed=seed
#         )
#         super(CustomRain, self).__init__(aug, always_apply, p)
#
#
# # 自定义雪花变换
# class CustomSnowflakes(ImgAugTransform):
#     def __init__(self, flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=None, always_apply=False, p=1.0):
#         aug = iaa.Snowflakes(
#             flake_size=flake_size,
#             speed=speed,
#             seed=seed
#         )
#         super(CustomSnowflakes, self).__init__(aug, always_apply, p)
#
#
# iaa_weather_list = [
#
#     # 0. Normal
#     A.NoOp(),
#     # 1. Fog
#     A.Compose([
#         CustomCloudLayer()
#     ]),
#     # 2. Rain
#     A.Compose([
#         CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
#         CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
#         CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
#         CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
#         CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),
#     ]),
#     # 3. Snow
#     A.Compose([
#         CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
#         CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
#         CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
#         CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
#         CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),
#     ]),
#     # 4. Fog+Rain
#     A.Compose([
#         CustomCloudLayer(),
#         CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
#         CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36)
#     ]),
#     # 5. Fog+Snow
#     A.Compose([
#         CustomCloudLayer(),
#         CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
#         CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36)
#     ]),
#     # 6. Rain+Snow
#     A.Compose([
#         CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
#         CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
#         CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
#         CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
#         CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
#     ]),
#     # 7. Dark
#     A.Compose([
#         A.OneOf([
#             A.GaussianBlur(blur_limit=(9, 11), p=0.5),
#             A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.5),
#         ]),
#         A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.15), contrast_limit=(0.2, 0.2), p=1)
#     ]),
#     # 8. Over-exposure
#     A.Compose([
#         A.RandomBrightnessContrast(brightness_limit=(0, 0.3), contrast_limit=(1.3, 1.6), p=1)
#     ]),
#     # 9. Wind
#     A.Compose([
#         A.MotionBlur(blur_limit=15, p=1)
#     ])
# ]
#
#
# def get_transforms(img_size,
#                    mean=[0.485, 0.456, 0.406],
#                    std=[0.229, 0.224, 0.225]):
#     weather_id = 0
#     val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
#
#                                 # Multi-weather U1652 settings.
#                                 # iaa_weather_list[weather_id],
#
#                                 A.Normalize(mean, std),
#                                 ToTensorV2(),
#                                 ])
#
#     train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
#                                       A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
#
#                                       # Multi-weather U1652 settings.
#                                       # A.OneOf(iaa_weather_list, p=1.0),
#
#                                       A.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.3,
#                                                   always_apply=False, p=0.5),
#                                       # A.OneOf([
#                                       #     A.AdvancedBlur(p=1.0),
#                                       #     A.Sharpen(p=1.0),
#                                       # ], p=0.3),
#                                       # A.OneOf([
#                                       #     A.GridDropout(ratio=0.4, p=1.0),
#                                       #     A.CoarseDropout(max_holes=25,
#                                       #                     max_height=int(0.2 * img_size[0]),
#                                       #                     max_width=int(0.2 * img_size[0]),
#                                       #                     min_holes=10,
#                                       #                     min_height=int(0.1 * img_size[0]),
#                                       #                     min_width=int(0.1 * img_size[0]),
#                                       #                     p=1.0),
#                                       # ], p=0.3),
#                                       # A.RandomRotate90(p=1.0),
#                                       A.Normalize(mean, std),
#                                       ToTensorV2(),
#                                       ])
#
#     train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
#                                         A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
#
#                                         # Multi-weather U1652 settings.
#                                         # A.OneOf(iaa_weather_list, p=1.0),
#
#                                         A.ColorJitter(brightness=0.15, contrast=0.7, saturation=0.3, hue=0.3,
#                                                       always_apply=False, p=0.5),
#                                         # A.OneOf([
#                                         #     A.AdvancedBlur(p=1.0),
#                                         #     A.Sharpen(p=1.0),
#                                         # ], p=0.3),
#                                         # A.OneOf([
#                                         #     A.GridDropout(ratio=0.4, p=1.0),
#                                         #     A.CoarseDropout(max_holes=25,
#                                         #                     max_height=int(0.2 * img_size[0]),
#                                         #                     max_width=int(0.2 * img_size[0]),
#                                         #                     min_holes=10,
#                                         #                     min_height=int(0.1 * img_size[0]),
#                                         #                     min_width=int(0.1 * img_size[0]),
#                                         #                     p=1.0),
#                                         # ], p=0.3),
#                                         A.Normalize(mean, std),
#                                         ToTensorV2(),
#                                         ])
#
#     return val_transforms, train_sat_transforms, train_drone_transforms
