import os
from PIL import Image
from torch.utils.data import Dataset

class SUESDataset(Dataset):
    def __init__(self, root_dir, transform_sat=None, transform_drone=None, train_ids=None, global_label_map=None):
        """
        root_dir: SUES-200-512x512 根目录
        transform_sat: 卫星图的transform
        transform_drone: 无人机图的transform
        train_ids: 用于训练的ID列表（None表示使用所有ID）
        global_label_map: 可选，全局标签映射 {id: label}，如果提供则使用它，否则自动生成连续标签
        """
        self.root = root_dir
        self.transform_sat = transform_sat
        self.transform_drone = transform_drone

        sat_dir = os.path.join(root_dir, 'satellite-view')
        drone_dir = os.path.join(root_dir, 'drone_view_512')

        # 获取所有ID（子文件夹名）
        all_ids = [d for d in os.listdir(sat_dir) if os.path.isdir(os.path.join(sat_dir, d))]
        if train_ids is not None:
            all_ids = [i for i in all_ids if i in train_ids]
        self.ids = sorted(all_ids)

        # 确定标签映射
        if global_label_map is not None:
            label_map = global_label_map
        else:
            label_map = {id_: idx for idx, id_ in enumerate(self.ids)}

        # 构建样本对列表 [(sat_path, drone_path, label), ...]
        self.samples = []
        for id_ in self.ids:
            if id_ not in label_map:
                print(f"Warning: ID {id_} not found in label_map, skipping")
                continue
            label = label_map[id_]
            sat_path = os.path.join(sat_dir, id_, '0.png')
            if not os.path.exists(sat_path):
                print(f"Warning: satellite image not found for ID {id_}: {sat_path}")
                continue
            drone_id_dir = os.path.join(drone_dir, id_)
            if not os.path.exists(drone_id_dir):
                print(f"Warning: drone directory not found for ID {id_}: {drone_id_dir}")
                continue

            # 递归查找所有图片文件
            drone_files = []
            for root, dirs, files in os.walk(drone_id_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        drone_files.append(os.path.join(root, f))

            for drone_path in drone_files:
                self.samples.append((sat_path, drone_path, label))

        print(f"Loaded {len(self.samples)} sample pairs for {len(self.ids)} IDs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sat_path, drone_path, label = self.samples[idx]
        sat_img = Image.open(sat_path).convert('RGB')
        drone_img = Image.open(drone_path).convert('RGB')
        if self.transform_sat:
            sat_img = self.transform_sat(sat_img)
        if self.transform_drone:
            drone_img = self.transform_drone(drone_img)
        return sat_img, drone_img, label