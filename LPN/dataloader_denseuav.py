import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def parse_coord(coord_str):
    match = re.search(r'[-+]?\d*\.?\d+', coord_str)
    if match:
        return float(match.group())
    return 0.0

def load_coords(root_dir):
    coord_map = {}
    coord_files = ['Dense_GPS_ALL.txt', 'Dense_GPS_test.txt']
    for filename in coord_files:
        coord_file = os.path.join(root_dir, filename)
        if not os.path.exists(coord_file):
            print(f"Warning: Coordinate file {coord_file} not found.")
            continue
        with open(coord_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    img_path = parts[0]
                    x = parse_coord(parts[1])
                    y = parse_coord(parts[2])
                    coord_map[img_path] = (x, y)
        print(f"Loaded {len(coord_map)} coordinates from {filename}")
    return coord_map  # 只返回字典

class DenseUAVSingleDataset(Dataset):
    def __init__(self, root_dir, mode='query', transform=None, coords=None, class_coord=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.coords = coords if coords is not None else {}
        self.class_coord = class_coord if class_coord is not None else {}
        self.samples = []  # (img_path, label, rel_path)
        if mode == 'query':
            base_dir = os.path.join(root_dir, 'test', 'query_drone')
            prefix = 'test/query_drone'
        elif mode == 'gallery':
            base_dir = os.path.join(root_dir, 'test', 'gallery_satellite')
            prefix = 'test/gallery_satellite'
        else:
            raise ValueError("mode must be 'query' or 'gallery'")
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Directory not found: {base_dir}")
        classes = sorted(os.listdir(base_dir))
        for cls_id, cls in enumerate(classes):
            class_dir = os.path.join(base_dir, cls)
            for img_name in os.listdir(class_dir):
                rel_path = f"{prefix}/{cls}/{img_name}"
                self.samples.append((os.path.join(class_dir, img_name), cls_id, rel_path))
        print(f"Loaded {len(self.samples)} samples for {mode}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, rel_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # 获取坐标：优先从文件路径匹配，其次从类别ID匹配
        coord = self.coords.get(rel_path, None)
        if coord is None:
            coord = self.class_coord.get(label, (0.0, 0.0))
        return img, label, np.array(coord, dtype=np.float32)

def get_denseuav_loaders(root_dir, batch_size=32, num_workers=4):
    coord_map, class_coord = load_coords(root_dir)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 为了保持兼容，我们仍然保留训练集加载（但这里只用于评估，所以可以不加载训练集）
    # 但 get_denseuav_loaders 可能被训练脚本调用，所以保留
    # 为了简化，我们只创建 query 和 gallery loader
    query_dataset = DenseUAVSingleDataset(root_dir, 'query', test_transform, coord_map, class_coord)
    gallery_dataset = DenseUAVSingleDataset(root_dir, 'gallery', test_transform, coord_map, class_coord)

    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

    # 为了兼容训练脚本，我们返回虚拟的训练loader和类别数
    # 但训练脚本不使用这些返回值，所以我们可以返回 None 或 dummy
    # 但为了保持原接口，我们简单返回 (None, query_loader, gallery_loader, 2256)
    # 注意：训练脚本如果调用 get_denseuav_loaders 期望返回 train_loader, query_loader, gallery_loader, num_classes
    # 如果训练脚本不需要 query 和 gallery，可能不会调用这个函数，所以我们可以安全返回。
    # 但为了通用性，我们仍然返回4个值。
    # 我们可以创建一个 dummy train_loader
    dummy_train = None
    return dummy_train, query_loader, gallery_loader, 2256  # 类别数固定为2256