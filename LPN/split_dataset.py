import os
import random
import shutil
from collections import defaultdict

def split_dataset(src_root, dst_root, ratios, seed=42):
    """
    src_root: 原始训练集根目录，包含 satellite/, drone/, street/, google/ 子文件夹
    dst_root: 输出根目录，会创建 train_quarter, train_half, train_three_quarter
    ratios: 字典，{'quarter':0.25, 'half':0.5, 'three_quarter':0.75}
    """
    random.seed(seed)
    for view in ['satellite', 'drone', 'street', 'google']:
        view_path = os.path.join(src_root, view)
        if not os.path.exists(view_path):
            continue
        classes = [d for d in os.listdir(view_path) if os.path.isdir(os.path.join(view_path, d))]
        for cls in classes:
            cls_path = os.path.join(view_path, cls)
            images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
            num = len(images)
            for name, ratio in ratios.items():
                target_num = max(1, int(num * ratio))
                sampled = random.sample(images, target_num)
                dst_cls_dir = os.path.join(dst_root, name, view, cls)
                os.makedirs(dst_cls_dir, exist_ok=True)
                for img in sampled:
                    shutil.copy(os.path.join(cls_path, img), os.path.join(dst_cls_dir, img))
        print(f"Finished {view}")

if __name__ == '__main__':
    src = '/root/autodl-tmp/University-Release/train'
    dst = '/root/autodl-tmp/University-Release'
    ratios = {'quarter':0.25, 'half':0.5, 'three_quarter':0.75}
    split_dataset(src, dst, ratios, seed=42)
    print("Done!")