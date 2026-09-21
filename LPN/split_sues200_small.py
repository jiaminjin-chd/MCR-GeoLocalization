import os
import random
import shutil


def split_sues200_small(src_root, dst_root, ratios, seed=42):
    """
    按高度分别采样：保证每个高度子文件夹内的图像按比例保留。
    src_root: 原始数据集根目录（如 /root/autodl-tmp/SUES-200-512x512）
    dst_root: 输出根目录（如 /root/autodl-tmp/SUES-200-512x512_small）
    ratios:   {'quarter':0.25, 'half':0.5, 'three_quarter':0.75}
    """
    random.seed(seed)
    sat_src = os.path.join(src_root, 'satellite-view')
    drone_src = os.path.join(src_root, 'drone_view_512')

    # 获取所有 ID（卫星视图和无人机视图共有的）
    ids = [d for d in os.listdir(sat_src) if os.path.isdir(os.path.join(sat_src, d))]
    ids.sort()
    print(f"找到 {len(ids)} 个 ID")

    for name, ratio in ratios.items():
        print(f"\n===== 生成 {name} (比例 {ratio}) =====")
        dst_sub = os.path.join(dst_root, name)
        os.makedirs(dst_sub, exist_ok=True)

        # 1. 卫星视图完整复制（每个ID一张图）
        sat_dst = os.path.join(dst_sub, 'satellite-view')
        if os.path.exists(sat_dst):
            shutil.rmtree(sat_dst)
        shutil.copytree(sat_src, sat_dst)
        print(f"  卫星视图已复制，共 {len(ids)} 个ID")

        # 2. 无人机视图按高度采样
        drone_dst = os.path.join(dst_sub, 'drone_view_512')
        os.makedirs(drone_dst, exist_ok=True)
        total_sampled = 0

        for id_ in ids:
            id_src = os.path.join(drone_src, id_)
            if not os.path.isdir(id_src):
                print(f"  警告: ID {id_} 没有无人机目录，跳过")
                continue

            # 遍历高度子文件夹
            for height in os.listdir(id_src):
                height_src = os.path.join(id_src, height)
                if not os.path.isdir(height_src):
                    continue
                # 收集该高度下所有图像
                images = [f for f in os.listdir(height_src) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if not images:
                    continue
                # 计算采样数量（至少1张）
                num_keep = max(1, int(len(images) * ratio))
                sampled = random.sample(images, num_keep)
                # 复制到目标目录（保持目录结构）
                for img in sampled:
                    src_file = os.path.join(height_src, img)
                    dst_file = os.path.join(drone_dst, id_, height, img)
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                total_sampled += len(sampled)

        # 统计并打印
        total_drone_files = sum(len(files) for _, _, files in os.walk(drone_dst))
        print(f"  无人机视图采样完成，共 {total_drone_files} 张图片（原总数比例约 {ratio:.0%}）")


if __name__ == '__main__':
    src_root = '/root/autodl-tmp/SUES-200-512x512'  # 修改为你的原始数据路径
    dst_root = '/root/autodl-tmp/SUES-200-512x512_small'  # 输出路径
    ratios = {'quarter': 0.25, 'half': 0.5, 'three_quarter': 0.75}
    split_sues200_small(src_root, dst_root, ratios, seed=42)
    print("\n所有小样本数据集已生成完毕！")