import os
import shutil
import argparse
import yaml

def convert_subset(lpn_root, dac_root, full_test_root, indexs_yaml, heights=[150,200,250,300]):
    """
    lpn_root: LPN小样本数据集根目录（例如 .../SUES-200-512x512_small/quarter）
    dac_root: 输出的DAC格式根目录（例如 .../SUES-200-512x512_small_dac/quarter）
    full_test_root: 全样本DAC格式根目录（包含 Testing/ 文件夹）
    indexs_yaml: 官方训练ID列表文件路径
    """
    # 加载训练ID
    with open(indexs_yaml, 'r') as f:
        data = yaml.safe_load(f)
        train_ids = set(data['index'])
    print(f"Loaded {len(train_ids)} training IDs")

    # 创建输出目录
    training_dir = os.path.join(dac_root, 'Training')
    os.makedirs(training_dir, exist_ok=True)

    # 卫星图源目录
    sat_src = os.path.join(lpn_root, 'satellite-view')
    # 无人机图源目录
    drone_src = os.path.join(lpn_root, 'drone_view_512')
    # 获取源数据中的所有 ID
    all_ids = [d for d in os.listdir(sat_src) if os.path.isdir(os.path.join(sat_src, d))]
    print(f"Found {len(all_ids)} IDs in source data")

    for height in heights:
        print(f"Processing height {height}m...")
        sat_dst = os.path.join(training_dir, str(height), 'satellite')
        drone_dst = os.path.join(training_dir, str(height), 'drone')
        os.makedirs(sat_dst, exist_ok=True)
        os.makedirs(drone_dst, exist_ok=True)

        for id_ in all_ids:
            if id_ not in train_ids:
                continue  # 跳过测试集 ID
            # 卫星图
            src_sat = os.path.join(sat_src, id_, '0.png')
            if os.path.exists(src_sat):
                dst_sat_dir = os.path.join(sat_dst, id_)
                os.makedirs(dst_sat_dir, exist_ok=True)
                dst_sat = os.path.join(dst_sat_dir, '0.png')
                if not os.path.exists(dst_sat):
                    try:
                        os.link(src_sat, dst_sat)   # 硬链接节省空间
                    except OSError:
                        shutil.copy2(src_sat, dst_sat)
            else:
                print(f"Warning: missing satellite for {id_}")

            # 无人机图
            src_drone_height = os.path.join(drone_src, id_, str(height))
            if os.path.isdir(src_drone_height):
                dst_drone_dir = os.path.join(drone_dst, id_)
                os.makedirs(dst_drone_dir, exist_ok=True)
                for img in os.listdir(src_drone_height):
                    src_img = os.path.join(src_drone_height, img)
                    dst_img = os.path.join(dst_drone_dir, img)
                    if not os.path.exists(dst_img):
                        try:
                            os.link(src_img, dst_img)
                        except OSError:
                            shutil.copy2(src_img, dst_img)

    # 创建软链接指向全样本的 Testing 目录
    test_link = os.path.join(dac_root, 'Testing')
    if not os.path.exists(test_link):
        os.symlink(full_test_root, test_link)
        print(f"Created symlink: {test_link} -> {full_test_root}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lpn_root', required=True, help='LPN小样本数据集根目录')
    parser.add_argument('--dac_root', required=True, help='输出DAC格式根目录')
    parser.add_argument('--full_test_root', required=True, help='全样本DAC格式根目录（包含 Testing/）')
    parser.add_argument('--indexs_yaml', required=True, help='indexs.yaml 文件路径')
    args = parser.parse_args()
    convert_subset(args.lpn_root, args.dac_root, args.full_test_root, args.indexs_yaml)
    print("转换完成")