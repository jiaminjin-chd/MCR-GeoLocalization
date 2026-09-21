import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, help='experiment name')
args = parser.parse_args()

model_dir = os.path.join('./model', args.name)
pth_files = glob.glob(os.path.join(model_dir, 'net_*.pth'))

def epoch_num(f):
    basename = os.path.basename(f)
    if basename == 'net_last.pth':
        return float('inf')
    try:
        return int(basename.split('_')[1].split('.')[0])
    except:
        return -1

pth_files.sort(key=epoch_num)
if pth_files:
    # 保留最后一个（最大epoch或 net_last.pth）
    for f in pth_files[:-1]:
        os.remove(f)
        print(f"Removed {f}")
    print(f"Kept {pth_files[-1]}")
else:
    print("No .pth files found.")