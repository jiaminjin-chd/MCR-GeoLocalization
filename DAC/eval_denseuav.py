import argparse

import torch
from torch.utils.data import DataLoader

from sample4geo.dataset.denseuav_v2 import (
    DenseUAVDatasetEvalV2,
    get_transforms
)

from sample4geo.model import TimmModel

from sample4geo.evaluate.denseuav_v2 import evaluate


# ==========================================================
# Args
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--direction",
        type=str,
        default="D2S",
        choices=["D2S", "S2D"]
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=24
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8
    )

    return parser.parse_args()


# ==========================================================
# Config
# ==========================================================

class Config:
    pass


def build_config(args):

    config = Config()

    config.device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config.gpu_ids = [0]

    config.verbose = True

    config.handcraft_model = True

    config.weight_infonce = 1.0

    config.weight_cls = 0.5

    config.weight_dsa = 0.1

    config.clip_grad = 1.0

    config.scheduler = "cosine"

    # 这里只是为了保持 evaluate() 的 config 接口一致
    config.direction = args.direction

    return config


# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    config = build_config(args)

    print("=" * 70)
    print("DenseUAV V2 Checkpoint Evaluation")
    print("=" * 70)

    print("Checkpoint:", args.checkpoint)
    print("Direction :", args.direction)
    print("Device    :", config.device)

    # ======================================================
    # Dataset paths
    # ======================================================

    if args.direction == "D2S":

        test_query = (
            "/root/autodl-tmp/DenseUAV/"
            "test/query_drone"
        )

        test_gallery = (
            "/root/autodl-tmp/DenseUAV/"
            "test/gallery_satellite"
        )

    else:

        test_query = (
            "/root/autodl-tmp/DenseUAV/"
            "test/gallery_satellite"
        )

        test_gallery = (
            "/root/autodl-tmp/DenseUAV/"
            "test/query_drone"
        )

    gps_test = (
        "/root/autodl-tmp/DenseUAV/"
        "Dense_GPS_test.txt"
    )

    gps_all = (
        "/root/autodl-tmp/DenseUAV/"
        "Dense_GPS_ALL.txt"
    )

    print()
    print("Test query  :", test_query)
    print("Test gallery:", test_gallery)

    # ======================================================
    # Transform
    # ======================================================

    val_transform = get_transforms(
        is_train=False
    )

    # ======================================================
    # Dataset
    # ======================================================

    query_dataset = DenseUAVDatasetEvalV2(

        test_query,

        val_transform

    )

    gallery_dataset = DenseUAVDatasetEvalV2(

        test_gallery,

        val_transform

    )

    query_loader = DataLoader(

        query_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=True

    )

    gallery_loader = DataLoader(

        gallery_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=True

    )

    print()
    print("Query images  :", len(query_dataset))
    print("Gallery images:", len(gallery_dataset))

    # ======================================================
    # Model
    # ======================================================

    print()
    print("=" * 70)
    print("Loading model")
    print("=" * 70)

    model = TimmModel(

        'vit_base_patch16_384',

        num_classes=2256,

        pretrained=False,

        img_size=384,

        local_weight_path=None

    )

    checkpoint = torch.load(

        args.checkpoint,

        map_location="cpu",

        weights_only=False

    )

    model.load_state_dict(
        checkpoint["model"]
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "Unknown")
    )

    print(
        "Checkpoint direction:",
        checkpoint.get("direction", "Unknown")
    )

    print(
        "Checkpoint mask ratio:",
        checkpoint.get("mask_ratio", "Unknown")
    )

    if "metrics" in checkpoint:

        print(
            "Checkpoint saved metrics:",
            checkpoint["metrics"]
        )

    model = model.to(
        config.device
    )

    model.eval()

    # ======================================================
    # Evaluation
    # ======================================================

    print()
    print("=" * 70)
    print(
        f"Evaluating checkpoint as {args.direction}"
    )
    print("=" * 70)

    metrics = evaluate(

        config,

        model,

        query_loader,

        gallery_loader,

        test_query,

        test_gallery,

        gps_test,

        gps_all,

        step=0

    )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print("Checkpoint:", args.checkpoint)

    print("Direction:", args.direction)

    print(
        f"R1  = {metrics['R1']:.6f}"
    )

    print(
        f"R5  = {metrics['R5']:.6f}"
    )

    print(
        f"R10 = {metrics['R10']:.6f}"
    )

    print(
        f"AP  = {metrics['AP']:.6f}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()