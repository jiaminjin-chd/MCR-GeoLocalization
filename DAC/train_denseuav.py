import os
import argparse
import json
import datetime

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sample4geo.dataset.denseuav_v2 import (
    DenseUAVDatasetTrainV2,
    DenseUAVDatasetEvalV2,
    get_transforms
)

from sample4geo.model import TimmModel

from sample4geo.evaluate.denseuav_v2 import evaluate

from sample4geo.trainer import train

from sample4geo.loss.loss import InfoNCE
from sample4geo.loss.DSA_loss import DSA_loss

def parse_args():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.0
    )


    parser.add_argument(
        "--direction",
        type=str,
        default="D2S"
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=24
    )


    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5
    )


    parser.add_argument(
        "--num_workers",
        type=int,
        default=8
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_denseuav"
    )


    # 新增
    parser.add_argument(
        "--resume",
        type=str,
        default=None
    )


    return parser.parse_args()

class Config:
    pass


def build_config(args):

    config = Config()

    config.device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config.gpu_ids=[0]

    config.verbose=True

    config.handcraft_model=True

    config.weight_infonce=1.0

    config.weight_cls=0.5

    config.weight_dsa=0.1


    config.clip_grad=1.0


    config.scheduler="cosine"


    config.mask_ratio=args.mask_ratio

    config.direction=args.direction


    return config




def main():


    args=parse_args()

    config=build_config(args)

    run_time=datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    save_dir=os.path.join(

        "./results_denseuav",

        "full-denseuav",

        args.direction,

        f"mask_{args.mask_ratio:.2f}",

        run_time

    )

    os.makedirs(
        save_dir,
        exist_ok=True)
    

    log_file=os.path.join(
        save_dir,
        "train.log"
    )

    print("="*70)

    print(
        "Save directory:",
        save_dir
    )

    print("="*70)

    # 保存配置

    with open(
        os.path.join(save_dir,"config.json"),
        "w"
    ) as f:

        json.dump(

            {

            "model":
            "vit_base_patch16_384",

            "dataset":
            "DenseUAV",

            "direction":
            args.direction,

            "mask_ratio":
            args.mask_ratio,

            "epochs":
            args.epochs,

            "batch_size":
            args.batch_size,

            "lr":
            args.lr,

            "time":
            run_time

            },

            f,

            indent=4

        )



    print("="*70)

    print(
        "DenseUAV ViT baseline"
    )

    print("="*70)


    if args.direction=="D2S":


        train_query="/root/autodl-tmp/DenseUAV/train/drone"

        train_gallery="/root/autodl-tmp/DenseUAV/train/satellite"


        test_query="/root/autodl-tmp/DenseUAV/test/query_drone"

        test_gallery="/root/autodl-tmp/DenseUAV/test/gallery_satellite"

    else:

        train_query="/root/autodl-tmp/DenseUAV/train/satellite"

        train_gallery="/root/autodl-tmp/DenseUAV/train/drone"


        test_query="/root/autodl-tmp/DenseUAV/test/gallery_satellite"

        test_gallery="/root/autodl-tmp/DenseUAV/test/query_drone"


    gps_test="/root/autodl-tmp/DenseUAV/Dense_GPS_test.txt"

    gps_all="/root/autodl-tmp/DenseUAV/Dense_GPS_ALL.txt"

    print(train_query)

    print(train_gallery)

    print(test_query)

    print(test_gallery)

    # ==========================
    # Dataset
    # ==========================


    train_transform_query,train_transform_gallery = get_transforms(
        is_train=True
    )

    train_dataset=DenseUAVDatasetTrainV2(

        train_query,

        train_gallery,

        train_transform_query,

        train_transform_gallery,
        mask_ratio=args.mask_ratio

    )


    train_loader=DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=args.num_workers,

        pin_memory=True,

        drop_last=True

    )


    val_transform=get_transforms(
        is_train=False
    )

    query_dataset=DenseUAVDatasetEvalV2(

        test_query,

        val_transform

    )


    gallery_dataset=DenseUAVDatasetEvalV2(

        test_gallery,

        val_transform

    )


    query_loader=DataLoader(

        query_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=True

    )


    gallery_loader=DataLoader(

        gallery_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=True

    )


    print(
        "Query images:",
        len(query_dataset)
    )

    print(
        "Gallery images:",
        len(gallery_dataset)
    )


    # ==========================
    # Model
    # ==========================


    model=TimmModel(
        'vit_base_patch16_384',
        num_classes=2256,
        pretrained=False,
        img_size=384,
        local_weight_path="/root/autodl-tmp/DAC-main/vit_base_patch16_384.safetensors"
    )


    checkpoint=None


    if args.resume is not None:

        print(
            "Loading checkpoint:",
            args.resume
        )

        checkpoint=torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False
        )


        model.load_state_dict(
            checkpoint["model"]
        )


        print(
            "Loaded epoch:",
            checkpoint["epoch"]
        )


    model=model.to(config.device)
    
    best_ap=0.0

    start_epoch=1



    if checkpoint is not None:


        start_epoch = checkpoint["epoch"] + 1


        if "metrics" in checkpoint:

            best_ap=checkpoint["metrics"]["AP"]


        print(
            "Continue from epoch:",
            start_epoch
        )


    optimizer=AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )


    total_epochs=max(args.epochs,50)
        
    scheduler=CosineAnnealingLR(
        optimizer,
        T_max=total_epochs*len(train_loader)
    )


    scaler=GradScaler()



    if checkpoint is not None:


        if "optimizer" in checkpoint:

            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )


        if "scheduler" in checkpoint:

            scheduler.load_state_dict(
                checkpoint["scheduler"]
            )


        if "scaler" in checkpoint:

            scaler.load_state_dict(
                checkpoint["scaler"]
            )


        print(
            "Optimizer state restored"
        )


    loss_functions={}



    ce=nn.CrossEntropyLoss()


    loss_functions["infoNCE"]=InfoNCE(

        ce,

        device=config.device

    )

    loss_functions["DSA_loss"]=DSA_loss(

        ce,

        device=config.device

    )
    print()
    print("="*70)
    print("Initial Evaluation")
    print("="*70)


    initial_metrics = evaluate(

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


    print(
        "Initial metrics:",
        initial_metrics
    )



    best_metrics=None



    # ==========================
    # Training loop
    # ==========================

    for epoch in range(start_epoch, args.epochs + 1):


        print()

        print(
            "="*20,
            "Epoch",
            epoch,
            "="*20
        )



        train_loss=train(

            config,

            model,

            train_loader,

            loss_functions,

            optimizer,

            epoch,

            len(train_loader),

            scheduler=scheduler,

            scaler=scaler

        )


        print(
            "Train loss:",
            train_loss
        )



        print()

        print(
            "Evaluation after epoch",
            epoch
        )



        metrics=evaluate(

            config,

            model,

            query_loader,

            gallery_loader,

            test_query,

            test_gallery,

            gps_test,

            gps_all,

            step=epoch

        )


        print(
            "Epoch",
            epoch,
            metrics
        )



        # ==========================
        # 保存epoch日志
        # ==========================

        with open(

            log_file,

            "a"

        ) as f:


            f.write(

                f"Epoch {epoch}, "

                f"R1={metrics['R1']:.4f}, "

                f"R5={metrics['R5']:.4f}, "

                f"R10={metrics['R10']:.4f}, "

                f"AP={metrics['AP']:.4f}\n"

            )



        # ==========================
        # 保存最佳模型
        # ==========================


        if metrics["AP"] > best_ap:


            best_ap=metrics["AP"]

            best_metrics=metrics



            save_path=os.path.join(

                save_dir,

                "best.pth"

            )


            torch.save(
            {

            "epoch":epoch,


            "model":
            model.state_dict(),


            "optimizer":
            optimizer.state_dict(),


            "scheduler":
            scheduler.state_dict(),


            "scaler":
            scaler.state_dict(),


            "metrics":
            metrics,


            "direction":
            args.direction,


            "mask_ratio":
            args.mask_ratio

            },
            save_path
            )


            print(
                "Saved best model:",
                save_path
            )



    # ==========================
    # Training finished
    # ==========================


    print("="*70)

    print(
        "Training Finished"
    )

    print("="*70)


    print(
        "Best AP:",
        best_ap
    )


    # ==========================
    # 保存最终结果
    # ==========================


    if best_metrics is None:

        best_metrics=initial_metrics



    result_file=os.path.join(

        save_dir,

        "result.txt"

    )


    with open(

        result_file,

        "w"

    ) as f:


        f.write(
            "dataset=full-denseuav\n"
        )


        f.write(
            "model=vit_base_patch16_384\n"
        )


        f.write(
            f"direction={args.direction}\n"
        )


        f.write(
            f"mask_ratio={args.mask_ratio}\n"
        )


        f.write(
            f"epochs={args.epochs}\n"
        )


        f.write(
            f"batch_size={args.batch_size}\n"
        )


        f.write(
            f"lr={args.lr}\n"
        )


        f.write(
            "\nBest Result\n"
        )


        f.write(
            f"R1={best_metrics['R1']}\n"
        )


        f.write(
            f"R5={best_metrics['R5']}\n"
        )


        f.write(
            f"R10={best_metrics['R10']}\n"
        )


        f.write(
            f"AP={best_metrics['AP']}\n"
        )



    print(
        "Saved result:",
        result_file
    )


if __name__=="__main__":

    main()