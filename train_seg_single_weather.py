# train_seg_single_weather.py
import argparse
import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from data.loader import (
    TrainLoadDatasetMulti,
    ValLoadDatasetMulti,
    TestLoadDatasetMulti
)
from data.weather_utils import make_sample_list
from models.make_model import make_model
from seg_trainer import SegmentationTrainer


def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args):
    # 선택된 날씨만 사용
    weather = args.weather
    weather_dirs = {weather: getattr(args, f"{weather}_dir")}

    # sample list 생성
    train_samples = make_sample_list(
        root=str(Path(args.data_root) / args.train_fold),
        scene_dir=args.clear_dir,
        weather_dirs={w: str(Path(args.data_root) / args.train_fold / d) for w, d in weather_dirs.items()},
        seg_dir=args.seg_dir
    )
    val_samples = make_sample_list(
        root=str(Path(args.data_root) / args.val_fold),
        scene_dir=args.clear_dir,
        weather_dirs={w: str(Path(args.data_root) / args.val_fold / d) for w, d in weather_dirs.items()},
        seg_dir=args.seg_dir
    )
    test_samples = make_sample_list(
        root=str(Path(args.data_root) / args.test_fold),
        scene_dir=args.clear_dir,
        weather_dirs={w: str(Path(args.data_root) / args.test_fold / d) for w, d in weather_dirs.items()},
        seg_dir=args.seg_dir
    )

    # Dataset 생성 (RGB only)
    train_ds = TrainLoadDatasetMulti(train_samples, crop_size=args.crop_size)
    val_ds   = ValLoadDatasetMulti(val_samples,   crop_size=args.crop_size)
    test_ds  = TestLoadDatasetMulti(test_samples)

    # DataLoader 생성
    train_loader = DataLoader(train_ds, batch_size=args.train_bs, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.eval_bs, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def main(args):
    set_seed()
    train_loader, val_loader, test_loader = build_dataloaders(args)

    # 모델 생성 (예: SegFormer B5)
    model = make_model(
        model_name='segformer_b5',
        n_classes=args.num_classes,
        device=args.device
    )

    # Trainer 초기화
    trainer = SegmentationTrainer(
        model=model,
        save_root=args.save_root,
        weights_dir=args.weights_dir,
        results_dir=args.results_dir,
        device=args.device,
        lr=args.lr,
        num_epochs=args.num_epochs,
        n_classes=args.num_classes,
        ignore_index=args.ignore_index
    )

    # 학습 및 검증
    trainer.train(train_loader, val_loader, test_loader)
    # 테스트 수행
    trainer.test(test_loader)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Single-weather RGB-only Segmentation")
    # 데이터 폴더
    p.add_argument("--data_root",   default="/content/dataset/")
    p.add_argument("--train_fold",  default="fold2_train")
    p.add_argument("--val_fold",    default="fold2_valid")
    p.add_argument("--test_fold",   default="fold2_test")
    p.add_argument("--clear_dir",   default="clear")
    p.add_argument("--seg_dir",     default="seg_label")
    p.add_argument("--rain_dir",    default="rain")
    p.add_argument("--snow_dir",    default="snow")
    p.add_argument("--fog_dir",     default="fog")
    p.add_argument("--flare_dir",   default="flare")
    # 분류할 날씨 선택
    p.add_argument("--weather",      choices=["rain","snow","fog","flare"], default="flare",
                   help="학습에 사용할 날씨 종류")
    # 하이퍼파라미터
    p.add_argument(
    "--crop_size",
    nargs=2,                 # 두 값(H, W)
    type=int,
    metavar=("H","W"),
    default=(288, 960),      # 원하는 기본값
    help="Crop size as H W, e.g., 288 960",
)
    p.add_argument("--train_bs",    type=int, default=4)
    p.add_argument("--eval_bs",     type=int, default=1)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_epochs",  type=int, default=400)
    p.add_argument("--num_classes", type=int, default=12)
    p.add_argument("--ignore_index",type=int, default=11)

    # p.add_argument("--encoder_name", default="resnet50", help = "UNet++ encoder name (smp)")
    # p.add_argument("--encoder_weights", default="imagenet", help = "UNet++ encoder weights (smp)")
  
    # 모델 설정
    # p.add_argument("--model_name",  default="unetpp")
    # p.add_argument("--variant",     default="")
    # 저장 경로
    p.add_argument("--save_root",   default="/content/drive/MyDrive/weather_pthweather_seg_single2")
    p.add_argument("--weights_dir", default="/content/drive/MyDrive/weather_pth/weather_seg_single2/flare2/weights")
    p.add_argument("--results_dir", default="/content/drive/MyDrive/weather_pth/weather_seg_single2/flare2/results")
    p.add_argument("--device",      default="cuda")
    args = p.parse_args()
    main(args)


