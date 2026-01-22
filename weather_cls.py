# train_weather_cls_mobilenet.py

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2

from data.loader_RGBD import (
    TrainLoadDatasetMulti_RGBD,
    ValLoadDatasetMulti_RGBD,
    TestLoadDatasetMulti_RGBD
)
from data.weather_utils import make_sample_list
from weather_trainer import WeatherClassificationTrainer


def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args):
    # 날씨별 디렉토리 매핑
    weather_dirs = {
        "rain":  args.rain_dir,
        "snow":  args.snow_dir,
        "fog":   args.fog_dir,
        "flare": args.flare_dir,
    }

    # sample_list 생성
    train_samples = make_sample_list(
        root=os.path.join(args.data_root, args.train_fold),
        scene_dir=args.clear_dir,
        weather_dirs={w: os.path.join(args.data_root, args.train_fold, d)
                      for w, d in weather_dirs.items()},
        seg_dir=args.seg_dir,
    )
    val_samples = make_sample_list(
        root=os.path.join(args.data_root, args.val_fold),
        scene_dir=args.clear_dir,
        weather_dirs={w: os.path.join(args.data_root, args.val_fold, d)
                      for w, d in weather_dirs.items()},
        seg_dir=args.seg_dir,
    )
    test_samples = make_sample_list(
        root=os.path.join(args.data_root, args.test_fold),
        scene_dir=args.clear_dir,
        weather_dirs={w: os.path.join(args.data_root, args.test_fold, d)
                      for w, d in weather_dirs.items()},
        seg_dir=args.seg_dir,
    )

    # RGB, Depth 경로 함수
    base_dir = lambda fold: os.path.join(args.data_root, fold)
    depth_dir = lambda fold: os.path.join(args.depth_root, fold, "norm_16bit")

    # Dataset 생성
    train_ds = TrainLoadDatasetMulti_RGBD(
        train_samples, args.crop_size,
        base_dir(args.train_fold), depth_dir(args.train_fold)
    )
    val_ds = ValLoadDatasetMulti_RGBD(
        val_samples, args.crop_size,
        base_dir(args.val_fold), depth_dir(args.val_fold)
    )
    test_ds = TestLoadDatasetMulti_RGBD(
        test_samples,
        base_dir(args.test_fold), depth_dir(args.test_fold)
    )

    # DataLoader 생성
    train_loader = DataLoader(train_ds, batch_size=args.train_bs, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.eval_bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def main(args):
    set_seed()

    train_loader, val_loader, test_loader = build_dataloaders(args)

    # MobileNetV2 모델 설정
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 4)  # 4개의 날씨 클래스
    model = model.to(args.device)

    # Trainer 초기화
    trainer = WeatherClassificationTrainer(
        model=model,
        save_dir=args.save_dir,
        device=args.device,
        lr=args.lr,
        num_epochs=args.num_epochs
    )

    # 학습 & 검증
    trainer.train(train_loader, val_loader)
    # 테스트(best_val_loss, best_val_acc 두 모델 모두)
    trainer.test(test_loader)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Weather Classification with MobileNetV2")
    p.add_argument("--data_root",  default="/content/dataset/")
    p.add_argument("--depth_root", default="/content/drive/MyDrive/weather_depth2")
    p.add_argument("--train_fold", default="fold2_train")
    p.add_argument("--val_fold",   default="fold2_valid")
    p.add_argument("--test_fold",  default="fold2_test")
    p.add_argument("--clear_dir",  default="clear")
    p.add_argument("--seg_dir",    default="seg_label")
    p.add_argument("--rain_dir",   default="rain")
    p.add_argument("--snow_dir",   default="snow")
    p.add_argument("--fog_dir",    default="fog")
    p.add_argument("--flare_dir",  default="flare")
    p.add_argument("--crop_size",  type=int,   default=352)
    p.add_argument("--train_bs",   type=int,   default=32)
    p.add_argument("--eval_bs",    type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int,   default=10)
    p.add_argument("--save_dir",   default="/content/drive/MyDrive/weather_pth/2classifier2/")
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    main(args)
