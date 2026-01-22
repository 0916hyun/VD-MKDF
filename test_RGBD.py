# test_dataloader_rgbd.py

import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from data.loader_RGBD import (
    TrainLoadDatasetMulti_RGBD,  # import 에러 방지용
    ValLoadDatasetMulti_RGBD,    # import 에러 방지용
    TestLoadDatasetMulti_RGBD
)
from data.weather_utils import make_sample_list

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_samples(root, fold, clear_dir, weather_dirs, seg_dir):
    base = Path(root) / fold
    fold_weather_dirs = {w: str(base / d) for w, d in weather_dirs.items()}
    return make_sample_list(
        root=str(base),
        scene_dir=clear_dir,
        weather_dirs=fold_weather_dirs,
        seg_dir=seg_dir,
    )

def build_dataloaders(args):
    set_seed(1234)
    weather_dirs = {
        "rain":  args.rain_dir,
        "snow":  args.snow_dir,
        "fog":   args.fog_dir,
        "flare": args.flare_dir,
    }

    # test samples
    test_samples = build_samples(
        args.data_root, args.test_fold,
        args.clear_dir, weather_dirs, args.seg_dir
    )

    base_test      = Path(args.data_root) / args.test_fold
    depth_root_test= Path(args.depth_root) / args.test_fold / "norm_16bit"

    test_ds = TestLoadDatasetMulti_RGBD(
        sample_list=test_samples,
        base_dir=base_test,
        depth_root=depth_root_test
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False
    )
    return test_loader

def parse_args():
    parser = argparse.ArgumentParser(description="RGB-D SegFormer: Test 전용 DataLoader")

    # 데이터 경로 및 폴드
    parser.add_argument("--data_root",    default="/content/dataset/")
    parser.add_argument("--train_fold",   default="fold1_train")
    parser.add_argument("--val_fold",     default="fold1_valid")
    parser.add_argument("--test_fold",    default="fold1_test")
    parser.add_argument("--clear_dir",    default="clear")
    parser.add_argument("--seg_dir",      default="seg_label")
    parser.add_argument("--rain_dir",     default="rain")
    parser.add_argument("--snow_dir",      default="snow")
    parser.add_argument("--fog_dir",      default="fog")
    parser.add_argument("--flare_dir",    default="flare")
    parser.add_argument("--depth_root",   default="/content/drive/MyDrive/weather_depth")

    # DataLoader 설정
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--num_workers",  type=int, default=4)

    # 체크포인트 & 결과 경로
    parser.add_argument("--weights_dir",  default="/content/drive/MyDrive/weather_pth/segf_gating_0123")
    parser.add_argument("--results_dir",  default="/content/drive/MyDrive/weather_pth/segf_gating_0123")
    parser.add_argument("--ckpt_loss",    default="best_val_loss.pth")
    parser.add_argument("--ckpt_miou",    default="best_val_miou.pth")

    # 기타
    parser.add_argument("--num_classes",  type=int, default=12)
    parser.add_argument("--ignore_index", type=int, default=11)
    parser.add_argument("--device",       default="cuda")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_loader = build_dataloaders(args)
    print(f"Test loader ready: {len(test_loader.dataset)} samples")


