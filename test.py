# test_dataloader.py

import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from data.loader import (
    TrainLoadDatasetMulti,  # import 에러 방지용
    ValLoadDatasetMulti,    # import 에러 방지용
    TestLoadDatasetMulti
)
from data.weather_utils import make_sample_list

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_samples(root, fold, clear_dir, seg_dir, weather_dirs):
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

    # train/val 호출만 (실제로는 쓰이지 않음)
    _ = build_samples(args.data_root, args.train_fold, args.clear_dir, args.seg_dir, weather_dirs)
    _ = build_samples(args.data_root, args.val_fold,   args.clear_dir, args.seg_dir, weather_dirs)
    test_samples = build_samples(args.data_root, args.test_fold,  args.clear_dir, args.seg_dir, weather_dirs)

    test_ds = TestLoadDatasetMulti(test_samples)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers - 2),
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=False
    )
    return test_loader

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-weather UNet: Test 전용 DataLoader")

    # 데이터 경로 및 폴드
    parser.add_argument("--data_root",    default="/content/dataset/")
    parser.add_argument("--train_fold",   default="fold1_train")
    parser.add_argument("--val_fold",     default="fold1_valid")
    parser.add_argument("--test_fold",    default="fold1_test")
    parser.add_argument("--clear_dir",    default="clear")
    parser.add_argument("--seg_dir",      default="seg_label")
    parser.add_argument("--rain_dir",     default="rain")
    parser.add_argument("--snow_dir",     default="snow")
    parser.add_argument("--fog_dir",      default="fog")
    parser.add_argument("--flare_dir",    default="flare")

    # DataLoader 설정
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--num_workers",  type=int, default=4)

    # 체크포인트 & 테스트 설정
    parser.add_argument("--weights_dir",  default="/content/drive/MyDrive/weather_pth/kitti_hgformer_new_1/")
    parser.add_argument("--results_dir",  default="/content/drive/MyDrive/weather_pth/kitti_hgformer_new_1/images")
    parser.add_argument("--ckpt_loss",    default="best_val_loss.pth")
    parser.add_argument("--ckpt_miou",    default="best_val_miou.pth")
    parser.add_argument("--num_classes",  type=int, default=12)
    parser.add_argument("--ignore_index", type=int, default=11)
    parser.add_argument("--device",       default="cuda")
    # HGFormer configuration (mirrors train.py options)
    parser.add_argument(
        "--model_variant",
        default="",
        help=(
            "Optional model variant hint. When using HGFormer, known presets will "
            "automatically populate --hgformer_config if it is not provided."
        ),
    )
    parser.add_argument(
        "--hgformer_config",
        dest="hgformer_cfg",
        default="",
        help=(
            "Path to the HGFormer Detectron2 config. If omitted, a preset is selected "
            "based on --model_variant when possible."
        ),
    )
    parser.add_argument("--hgformer_weights", default="")
    parser.add_argument("--hgformer_opts", nargs="*", default=None,
                        help="Additional Detectron2 config overrides for HGFormer.")

    # Visualization options
    parser.set_defaults(export_color_maps=True, export_gradcam=False, gradcam_verbose=False)
    parser.add_argument("--no_color_maps", action="store_false", dest="export_color_maps",
                        help="Disable exporting colorized segmentation maps")
    parser.add_argument("--no_gradcam", action="store_false", dest="export_gradcam",
                        help="Disable Grad-CAM export")
    parser.add_argument("--gradcam_target_layer", default="",
                        help="Dotted path to the target layer used for Grad-CAM (auto-detected if empty)")
    parser.add_argument(
        "--gradcam_classes",
        nargs="*",
        default=["Pole"],
        help=(
            "Class names or indices to export Grad-CAM overlays for."
            " Defaults to ['Pole']; pass nothing to export all classes."
        ),
    )
    parser.add_argument("--gradcam_alpha", type=float, default=0.6,
                        help="Blend weight for Grad-CAM overlays (higher means stronger heatmap colors)")
    parser.add_argument("--gradcam_gamma", type=float, default=0.5,
                        help="Gamma correction applied to Grad-CAM heatmaps ( >1 amplifies strong activations)")

    return parser.parse_args()
