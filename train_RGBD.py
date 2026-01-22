import argparse
import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from data.loader_RGBD import (
    TrainLoadDatasetMulti_RGBD,
    ValLoadDatasetMulti_RGBD,
    TestLoadDatasetMulti_RGBD
)
from data.weather_utils import make_sample_list
from RGBD_trainer import Trainer
from models.make_model import make_model


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
    weather_dirs = {
        "rain":  args.rain_dir,
        "snow":  args.snow_dir,
        "fog":   args.fog_dir,
        "flare": args.flare_dir,
    }

    train_samples = build_samples(
        args.data_root, args.train_fold, args.clear_dir, args.seg_dir, weather_dirs
    )
    val_samples = build_samples(
        args.data_root, args.val_fold, args.clear_dir, args.seg_dir, weather_dirs
    )
    test_samples = build_samples(
        args.data_root, args.test_fold, args.clear_dir, args.seg_dir, weather_dirs
    )

    base_train      = Path(args.data_root) / args.train_fold
    depth_root_train= Path(args.depth_root) / args.train_fold / "norm_16bit"

    base_val        = Path(args.data_root) / args.val_fold
    depth_root_val  = Path(args.depth_root) / args.val_fold   / "norm_16bit"

    base_test       = Path(args.data_root) / args.test_fold
    depth_root_test = Path(args.depth_root) / args.test_fold  / "norm_16bit"

    train_ds = TrainLoadDatasetMulti_RGBD(
        train_samples, args.crop_size,
        base_dir=base_train,
        depth_root=depth_root_train
    )
    val_ds   = ValLoadDatasetMulti_RGBD(
        val_samples, args.crop_size,
        base_dir=base_val,
        depth_root=depth_root_val
    )
    test_ds  = TestLoadDatasetMulti_RGBD(
        test_samples,
        base_dir=base_test,
        depth_root=depth_root_test
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.train_bs, shuffle=True,
        num_workers=4, drop_last=False, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_bs, shuffle=False,
        num_workers=4, drop_last=False, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=4, drop_last=False, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    return train_loader, val_loader, test_loader


def main(args):
    set_seed(1234)
    args.device = torch.device(args.device)

    train_loader, val_loader, test_loader = build_dataloaders(args)

#     model = make_model(
#     model_name      = 'unetpp_rgbd_fuse_selective',  # ← 여기
#     encoder_name    = 'resnet50',
#     encoder_weights = 'imagenet', #'imagenet'
#     in_channels     = 3,
#     depth_channels  = 1,
#     n_classes       = 12,
#     patch_sizes     = [7,7,5], #7,7,5,3
#     fuse_stages     = [0,1,2],
#     device          = 'cuda',
# )
    model = make_model(
        model_name      = 'segb5_rgbd_fuse_stageselective_enc_update',  # segb5_rgbd_fuse_stageselective_enc_update
        n_classes       = 12,
        patch_sizes     = [7, 7, 5, 3], 
        fuse_stages     = [0, 1, 2, 3],
        attn_variant    = 'base',
        cp_rank         = 16,
        device          = 'cuda'
    )

    # model = make_model(
    #     model_name   = 'segb5_rgbd_fuse_stageselective_encupdate_switch',  # ★ 변경
    #     n_classes    = 12,
    #     patch_sizes  = [7, 7, 5, 3],
    #     fuse_stages  = [0, 1, 2, 3],
    #     device       = 'cuda',
    #     blend        = 'mulcoeff',   # or 'avg' / 'mulcoeff' / 'learnable'
    # )

    
    
    if args.disable_rgb:
        model.freeze_rgb()
        print("🔒 RGB encoder frozen, only Depth → Decoder 학습")
    if args.disable_depth:
        model.freeze_depth()
        print("🔒 Depth encoder frozen, only RGB → Decoder 학습")

    trainer = Trainer(
        model=model,
        save_root=args.save_root,
        weights_dir=args.weights_dir,
        results_dir=args.results_dir,
        device=args.device,
        lr=args.lr,
        num_epochs=args.num_epoch,
        n_classes=args.num_classes,
    ).to(args.device)

    

    trainer.train(
        train_loader,
        val_loader,
        test_loader,
        resume_ckpt=args.resume_ckpt
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RGB-D SegFormer B5 trainer.")

    # 데이터 경로
    p.add_argument("--data_root",  default="/content/dataset/")
    p.add_argument("--train_fold", default="fold2_train")
    p.add_argument("--val_fold",   default="fold2_valid")
    p.add_argument("--test_fold",  default="fold2_test")

    p.add_argument("--clear_dir", default="clear")
    p.add_argument("--seg_dir",   default="seg_label")
    p.add_argument("--rain_dir",  default="rain")
    p.add_argument("--snow_dir",  default="snow")
    p.add_argument("--fog_dir",   default="fog")
    p.add_argument("--flare_dir", default="flare")

    # depth 경로
    p.add_argument("--depth_root", default="/content/drive/MyDrive/weather_depth2")

    # 학습 설정
    # p.add_argument("--crop_size",  type=int, default=(288, 960))
    p.add_argument(
    "--crop_size",
    nargs=2,                 # ← 두 개의 인자(H, W) 받기
    type=int,
    metavar=("H", "W"),
    default=(288, 960)
)
    p.add_argument("--train_bs",   type=int, default=4)
    p.add_argument("--eval_bs",    type=int, default=1)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--num_epoch",  type=int, default=400)
    p.add_argument("--num_classes",type=int, default=12)

    # 저장 경로
    p.add_argument("--save_root",   default="/content/drive/MyDrive/weather_pth")
    p.add_argument("--weights_dir", default="/content/drive/MyDrive/weather_pth/segf2_gating_0123_enc2_288")
    p.add_argument("--results_dir", default="/content/drive/MyDrive/weather_pth/segf2_gating_0123_enc2_288")

    p.add_argument("--device",      default="cuda")
    p.add_argument("--resume_ckpt", default="/content/drive/MyDrive/weather_pth/segf2_gating_0123_enc2_288/last_epoch.pth")

    # ─── RGB/Depth 브랜치 On·Off 옵션 ───
    p.add_argument("--disable_rgb",   action="store_true", default=False)
    p.add_argument("--disable_depth", action="store_true", default=False)

    args = p.parse_args()
    main(args) 
