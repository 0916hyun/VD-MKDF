import argparse, random, os, numpy as np, torch
from torch.utils.data import DataLoader
from pathlib import Path
from data.loader import (
    TrainLoadDatasetMulti, ValLoadDatasetMulti, TestLoadDatasetMulti
)
from data.weather_utils import make_sample_list
from unet_trainer import Trainer 
# from origin_trainer import Trainer 
from models.make_model import make_model


HGFORMER_PRESET_CONFIGS = {
    "hgformer_swin_tiny_bs16_20k": "HGFormer/configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml",
    "swin_tiny": "HGFormer/configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml",
    "hgformer_swin_large_in21k_384_bs16_20k": "HGFormer/configs/cityscapes/hgformer_swin_large_IN21K_384_bs16_20k.yaml",
    "swin_large": "HGFormer/configs/cityscapes/hgformer_swin_large_IN21K_384_bs16_20k.yaml",
    "hgformer_r50_bs16_20k": "HGFormer/configs/cityscapes/hgformer_R50_bs16_20k.yaml",
    "r50": "HGFormer/configs/cityscapes/hgformer_R50_bs16_20k.yaml",
    "hgformer_r50_bs16_20k_mapillary": "HGFormer/configs/mapillary/hgformer_R50_bs16_20k_mapillary.yaml",
    "r50_mapillary": "HGFormer/configs/mapillary/hgformer_R50_bs16_20k_mapillary.yaml",
    "hgformer_swin_tiny_bs16_20k_mapillary": "HGFormer/configs/mapillary/hgformer_swin_tiny_bs16_20k_mapillary.yaml",
    "swin_tiny_mapillary": "HGFormer/configs/mapillary/hgformer_swin_tiny_bs16_20k_mapillary.yaml",
}

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_samples(root, fold, clear_dir, seg_dir, weather_dirs):
    """
    root/fold/ 하위에
       clear_dir/, seg_dir/, weather_dirs[w]/  가 존재한다는 가정.
    fold 단위로 (scene, weather) 샘플 리스트 생성.
    """
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
    val_samples   = build_samples(
        args.data_root, args.val_fold,   args.clear_dir, args.seg_dir, weather_dirs
    )
    test_samples  = build_samples(
        args.data_root, args.test_fold,  args.clear_dir, args.seg_dir, weather_dirs
    )

    train_ds = TrainLoadDatasetMulti(train_samples, crop_size=args.crop_size)
    val_ds   = ValLoadDatasetMulti(val_samples, crop_size=args.crop_size)
    test_ds  = TestLoadDatasetMulti(test_samples)

    train_loader = DataLoader(
        train_ds, batch_size=args.train_bs, shuffle=True,
        num_workers=4, drop_last=False, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_bs, shuffle=False, num_workers=4, drop_last=False, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    return train_loader, val_loader, test_loader

def main(args):
    set_seed(1234)

    train_loader, val_loader, test_loader = build_dataloaders(args)

    variant = (args.model_variant or "").strip()
    model_kwargs = {
        "variant": variant,
    }
    if args.model_name.lower() == "hgformer":
        cfg_path = args.hgformer_config
        variant_key = variant.lower()
        if not cfg_path:
            if not variant_key:
                variant_key = "hgformer_swin_tiny_bs16_20k"
            cfg_path = HGFORMER_PRESET_CONFIGS.get(variant_key)
            if not cfg_path:
                raise ValueError(
                    "HGFormer requires --hgformer_config or a known --model_variant preset."
                )
            if not variant:
                variant = variant_key
                model_kwargs["variant"] = variant
        extra = {
            "cfg_path": cfg_path,
            "weights_path": args.hgformer_weights or None,
            "cfg_options": args.hgformer_opts,
            "ignore_index": args.ignore_index,
        }
        model_kwargs.update(extra)

    model = make_model(
        model_name=args.model_name,
        n_classes=args.num_classes,
        device=args.device,
        **model_kwargs,
    )

    trainer = Trainer(
        model=model,
        save_root=args.save_root,
        weights_dir=args.weights_dir,
        results_dir=args.results_dir,
        device=args.device,
        lr=args.lr,
        num_epochs=args.num_epoch,
        n_classes=args.num_classes,
        ignore_index=args.ignore_index,
    ).to(args.device)

    trainer.train(
        train_loader,
        val_loader,
        test_loader,
        resume_ckpt=args.resume_ckpt
    )



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-weather UNet trainer.")


    p.add_argument("--data_root", default="/content/dataset/")
    p.add_argument("--train_fold", default="fold1_train")
    p.add_argument("--val_fold",   default="fold1_valid")
    p.add_argument("--test_fold",  default="fold1_test")


    p.add_argument("--clear_dir", default="clear")
    p.add_argument("--seg_dir",   default="seg_label")


    p.add_argument("--rain_dir",  default="rain")
    p.add_argument("--snow_dir",  default="snow")
    p.add_argument("--fog_dir",   default="fog")
    p.add_argument("--flare_dir", default="flare")


    p.add_argument(
        "--crop_size",
        nargs=2,  # ← 두 개(H, W) 받기
        type=int,
        metavar=("H", "W"),
        default=(288, 960),  # 원하는 기본값
        help="Crop size as H W (e.g., 288 960)."
    )
    p.add_argument("--train_bs",   type=int, default=16) #32
    p.add_argument("--eval_bs",    type=int, default=1)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--num_epoch",  type=int, default=400)
    p.add_argument("--num_classes",type=int, default=12)
    p.add_argument("--ignore_index", type=int, default=11)


    p.add_argument("--save_root",   default="/content/drive/MyDrive/weather_pth")
    p.add_argument("--weights_dir", default="/content/drive/MyDrive/weather_pth/kitti_hgformer_new_1")
    p.add_argument("--results_dir", default="/content/drive/MyDrive/weather_pth/kitti_hgformer_new_1")


    p.add_argument("--device", default="cuda")
    p.add_argument("--resume_ckpt", default="")

     # 기본 학습 모델을 HGFormer Swin-Large(IN21K, 384, bs16, 20k)으로 설정해
    # 테스트에 사용 중인 설정과 동일하게 맞춘다.
    p.add_argument("--model_name", default="hgformer")
    p.add_argument(
        "--model_variant",
        default="hgformer_swin_large_in21k_384_bs16_20k",
        help=(
            "Optional model variant hint. When using HGFormer, known presets will "
            "automatically populate --hgformer_config if it is not provided."
        ),
    )
    p.add_argument(
        "--hgformer_config",
        default="HGFormer/configs/cityscapes/hgformer_swin_large_IN21K_384_bs16_20k.yaml",
        help=(
            "Path to a Detectron2 config for HGFormer. If omitted, a preset will be "
            "selected based on --model_variant when possible."
        ),
    )
    p.add_argument("--hgformer_weights", default="")
    p.add_argument("--hgformer_opts", nargs="*", default=None,
                   help="Additional Detectron2 config overrides for HGFormer.")

    args = p.parse_args()
    main(args)
