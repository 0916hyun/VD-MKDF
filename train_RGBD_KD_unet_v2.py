# train_RGBD_KD_unet_v1.py
# -*- coding: utf-8 -*-
import argparse, os, random, numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2

from data.loader_RGBD import (
    TrainLoadDatasetMulti_RGBD,
    ValLoadDatasetMulti_RGBD,
    TestLoadDatasetMulti_RGBD,
)
from data.weather_utils import make_sample_list

from models.make_model import make_model
from RGBD_trainer_KD_unet_v2 import KDTrainerUNetV2, SMPUnetKDWrapper


# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int = 1234):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_samples(root, fold, clear_dir, seg_dir, weather_dirs):
    base = Path(root) / fold
    fold_weather_dirs = {w: str(base / d) for w, d in weather_dirs.items()}
    return make_sample_list(root=str(base), scene_dir=clear_dir, weather_dirs=fold_weather_dirs, seg_dir=seg_dir)

def build_dataloaders(args):
    weather_dirs = {"rain": args.rain_dir, "snow": args.snow_dir, "fog": args.fog_dir, "flare": args.flare_dir}
    train_samples = build_samples(args.data_root, args.train_fold, args.clear_dir, args.seg_dir, weather_dirs)
    val_samples   = build_samples(args.data_root, args.val_fold,   args.clear_dir, args.seg_dir, weather_dirs)
    test_samples  = build_samples(args.data_root, args.test_fold,  args.clear_dir, args.seg_dir, weather_dirs)

    base_train = Path(args.data_root) / args.train_fold
    base_val   = Path(args.data_root) / args.val_fold
    base_test  = Path(args.data_root) / args.test_fold

    depth_root_train = Path(args.depth_root) / args.train_fold / "norm_16bit"
    depth_root_val   = Path(args.depth_root) / args.val_fold   / "norm_16bit"
    depth_root_test  = Path(args.depth_root) / args.test_fold  / "norm_16bit"

    train_ds = TrainLoadDatasetMulti_RGBD(train_samples, args.crop_size, base_dir=base_train, depth_root=depth_root_train)
    val_ds   = ValLoadDatasetMulti_RGBD(val_samples,   args.crop_size, base_dir=base_val,   depth_root=depth_root_val)
    test_ds  = TestLoadDatasetMulti_RGBD(test_samples, base_dir=base_test, depth_root=depth_root_test)

    def make_loader(ds, bs, shuffle):
        kwargs = dict(batch_size=bs, shuffle=shuffle, num_workers=args.num_workers,
                      drop_last=False, pin_memory=True)
        if args.num_workers and args.num_workers > 0:
            kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
        return DataLoader(ds, **kwargs)

    return make_loader(train_ds, args.train_bs, True), make_loader(val_ds, args.eval_bs, False), make_loader(test_ds, 1, False)


# ─────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────
def robust_load_teacher(model: nn.Module, ckpt_path: str, device: torch.device):
    if not ckpt_path: return model.eval().to(device)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and ("model_state_dict" in sd):
        model.load_state_dict(sd["model_state_dict"], strict=True)
    else:
        model.load_state_dict(sd, strict=True)
    for p in model.parameters(): p.requires_grad = False
    return model.eval().to(device)

def _unwrap_hf_target(wrapper_or_hf: nn.Module) -> nn.Module:
    if hasattr(wrapper_or_hf, "model") and isinstance(wrapper_or_hf.model, nn.Module): return wrapper_or_hf.model
    if hasattr(wrapper_or_hf, "net")   and isinstance(wrapper_or_hf.net,   nn.Module): return wrapper_or_hf.net
    return wrapper_or_hf

def load_ckpt_into_hf_like(wrapper_or_hf: nn.Module, ckpt_path: str,
                           strict_backbone: bool = False, ignore_head_mismatch: bool = True):
    if not ckpt_path: return
    sd = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(sd, dict): raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."): k = k[7:]
        if k.startswith("model.") : k = k[6:]
        new_sd[k] = v
    target = _unwrap_hf_target(wrapper_or_hf)
    if ignore_head_mismatch:
        try:
            cur = target.state_dict()
            drop = []
            for k in list(new_sd.keys()):
                if "decode_head.classifier" in k and (k not in cur or cur[k].shape != new_sd[k].shape):
                    drop.append(k)
            for k in drop: new_sd.pop(k, None)
        except Exception:
            pass
    target.load_state_dict(new_sd, strict=strict_backbone)


# ─────────────────────────────────────────────────────────────
# Build models
# ─────────────────────────────────────────────────────────────
def build_teacher_main(num_classes: int, ckpt: str, device: torch.device):
    m = make_model(
        model_name='segb5_rgbd_fuse_stageselective_enc_update',
        n_classes=num_classes, patch_sizes=[7,7,5,3], fuse_stages=[0,1,2,3], device=str(device)
    )
    return robust_load_teacher(m, ckpt, device)

def build_teacher_sub(name: str, num_classes: int, ckpt: str, device: torch.device):
    m = make_model(name, n_classes=num_classes).to(device)
    if ckpt and os.path.isfile(ckpt):
        load_ckpt_into_hf_like(m, ckpt, strict_backbone=False, ignore_head_mismatch=True)
    for p in m.parameters(): p.requires_grad = False
    return m.eval()

def build_student_unet(num_classes: int, encoder_name: str, encoder_weights: str|None, device: torch.device):
    return SMPUnetKDWrapper(num_classes=num_classes, encoder_name=encoder_name, encoder_weights=encoder_weights).to(device)

def build_classifier(ckpt: str, device: torch.device):
    clf = mobilenet_v2(weights=None); clf.classifier[1] = nn.Linear(clf.last_channel, 4)
    if ckpt and os.path.isfile(ckpt):
        sd = torch.load(ckpt, map_location="cpu"); clf.load_state_dict(sd, strict=True)
    for p in clf.parameters(): p.requires_grad = False
    return clf.eval().to(device)


# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="KD v1 + Gating Distill (Main RGB-D + 4 Sub RGB, Student: smp.Unet-ResNet50)")

    # data
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
    p.add_argument("--depth_root", default="/content/drive/MyDrive/weather_depth2")

    # loader
    # p.add_argument("--crop_size",  type=int, default=352)
    p.add_argument(
    "--crop_size",
    nargs=2,                 # ← 두 값(H, W) 받기
    type=int,
    metavar=("H", "W"),
    default=(288, 960),      # 원하는 기본
    help="Crop size as H W, e.g., 288 960"
)
    p.add_argument("--train_bs",   type=int, default=36)
    p.add_argument("--eval_bs",    type=int, default=1)
    p.add_argument("--num_workers",type=int, default=4)

    # model names / ckpts
    p.add_argument("--teacher_sub_name",  default="segformer_b5")
    p.add_argument("--teacher_main_ckpt", default="/content/drive/MyDrive/weather_pth/segf2_gating_0123_enc2_288/best_test_miou.pth")
    p.add_argument("--teacher_sub_rain_ckpt",  default="/content/drive/MyDrive/weather_pth/weather_seg_single2/rain2/best_test_miou.pth")
    p.add_argument("--teacher_sub_snow_ckpt",  default="/content/drive/MyDrive/weather_pth/weather_seg_single2/snow2/best_test_miou.pth")
    p.add_argument("--teacher_sub_fog_ckpt",   default="/content/drive/MyDrive/weather_pth/weather_seg_single2/fog2/best_test_miou.pth")
    p.add_argument("--teacher_sub_flare_ckpt", default="/content/drive/MyDrive/weather_pth/weather_seg_single2/flare2/best_test_miou.pth")
    p.add_argument("--classifier_ckpt", default="/content/drive/MyDrive/weather_pth/2classifier2/best_val_loss.pth")

    # student (smp.unet)
    p.add_argument("--student_encoder",  default="resnet50")
    p.add_argument("--student_imagenet", type=int, default=1)  # 1: imagenet, 0: None
    p.add_argument("--student_ckpt",     default="")           # (옵션) resume용

    # training
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_epoch",   type=int,   default=400)
    p.add_argument("--num_classes", type=int,   default=12)
    p.add_argument("--ignore_index",type=int,   default=11)

    # gating distill
    p.add_argument("--lambda_gate", type=float, default=1.0)
    p.add_argument("--gate_stage",  type=int,   default=2)     # 0..3, 기본: 마지막 스테이지의 g 사용

    # loss weights (기본값은 기존 동작과 동일)
    p.add_argument("--lambda_ce", type=float, default=1.5)
    p.add_argument("--lambda_logit", type=float, default=0.012)
    p.add_argument("--lambda_feat", type=float, default=1.50)

    # save/results
    p.add_argument("--weights_dir", default="/content/drive/MyDrive/weather_pth/kd_unet2_v2_2_f2/weights")
    p.add_argument("--results_dir", default="/content/drive/MyDrive/weather_pth/kd_unet2_v2_2_f2/results")

    # test
    p.add_argument("--test_after",  type=int, default=300)
    p.add_argument("--test_h",      type=int, default=352)
    p.add_argument("--test_w",      type=int, default=1152)

    # misc
    p.add_argument("--device",      default="cuda")
    p.add_argument("--resume_ckpt", default="")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(1234)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(args)

    # teachers
    teacher_main = build_teacher_main(args.num_classes, args.teacher_main_ckpt, device)
    teachers_sub = [
        build_teacher_sub(args.teacher_sub_name, args.num_classes, args.teacher_sub_rain_ckpt,  device),
        build_teacher_sub(args.teacher_sub_name, args.num_classes, args.teacher_sub_snow_ckpt,  device),
        build_teacher_sub(args.teacher_sub_name, args.num_classes, args.teacher_sub_fog_ckpt,   device),
        build_teacher_sub(args.teacher_sub_name, args.num_classes, args.teacher_sub_flare_ckpt, device),
    ]
    # student
    encoder_weights = "imagenet" if args.student_imagenet == 1 else None
    student = build_student_unet(args.num_classes, args.student_encoder, encoder_weights, device)

    # (옵션) 학생 resume
    if args.student_ckpt and os.path.isfile(args.student_ckpt):
        sd = torch.load(args.student_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            student.load_state_dict(sd["model_state_dict"], strict=False)
        else:
            student.load_state_dict(sd, strict=False)

    # classifier
    classifier = build_classifier(args.classifier_ckpt, device)

    trainer = KDTrainerUNetV2(
        student=student,
        teacher_main=teacher_main,
        teachers_sub=teachers_sub,
        classifier=classifier,
        device=str(device),

        n_classes=args.num_classes,
        ignore_index=args.ignore_index,
        lr=args.lr,
        num_epochs=args.num_epoch,
        weights_dir=args.weights_dir,
        results_dir=args.results_dir,
        test_after=args.test_after,
        test_infer_size=(args.test_h, args.test_w),

        # loss weights
        lambda_ce=args.lambda_ce,
        lambda_logit=args.lambda_logit,
        lambda_feat=args.lambda_feat,

        # gating distill params
        lambda_gate=args.lambda_gate,
        gate_stage=args.gate_stage,
    ).to(device)

    trainer.train_loop(train_loader, val_loader, test_loader, resume_ckpt=args.resume_ckpt)


if __name__ == "__main__":
    main()
