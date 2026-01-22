# test_loop.py

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from segmentation_models_pytorch.metrics.functional import get_stats
# from models.make_model_mmseg import make_model

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None


DEFAULT_HGFORMER_CFG = "HGFormer/configs/cityscapes/hgformer_swin_large_IN21K_384_bs16_20k.yaml"


def _resolve_hgformer_cfg(args) -> str:
    cfg_path = (args.hgformer_cfg or "").strip()
    variant_key = (args.model_variant or "").strip().lower()

    if not cfg_path:
        cfg_path = DEFAULT_HGFORMER_CFG

    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"지정된 HGFormer 설정 파일을 찾을 수 없습니다: {cfg_path}"
        )
    return cfg_path.as_posix()

from test import parse_args, build_dataloaders
from unet_trainer import Trainer, resize_inference
from unet_trainer import WEATHER_NAMES
from models.make_model import make_model
from utils.stream_metrics import StreamSegMetrics


# -----------------------------------------------------------------------------
# Grad-CAM helper (auto target discovery)
# -----------------------------------------------------------------------------


class _HookPair:
    """Handle forward/backward hooks with compatibility fallback."""

    def __init__(self, module: torch.nn.Module, fwd_cb, bwd_cb):
        self._fwd_h = module.register_forward_hook(fwd_cb)
        if hasattr(module, "register_full_backward_hook"):
            self._bwd_h = module.register_full_backward_hook(bwd_cb)
        else:  # pragma: no cover - backward hook API fallback
            self._bwd_h = module.register_backward_hook(bwd_cb)

    def remove(self):
        for handle in (getattr(self, "_fwd_h", None), getattr(self, "_bwd_h", None)):
            try:
                if handle is not None:
                    handle.remove()
            except Exception:  # pragma: no cover - defensive cleanup
                pass


def _name_to_dotted(name: str) -> str:
    return re.sub(r"\.(\d+)(\.|$)", r"[\1]\2", name)


def resolve_layer(root_module: torch.nn.Module, dotted: str) -> torch.nn.Module:
    module = root_module
    for part in dotted.split('.'):
        if not part:
            continue
        if part.endswith(']'):
            attr, idx = part.split('[')
            module = getattr(module, attr)[int(idx[:-1])]
        elif part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def normalize_minmax(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if tensor.ndim == 4:
        bsz = tensor.size(0)
        flat = tensor.view(bsz, -1)
        minv = flat.min(dim=1, keepdim=True)[0]
        maxv = flat.max(dim=1, keepdim=True)[0]
        out = ((flat - minv) / (maxv - minv + eps)).view_as(tensor)
        return out.clamp_(0, 1)
    mn = tensor.min()
    mx = tensor.max()
    return ((tensor - mn) / (mx - mn + eps)).clamp_(0, 1)


_PRIOR_PATTERNS: List[str] = [
    "decode_head.classifier",
    "decode_head",
    "segmentation_head",
    "decoder.blocks",
    "decoder",
    "classifier",
    "logits",
    "final",
    "head",
]


def auto_pick_target_layer_path(model: torch.nn.Module) -> str:
    conv_names: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_names.append(name)

    if not conv_names:
        raise RuntimeError("No Conv2d modules found in the model; cannot attach Grad-CAM target.")

    for pattern in _PRIOR_PATTERNS:
        candidates = [n for n in conv_names if pattern in n]
        if candidates:
            return _name_to_dotted(candidates[-1])

    return _name_to_dotted(conv_names[-1])


class AutoGradCAM:
    """Model-agnostic Grad-CAM helper with optional auto target discovery."""

    def __init__(self, model: torch.nn.Module, target_path: str = "", verbose: bool = True):
        if not target_path:
            target_path = auto_pick_target_layer_path(model)
        self.model = model
        self.target_path = target_path
        self.target_module = resolve_layer(model, target_path)

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        def _fwd(_, __, output):
            self.activations = output.detach()

        def _bwd(_, grad_input, grad_output):
            grad = grad_output[-1]
            self.gradients = grad.detach()

        self._hooks = _HookPair(self.target_module, _fwd, _bwd)
        self._zero_kwargs = {"set_to_none": True} if "set_to_none" in torch.nn.Module.zero_grad.__code__.co_varnames else {}
        if verbose:
            print(f"[AutoGradCAM] target: {self.target_path}")

    def close(self):
        if hasattr(self, "_hooks") and self._hooks is not None:
            self._hooks.remove()
        self.activations = None
        self.gradients = None

    def _build_cam(self) -> torch.Tensor:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        return normalize_minmax(cam)

    def _resize_cam(self, cam: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(cam, size=input_size, mode="bilinear", align_corners=False)

    def generate(self, logits: torch.Tensor, class_idx: int, input_size: Tuple[int, int], *, retain_graph=True) -> torch.Tensor:
        self.model.zero_grad(**self._zero_kwargs)
        if logits.ndim == 4:
            score = logits[:, class_idx].mean()
        elif logits.ndim == 2:
            score = logits[:, class_idx].mean()
        else:
            raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")
        score.backward(retain_graph=retain_graph)
        cam = self._build_cam()
        return self._resize_cam(cam, input_size)

    def generate_for_classes(self, logits: torch.Tensor, class_indices: Iterable[int], input_size: Tuple[int, int]) -> List[torch.Tensor]:
        heats: List[torch.Tensor] = []
        n = len(class_indices)
        for i, class_idx in enumerate(class_indices):
            keep = (i < n - 1)  # 마지막 클래스만 retain_graph=False
            heats.append(self.generate(logits, int(class_idx), input_size, retain_graph=keep))
        return heats


def overlay_heatmap(image_tensor: torch.Tensor, heatmap_2d: torch.Tensor, *, alpha: float = 0.6, gamma: float = 1.0) -> Optional["Image.Image"]:
    if Image is None or np is None:
        return None

    img = image_tensor.detach().float().cpu()
    if img.ndim != 3 or img.size(0) not in (1, 3):
        raise ValueError("image_tensor must be shaped [C,H,W] with C=1 or 3 for Grad-CAM overlay")
    if img.size(0) == 1:
        img = img.repeat(3, 1, 1)
    img_np = img.permute(1, 2, 0).numpy()
    if img_np.max() > 1.0:
        img_np = img_np / 255.0

    heat = heatmap_2d.detach().float().cpu().clamp(0, 1)
    if gamma != 1.0:
        heat = heat.pow(gamma)
    heat_np = heat.numpy()

    color = np.zeros((*heat_np.shape, 3), dtype=np.float32)
    low_mask = heat_np <= 0.5
    if np.any(low_mask):
        twice = 2.0 * heat_np[low_mask]
        color[low_mask, 0] = 0.0
        color[low_mask, 1] = twice
        color[low_mask, 2] = 1.0 - twice
    high_mask = heat_np > 0.5
    if np.any(high_mask):
        twice = 2.0 * (heat_np[high_mask] - 0.5)
        color[high_mask, 0] = twice
        color[high_mask, 1] = 1.0 - twice
        color[high_mask, 2] = 0.0

    overlay = (img_np * (1.0 - alpha) + color * alpha).clip(0.0, 1.0)
    overlay_uint8 = (overlay * 255.0).astype(np.uint8)
    return Image.fromarray(overlay_uint8, mode="RGB")


# -----------------------------------------------------------------------------
# Label/color utilities
# -----------------------------------------------------------------------------


CLASS_NAMES: List[str] = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
    "Void",
]
CLASS_NAME_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
TRACKED_CLASS_NAMES: Tuple[str, ...] = ("Pole", "SignSymbol", "Car", "Pedestrian", "Bicyclist")

LABEL_COLOR_MAP: Dict[str, List[int]] = {
    "Sky": [128, 128, 128],
    "Building": [128, 0, 0],
    "Pole": [192, 192, 128],
    "Road": [128, 64, 128],
    "Pavement": [0, 0, 192],
    "Tree": [128, 128, 0],
    "SignSymbol": [192, 128, 128],
    "Fence": [64, 64, 128],
    "Car": [64, 0, 128],
    "Pedestrian": [64, 64, 0],
    "Bicyclist": [0, 128, 192],
    "Void": [0, 0, 0],
}


def build_color_lut(num_classes: int) -> np.ndarray:
    lut = np.zeros((num_classes, 3), dtype=np.uint8)
    for name, color in LABEL_COLOR_MAP.items():
        idx = CLASS_NAME_TO_INDEX.get(name)
        if idx is None or idx >= num_classes:
            continue
        lut[idx] = np.asarray(color, dtype=np.uint8)
    return lut


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def colorize_mask(mask: np.ndarray, color_lut: np.ndarray) -> Optional["Image.Image"]:
    if Image is None:
        return None
    max_idx = color_lut.shape[0] - 1
    colored = color_lut[mask.clip(min=0, max=max_idx)]
    return Image.fromarray(colored, mode="RGB")


def parse_gradcam_classes(requested: Optional[Iterable[str]]) -> List[int]:
    if not requested:
        requested = CLASS_NAMES
    indices: List[int] = []
    for entry in requested:
        if isinstance(entry, int):
            idx = entry
        else:
            text = str(entry)
            if text.isdigit():
                idx = int(text)
            else:
                idx = CLASS_NAME_TO_INDEX.get(text)
                if idx is None:
                    raise ValueError(f"Unknown Grad-CAM class specification: {entry}")
        if idx < 0 or idx >= len(CLASS_NAMES):
            raise ValueError(f"Grad-CAM class index {idx} out of range")
        indices.append(idx)
    return indices

def run_test(trainer, loader, label, ckpt_path, args):
    start_ep = trainer._load_ckpt(ckpt_path)
    tot_loss, seg_loss, mean_iou, class_iou_dict, per_weather = trainer._test(loader)

    results_root = ensure_dir(Path(args.results_dir) / label)
    color_root: Optional[Path] = None
    color_lut: Optional[np.ndarray] = None

    color_saved = 0
    if args.export_color_maps:
        color_root = ensure_dir(results_root / "color_maps")
        color_lut = build_color_lut(trainer.n_classes)
        if Image is None:
            print("[WARN] Pillow가 설치되지 않아 컬러 마스크를 저장할 수 없습니다. `pip install pillow` 후 다시 시도하세요.")

    gradcam_root: Optional[Path] = None
    gradcam_classes: List[int] = []
    cam_helper: Optional[AutoGradCAM] = None
    gradcam_saved = 0
    if args.export_gradcam:
        gradcam_root = ensure_dir(results_root / "gradcam")
        gradcam_classes = parse_gradcam_classes(args.gradcam_classes)
        cam_helper = AutoGradCAM(trainer.model, target_path=args.gradcam_target_layer, verbose=args.gradcam_verbose)

    all_tp = torch.zeros(trainer.n_classes, device=trainer.device)
    all_fp = torch.zeros(trainer.n_classes, device=trainer.device)
    dataset = getattr(loader, "dataset", None)

    tracked_classes = {
        CLASS_NAME_TO_INDEX[name]: {"iou": -1.0, "weather": "", "image": ""}
        for name in TRACKED_CLASS_NAMES
        if CLASS_NAME_TO_INDEX.get(name) is not None
    }
    best_sample = {"miou": -1.0, "weather": "", "image": ""}

    trainer.model.eval()
    for batch_idx, (inp, _, seg, weather_ids, filenames) in enumerate(loader):
        inp = inp.to(trainer.device)
        seg = seg.to(trainer.device)

        with torch.no_grad():
            logits = resize_inference(trainer.model, inp)
        preds = logits.argmax(1)

        tp, fp, _, _ = get_stats(
            preds,
            seg,
            mode="multiclass",
            num_classes=trainer.n_classes,
            ignore_index=-1,
        )
        if tp.dim() == 2:
            tp = tp.sum(dim=0)
            fp = fp.sum(dim=0)

        all_tp += tp.to(trainer.device)
        all_fp += fp.to(trainer.device)

        preds_np = preds.detach().cpu().numpy()
        seg_np = seg.detach().cpu().numpy()

        heats_per_class: Optional[List[torch.Tensor]] = None
        if cam_helper is not None and gradcam_classes:
            from torch.cuda.amp import autocast
            prob_maps: Optional[torch.Tensor] = None
            with torch.enable_grad():
                cam_input = inp.detach().clone()
                # Grad-CAM은 연산 dtype 불일치(scatter) 오류가 발생하므로 FP32로 강제 실행
                with autocast(enabled=False):
                    logits_cam = resize_inference(trainer.model, cam_input)
                prob_maps = logits_cam.softmax(dim=1)

            try:
                heats_per_class = cam_helper.generate_for_classes(
                    logits_cam,
                    gradcam_classes,
                    input_size=seg.shape[-2:],
                )
            except Exception as exc:
                heats_per_class = []
                if args.gradcam_verbose:
                    print(f"[Grad-CAM] generation failed: {exc}")

            # Grad-CAM 결과가 모두 0이거나 비정상일 때는 softmax 맵을 대체 열맵으로 사용
            if prob_maps is not None:
                prob_maps = prob_maps.detach()
                if heats_per_class is None:
                    heats_per_class = []
                for idx, cls_idx in enumerate(gradcam_classes):
                    fallback_heat = normalize_minmax(prob_maps[:, cls_idx:cls_idx + 1])
                    if len(heats_per_class) <= idx:
                        heats_per_class.append(fallback_heat)
                        continue
                    cam_heat = heats_per_class[idx]
                    if cam_heat is None or not torch.isfinite(cam_heat).any() or cam_heat.max() <= 0:
                        heats_per_class[idx] = fallback_heat

            # 그래프 및 텐서 즉시 해제
            del logits_cam, cam_input, prob_maps
            trainer.model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if isinstance(filenames, (list, tuple)):
            name_list = list(filenames)
        else:
            name_list = [filenames]

        bsz = inp.size(0)
        for b in range(bsz):
            global_idx = batch_idx * bsz + b
            weather_id = int(weather_ids[b]) if torch.is_tensor(weather_ids) else int(weather_ids[b])
            weather_name = WEATHER_NAMES.get(weather_id, "unknown")
            image_name = None
            sample_info = None
            if dataset is not None and hasattr(dataset, "samples") and global_idx < len(dataset.samples):
                sample_info = dataset.samples[global_idx]
                if isinstance(sample_info, dict):
                    weather_name = sample_info.get("weather", weather_name)
                    image_name = Path(sample_info.get("inp", f"sample_{global_idx}.png")).name
            loader_name = str(name_list[b]) if b < len(name_list) else f"sample_{global_idx}"
            if not loader_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                loader_name = f"{loader_name}.png"
            if not image_name:
                image_name = loader_name

            sample_preds = preds_np[b:b + 1]
            sample_labels = seg_np[b:b + 1]

            sample_metric = StreamSegMetrics(trainer.n_classes, ignore_index=trainer.ignore_index)
            sample_metric.update(sample_labels, sample_preds)
            sample_results = sample_metric.get_results()
            sample_miou = float(sample_results.get("Mean IoU", 0.0))
            if not np.isnan(sample_miou) and sample_miou > best_sample["miou"]:
                best_sample = {"miou": sample_miou, "weather": weather_name, "image": image_name}

            class_iou_map = sample_results.get("Class IoU", {})
            for name in TRACKED_CLASS_NAMES:
                class_idx = CLASS_NAME_TO_INDEX.get(name)
                if class_idx is None:
                    continue
                class_iou = float(class_iou_map.get(class_idx, float("nan")))
                if np.isnan(class_iou):
                    continue
                tracked = tracked_classes[class_idx]
                if class_iou > tracked["iou"]:
                    tracked_classes[class_idx] = {
                        "iou": class_iou,
                        "weather": weather_name,
                        "image": image_name,
                    }

            if color_root is not None and color_lut is not None:
                color_dir = ensure_dir(color_root / weather_name)
                mask_img = colorize_mask(sample_preds[0], color_lut)
                if mask_img is not None:
                    mask_img.save(color_dir / image_name)
                    color_saved += 1

            if heats_per_class is not None and gradcam_root is not None:
                rgb_for_overlay = inp[b].detach().cpu()
                for cls_idx, heat in zip(gradcam_classes, heats_per_class):
                    if heat.shape[0] <= b:
                        continue
                    heat_map = heat[b, 0]
                    overlay = overlay_heatmap(
                        rgb_for_overlay,
                        heat_map,
                        alpha=args.gradcam_alpha,
                        gamma=args.gradcam_gamma,
                    )
                    if overlay is None:
                        continue
                    class_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
                    save_dir = ensure_dir(gradcam_root / weather_name / class_name)
                    overlay.save(save_dir / image_name)
                    gradcam_saved += 1

    pixel_acc = float(all_tp.sum() / (all_tp.sum() + all_fp.sum()).clamp(min=1))
    valid = (all_tp + all_fp) > 0
    class_acc = float((all_tp[valid] / (all_tp[valid] + all_fp[valid])).mean()) if valid.any() else 0.0

    trainer._append_test_csv(
        epoch=start_ep,
        tot_loss=tot_loss,
        seg_loss=seg_loss,
        miou=mean_iou,
        pixel_acc=pixel_acc,
        class_acc=class_acc,
        per_weather=per_weather,
        class_iou=class_iou_dict,
    )

    print(f"\n=== Test on [{label}] (epoch {start_ep}) ===")
    print(f" Total Loss: {tot_loss:.4f}, Seg Loss: {seg_loss:.4f}")
    print(f" Mean IoU: {mean_iou:.4f}, Pixel Acc: {pixel_acc:.4f}, Class Acc: {class_acc:.4f}\n")

    if color_root is not None:
        print(f" Color maps saved: {color_saved} files → {color_root}")
    if gradcam_root is not None:
        print(f" Grad-CAM overlays saved: {gradcam_saved} files → {gradcam_root}")

    print(">> Weather-wise Metrics:")
    for wid, stats in per_weather.items():
        name = WEATHER_NAMES[wid]
        print(f"  - {name:6s}| Loss {stats['loss']:.4f}, mIoU {stats['mIoU']:.4f}")

    print("\n>> Class IoU:")
    for cls, iou in class_iou_dict.items():
        print(f"  - Class {cls:2d}: IoU {iou:.4f}")

    print("\n=== Best Samples ===")
    if best_sample["miou"] >= 0:
        print(
            f" Highest overall mIoU image: weather={best_sample['weather']} "
            f"image={best_sample['image']} (mIoU={best_sample['miou']:.4f})"
        )
    else:
        print(" Highest overall mIoU image: None")

    for name in TRACKED_CLASS_NAMES:
        class_idx = CLASS_NAME_TO_INDEX.get(name)
        if class_idx is None:
            continue
        entry = tracked_classes.get(class_idx, {"iou": -1.0, "weather": "", "image": ""})
        if entry["iou"] >= 0:
            print(
                f" Best IoU for {name}: weather={entry['weather']} "
                f"image={entry['image']} (IoU={entry['iou']:.4f})"
            )
        else:
            print(f" Best IoU for {name}: No valid sample found.")

    if cam_helper is not None:
        cam_helper.close()

    return {
        "epoch": start_ep,
        "total_loss": tot_loss,
        "seg_loss": seg_loss,
        "mean_iou": mean_iou,
        "pixel_acc": pixel_acc,
        "class_acc": class_acc,
        "color_saved": color_saved,
        "gradcam_saved": gradcam_saved,
    }

def main():
    args = parse_args()
    test_loader = build_dataloaders(args)

    def build_trainer(cfg_path: str) -> Trainer:
        model = make_model(
            model_name='hgformer',              # ← VBLC 분기
            n_classes=args.num_classes,     # CAMVID 등 클래스 수       # 또는 'conv'
            pretrained_backbone=None,       # 필요시 경로 지정
            device=args.device,
            cfg_path=cfg_path,
            cfg_options=args.hgformer_opts,
            weights_path=args.hgformer_weights or None,
            ignore_index=args.ignore_index,
        )

        return Trainer(
            model=model,
            save_root   = args.data_root,      # 실제로는 unused
            weights_dir = args.weights_dir,
            results_dir = args.results_dir,
            device      = args.device,
            lr          = 1e-4,
            num_epochs  = 0,
            n_classes   = args.num_classes,
            ignore_index= args.ignore_index
        ).to(args.device)

    cfg_path = _resolve_hgformer_cfg(args)

    # ① best_val_loss.pth
    for label, ckpt_name in [("best_val_miou", args.ckpt_loss)]:
        ckpt_path = os.path.join(args.weights_dir, ckpt_name)
        print(f"[HGFormer] {label}: {cfg_path} 구성으로 테스트를 실행합니다...")
        trainer = build_trainer(cfg_path)
        metrics = run_test(trainer, test_loader, label, ckpt_path, args)
        if metrics is not None:
            print(
                f"[Summary] {label} | "
                f"mIoU {metrics['mean_iou']:.4f}, PixelAcc {metrics['pixel_acc']:.4f}, "
                f"ClassAcc {metrics['class_acc']:.4f}, "
                f"ColorMaps {metrics['color_saved']}, GradCAMs {metrics['gradcam_saved']}"
            )



if __name__ == "__main__":
    main()