# midas_depth_estimation.py

import argparse, cv2, torch, numpy as np
from tqdm import tqdm
from pathlib import Path
from data.weather_utils import make_sample_list
from torch.utils.data import Dataset, DataLoader
from torch.hub import load_state_dict_from_url

# ───────────────────────────────────────────────────
# 1) MiDaS v3 모델 로드 (pretrained=False + 필터링 로드)
# ───────────────────────────────────────────────────
def load_midas3(device="cuda", model_type="DPT_BEiT_L_512"):
    repo = "isl-org/MiDaS:v3_1"
    midas = torch.hub.load(repo, model_type, pretrained=False, trust_repo=True)
    midas.to(device).eval()
    # b) 가중치 다운로드 & 불필요 키 제거
    if model_type == "DPT_BEiT_L_512":
        url = "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt"
    else:
        url = "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_large_384.pt"
    state_dict = load_state_dict_from_url(url, map_location="cpu", progress=True)
    state_dict = {k: v for k, v in state_dict.items()
                  if "relative_position_index" not in k}
    midas.load_state_dict(state_dict, strict=False)

    for blk in midas.pretrained.model.blocks:
        if hasattr(blk, "drop_path1") and not hasattr(blk, "drop_path"):
            blk.drop_path = blk.drop_path1

    # c) 전처리 함수 로드
    tfm = torch.hub.load(repo, "transforms", trust_repo=True)
    transform = tfm.beit512_transform if model_type == "DPT_BEiT_L_512" else tfm.dpt_transform
    return midas, transform

# ───────────────────────────────────────────────────
# 2) Dataset: inp 경로만 문자열로 반환
# ───────────────────────────────────────────────────
class ImgPathDataset(Dataset):
    def __init__(self, samples, base_dir):
        self.paths = [s["inp"] for s in samples]
        self.base_dir = Path(base_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return str(self.paths[idx])

# ───────────────────────────────────────────────────
# 3) Depth 저장 유틸: raw_mm(16-bit) + norm_8bit
# ───────────────────────────────────────────────────
def save_depth_versions(depth, save_root, rel_path):
    inv = depth.cpu().numpy()  # MiDaS inverse depth

    # 1) raw metric (mm)
    metric = 1.0 / (inv + 1e-6)
    depth_mm = (metric * 1000).clip(0, 65535).astype(np.uint16)
    raw_path = save_root / "raw_mm" / rel_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(raw_path), depth_mm)

    # 2) 0–255 정규화 (8bit)
    norm8 = (inv - inv.min()) / (np.ptp(inv) + 1e-6)
    depth_u8 = (norm8 * 255).astype(np.uint8)
    norm8_path = save_root / "norm_8bit" / rel_path
    norm8_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(norm8_path), depth_u8)

    # 3) 0–65535 정규화 (16bit)
    norm16 = ((inv - inv.min()) / (np.ptp(inv) + 1e-6) * 65535).clip(0, 65535).astype(np.uint16)
    norm16_path = save_root / "norm_16bit" / rel_path
    norm16_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(norm16_path), norm16)


# ───────────────────────────────────────────────────
# 4) Depth 추론 루프
# ───────────────────────────────────────────────────
@torch.no_grad()
def run_depth(loader, midas, transform, out_root, device="cuda"):
    out_root = Path(out_root)
    for img_path in tqdm(loader, desc="Depth inference"):
        # img_path는 이미 str
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        inp = transform(img).to(device)
        pred = midas(inp)                         # (B, H', W')
        pred = pred.unsqueeze(1)                  # (B, 1, H', W')
        pred = torch.nn.functional.interpolate(
            pred,
            size=img.shape[:2],                  # (orig_H, orig_W)
            mode="bicubic",
            align_corners=False
        )
        depth = pred.squeeze(1)[0]                # (orig_H, orig_W)

        rel = Path(img_path).relative_to(loader.dataset.base_dir)
        save_depth_versions(depth, out_root, rel.with_suffix(".png"))

# ───────────────────────────────────────────────────
# 5) main()
# ───────────────────────────────────────────────────
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    midas, transform = load_midas3(device, args.model_type)

    fold_root = Path(args.data_root) / args.fold
    weather_dirs = {
        "rain":  args.rain_dir,
        "snow":  args.snow_dir,
        "fog":   args.fog_dir,
        "flare": args.flare_dir,
    }
    samples = make_sample_list(
        root=str(fold_root),
        scene_dir=args.clear_dir,
        weather_dirs={w: str(fold_root / d) for w, d in weather_dirs.items()},
        seg_dir=args.seg_dir,
    )

    ds = ImgPathDataset(samples, base_dir=fold_root)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        collate_fn=lambda x: x[0]   
    )

    run_depth(loader, midas, transform, args.out_root, device)

if __name__ == "__main__":
    p = argparse.ArgumentParser("MiDaS v3 depth extractor (dual-format)")
    p.add_argument("--data_root", default="/content/dataset/")
    p.add_argument("--fold",      default="fold1_valid")
    p.add_argument("--clear_dir", default="clear")
    p.add_argument("--seg_dir",   default="seg_label")
    p.add_argument("--rain_dir",  default="rain")
    p.add_argument("--snow_dir",  default="snow")
    p.add_argument("--fog_dir",   default="fog")
    p.add_argument("--flare_dir",default="flare")
    p.add_argument("--out_root",  default="/content/drive/MyDrive/weather_depth2/fold1_valid")
    p.add_argument(
        "--model_type",
        choices=["DPT_BEiT_L_512", "DPT_Large"],
        default="DPT_BEiT_L_512"
    )
    args = p.parse_args()
    main(args)
