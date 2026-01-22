import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.stream_metrics import StreamSegMetrics
from segmentation_models_pytorch.metrics.functional import get_stats
import torch.nn.functional as F

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weather 이름 매핑
WEATHER_NAMES = {0: "rain", 1: "snow", 2: "fog", 3: "flare"}

# 리사이즈 후 inference 유틸

def resize_inference_rgbd(model, rgb, depth, dst=(352, 1152)):
    B, _, H, W = rgb.shape
    rgb_rs = F.interpolate(rgb, dst, mode="bilinear", align_corners=False)
    depth_rs = F.interpolate(depth, dst, mode="bilinear", align_corners=False)
    out_rs = model(rgb_rs, depth_rs)
    if isinstance(out_rs, tuple):
        out_rs = out_rs[0]
    return F.interpolate(out_rs, (H, W), mode="bilinear", align_corners=False)

# 날씨별 통계 유지 클래스
class WeatherStat:
    def __init__(self, n_classes):
        self.loss_sum = 0.0
        self.count = 0
        self.metric = StreamSegMetrics(n_classes)

    def update(self, loss, labels, preds, batch_sz):
        self.loss_sum += float(loss) * batch_sz
        self.count    += batch_sz
        self.metric.update(labels, preds)

    def summary(self):
        if self.count == 0:
            return 0.0, 0.0, {i: 0.0 for i in range(self.metric.n_classes)}
        results = self.metric.get_results()
        return self.loss_sum / self.count, results['Mean IoU'], results['Class IoU']

# Trainer 클래스 정의
class Trainer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        save_root: str,
        weights_dir: str,
        results_dir: str,
        n_classes: int,
        num_epochs: int,
        lr: float,                               # ← 기본값 없이 필수로
        device: str = "cuda",
        optimizer: optim.Optimizer | None = None, # ← 선택 인자(없으면 내부에서 Adam 생성)
        scheduler=None,
    ):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.n_classes = n_classes
        self.num_epochs = num_epochs
        self.sch = scheduler
        self.metrics = StreamSegMetrics(n_classes)

        # 경로 설정
        self.save_root = save_root
        self.weights_dir = weights_dir           # 절대경로 그대로 사용
        self.results_dir = results_dir           # 절대경로 그대로 사용
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # 옵티마이저: 전달 안 되면 여기서 생성
        if optimizer is None:
            self.opt = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.opt = optimizer

        # 체크포인트 경로
        self.last_ckpt_path      = os.path.join(weights_dir, "last_epoch.pth")
        self.best_val_loss_path  = os.path.join(weights_dir, "best_val_loss.pth")
        self.best_val_miou_path  = os.path.join(weights_dir, "best_val_miou.pth")
        self.best_test_miou_path = os.path.join(weights_dir, "best_test_miou.pth")

        # best 값 초기화
        self.best_val_loss = float('inf')
        self.best_val_miou = 0.0
        self.best_test_miou = 0.0

        # CSV 파일 경로
        self.csv_train      = os.path.join(weights_dir, "train_metrics.csv")
        self.csv_val        = os.path.join(weights_dir, "valid_metrics.csv")
        self.csv_test_log   = os.path.join(weights_dir, "best_test_metrics.csv")
        self.csv_test_all   = os.path.join(weights_dir, "test_metrics.csv")

        # CSV 초기화
        self._init_csv(self.csv_train)
        self._init_csv(self.csv_val)
        self._init_test_csv(self.csv_test_log)
        self._init_test_csv(self.csv_test_all)

    def _init_csv(self, path):
        header = ["epoch", "total_loss", "seg_loss", "mIoU"]
        for wid in range(4):
            header += [f"{WEATHER_NAMES[wid]}_loss", f"{WEATHER_NAMES[wid]}_mIoU"]
        header += [f"class_{i}_IoU" for i in range(self.n_classes)]
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def _init_test_csv(self, path):
        header = ["epoch", "total_loss", "seg_loss", "mIoU", "pixel_acc", "class_acc"]
        for wid in range(4):
            header += [f"{WEATHER_NAMES[wid]}_loss", f"{WEATHER_NAMES[wid]}_mIoU"]
        header += [f"class_{i}_IoU" for i in range(self.n_classes)]
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def _append_csv(self, path, row):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _epoch(self, loader, training: bool):
        # mode 설정
        if training:
            self.model.train()
        else:
            self.model.eval()
        self.metrics.reset()
        stats = {wid: WeatherStat(self.n_classes) for wid in range(4)}

        tot_sum = seg_sum = 0.0
        pbar = tqdm(loader, desc="[Train]" if training else "[Valid]", unit="batch")
        for rgb, tar, depth, seg, wid in pbar:
            rgb   = rgb.to(self.device)
            depth = depth.to(self.device)
            seg   = seg.to(self.device)

            out = self.model(rgb, depth)
            if isinstance(out, tuple):
                out = out[0]

            if out.shape[-2:] != seg.shape[-2:]:
                out = F.interpolate(out, size=seg.shape[-2:], mode='bilinear', align_corners=False)

            loss_seg = F.cross_entropy(out, seg, ignore_index=-1)

            if training:
                self.opt.zero_grad(set_to_none=True)
                loss_seg.backward()
                self.opt.step()

            tot_sum += float(loss_seg.item())
            seg_sum += float(loss_seg.item())

            # mIoU 집계용
            preds = out.argmax(1).detach().cpu().numpy()
            labels = seg.detach().cpu().numpy()
            self.metrics.update(labels, preds)

            # 날씨별 집계
            for b in range(rgb.size(0)):
                w = int(wid[b])
                stats[w].update(loss_seg.item(), labels[b:b+1], preds[b:b+1], 1)

        results = self.metrics.get_results()
        per_weather = {wid: stats[wid].summary() for wid in range(4)}
        return tot_sum/len(loader), seg_sum/len(loader), results['Mean IoU'], results['Class IoU'], per_weather

    @torch.no_grad()
    def _test(self, loader):
        self.model.eval()
        self.metrics.reset()
        stats = {wid: WeatherStat(self.n_classes) for wid in range(4)}
        tot_sum = seg_sum = 0.0

        # 기본 loss, mIoU 집계
        for rgb, tar, depth, seg, wid in tqdm(loader, desc="[Test]", unit="batch"):
            rgb = rgb.to(self.device)
            depth = depth.to(self.device)
            seg = seg.to(self.device)

            out = resize_inference_rgbd(self.model, rgb, depth)
            if isinstance(out, tuple):
                out = out[0]
            loss_seg = F.cross_entropy(out, seg, ignore_index=-1)

            tot_sum += float(loss_seg.item())
            seg_sum += float(loss_seg.item())
            preds = out.argmax(1).detach().cpu().numpy()
            labels = seg.detach().cpu().numpy()
            self.metrics.update(labels, preds)
            for b in range(rgb.size(0)):
                w = int(wid[b])
                stats[w].update(loss_seg.item(), labels[b:b+1], preds[b:b+1], 1)

        # multi-class pixel & class acc
        all_tp = torch.zeros(self.n_classes, device=self.device)
        all_fp = torch.zeros(self.n_classes, device=self.device)
        for rgb, tar, depth, seg, wid in loader:
            rgb, depth, seg = rgb.to(self.device), depth.to(self.device), seg.to(self.device)
            out   = resize_inference_rgbd(self.model, rgb, depth)
            preds = out.argmax(1)
            tp, fp, fn, tn = get_stats(preds, seg, mode='multiclass', num_classes=self.n_classes, ignore_index=-1)

            tp = tp.to(self.device)
            fp = fp.to(self.device)

            if tp.dim() == 2:
                tp, fp = tp.sum(0), fp.sum(0)
            all_tp += tp
            all_fp += fp

        pixel_acc = float(all_tp.sum() / (all_tp.sum() + all_fp.sum()).clamp(min=1))
        valid     = (all_tp + all_fp) > 0
        class_acc = float((all_tp[valid] / (all_tp[valid] + all_fp[valid])).mean()) if valid.any() else 0.0

        results     = self.metrics.get_results()
        per_weather = {wid: stats[wid].summary() for wid in range(4)}
        return tot_sum/len(loader), seg_sum/len(loader), results['Mean IoU'], results['Class IoU'], per_weather, pixel_acc, class_acc
    
    def _load_ckpt(self, pth):
        """
        resume_ckpt 혹은 Quick Test용 checkpoint(pth)를 로드합니다.
        - 단일 state_dict 저장 파일인 경우: torch.load(pth) 자체가 state_dict
        - 전체 dict({epoch, model_state_dict, optimizer_state_dict})인 경우: 딕셔너리 형태
        """
        ckpt = torch.load(pth, map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            # 마지막 체크포인트 형식
            self.model.load_state_dict(ckpt['model_state_dict'])
            # optimizer state도 함께 로드하려면 uncomment:
            # self.opt.load_state_dict(ckpt['optimizer_state_dict'])
            return ckpt.get('epoch', 0)
        else:
            # state_dict만 저장된 best_* 파일 형식
            self.model.load_state_dict(ckpt)

    def train(self, train_loader, val_loader, test_loader, resume_ckpt=None):
        start_ep = 0
        if resume_ckpt:
            start_ep = self._load_ckpt(resume_ckpt)

        for ep in range(start_ep, self.num_epochs):
            # Train / Val 단계
            tr_tot, tr_seg, tr_miou, tr_ciou, tr_w = self._epoch(train_loader, True)
            vl_tot, vl_seg, vl_miou, vl_ciou, vl_w = self._epoch(val_loader, False)

            # CSV 기록 (Train / Val)
            self._append_csv(self.csv_train, [ep+1, tr_tot, tr_seg, tr_miou] + 
                             [tr_w[wid][j] for wid in range(4) for j in range(2)] + 
                             [tr_ciou.get(i,0) for i in range(self.n_classes)])
            self._append_csv(self.csv_val,   [ep+1, vl_tot, vl_seg, vl_miou] + 
                             [vl_w[wid][j] for wid in range(4) for j in range(2)] + 
                             [vl_ciou.get(i,0) for i in range(self.n_classes)])

            # Best Val Loss / mIoU 체크포인트 저장
            if vl_tot < self.best_val_loss:
                self.best_val_loss = vl_tot
                torch.save(self.model.state_dict(), self.best_val_loss_path)
            if vl_miou > self.best_val_miou:
                self.best_val_miou = vl_miou
                torch.save(self.model.state_dict(), self.best_val_miou_path)

            # 300 epoch 이후 매 epoch Test 수행 및 Best Test mIoU 갱신
            if ep + 1 >= 350:
                ts_tot, ts_seg, ts_miou, ts_ciou, ts_w, pix_acc, cls_acc = self._test(test_loader)
                # 모든 Test 기록
                self._append_csv(
                    self.csv_test_all,
                    [ep+1, ts_tot, ts_seg, ts_miou, pix_acc, cls_acc]
                    + [ts_w[wid][j] for wid in range(4) for j in range(2)]
                    + [ts_ciou.get(i, 0) for i in range(self.n_classes)]
                )
                # Best Test mIoU 갱신
                if ts_miou > self.best_test_miou:
                    self.best_test_miou = ts_miou
                    torch.save(self.model.state_dict(), self.best_test_miou_path)
                    # Best Test 로그
                    self._append_csv(
                        self.csv_test_log,
                        [ep+1, ts_tot, ts_seg, ts_miou, pix_acc, cls_acc]
                        + [ts_w[wid][j] for wid in range(4) for j in range(2)]
                        + [ts_ciou.get(i, 0) for i in range(self.n_classes)]
                    )
                    print(f"[Epoch {ep+1}] New Best Test mIoU {ts_miou:.4f}, saved to {self.best_test_miou_path}")

            # 마지막 체크포인트 저장
            torch.save({
                'epoch': ep+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict()
            }, self.last_ckpt_path)

        # Quick Test: Best Val Loss / Best Val mIoU / Best Test mIoU 모델에 대해 최종 평가 & 기록
        for label, pth in [
            ("best_val_loss", self.best_val_loss_path),
            ("best_val_miou", self.best_val_miou_path),
            ("best_test_miou", self.best_test_miou_path),
        ]:
            if not os.path.exists(pth):
                print(f"\n=== Quick Test on [{label}] skipped (missing: {pth}) ===")
                continue
            self._load_ckpt(pth)
            q_tot, q_seg, q_miou, q_ciou, q_w, q_pix, q_cls = self._test(test_loader)
            print(f"\n=== Quick Test on [{label}] ===")
            print(f" Total Loss: {q_tot:.4f}, Seg Loss: {q_seg:.4f}")
            print(f" mIoU: {q_miou:.4f}, Pixel Acc: {q_pix:.4f}, Class Acc: {q_cls:.4f}")
            for wid in range(4):
                loss_w, miou_w, _ = q_w[wid]
                print(f"  - {WEATHER_NAMES[wid]} | loss {loss_w:.4f}, mIoU {miou_w:.4f}")
            for cls, iou in q_ciou.items():
                print(f"  - class {cls} IoU: {iou:.4f}")
            # Quick Test 기록 (test_metrics.csv)
            self._append_csv(
                self.csv_test_all,
                [self.num_epochs, q_tot, q_seg, q_miou, q_pix, q_cls]
                + [q_w[wid][j] for wid in range(4) for j in range(2)]
                + [q_ciou.get(i, 0) for i in range(self.n_classes)]
            )

        # Quick Test 모두 끝난 뒤, fuse stages를 원본 리스트 그대로 출력
        if hasattr(self.model, "fuse_stages_list"):
            print(f"Fuse stages (original): {self.model.fuse_stages_list}")
        elif hasattr(self.model, "fuse_stages"):
            try:
                print(f"Fuse stages (from set): {sorted(list(self.model.fuse_stages))}")
            except Exception:
                print(f"Fuse stages: {self.model.fuse_stages}")

        print(f"Training complete. Best Test mIoU: {self.best_test_miou:.4f}")
