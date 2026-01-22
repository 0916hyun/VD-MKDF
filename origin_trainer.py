import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, csv, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from tqdm import tqdm
import segmentation_models_pytorch as smp
from utils.stream_metrics import StreamSegMetrics
from segmentation_models_pytorch.metrics.functional import get_stats
from HRNet.lib.models.seg_hrnet import get_seg_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEATHER_NAMES = {0: "rain", 1: "snow", 2: "fog", 3: "flare"}

class WeatherStat:
    def __init__(self, n_classes):
        self.loss_sum = 0.
        self.count    = 0
        self.metric   = StreamSegMetrics(n_classes)

    def update(self, loss, labels, preds, batch_sz):
        self.loss_sum += loss * batch_sz
        self.count    += batch_sz
        self.metric.update(labels, preds)

    def summary(self):
        if self.count == 0:
            return 0., 0., {i: 0. for i in range(self.metric.n_classes)}
        m = self.metric.get_results()
        return self.loss_sum / self.count, m['Mean IoU'], m['Class IoU']

def resize_inference(model, img, dst=(352, 1152)):
    B, _, H, W = img.shape
    img_rs = torch.nn.functional.interpolate(img, dst, mode="bilinear", align_corners=False)
    out_rs = model(img_rs)
    if isinstance(out_rs, tuple):
        out_rs = out_rs[0]
    return torch.nn.functional.interpolate(out_rs, (H, W), mode="bilinear", align_corners=False)

class Trainer(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 save_root: str,
                 weights_dir: str,
                 results_dir: str,
                 device: torch.device,
                 lr: float,
                 num_epochs: int,
                 n_classes: int,
                 ignore_index: int = 11):
        super().__init__()
        self.device, self.lr, self.num_epochs = device, lr, num_epochs
        self.model = model.to(self.device)
        self.n_classes = n_classes

        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        self.weights_dir, self.results_dir = weights_dir, results_dir
        self.last_ckpt_path = os.path.join(weights_dir, "last_epoch.pth")

        self.opt      = optim.Adam(self.model.parameters(), lr=lr)
        self.crit_seg = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
        self.metrics  = StreamSegMetrics(n_classes)

        self.best_val_loss = float('inf')
        self.best_val_miou = 0.0
        self.best_loss_path = os.path.join(weights_dir, "best_val_loss.pth")
        self.best_miou_path = os.path.join(weights_dir, "best_val_miou.pth")

        self.csv_train = os.path.join(weights_dir, "train_metrics.csv")
        self.csv_val   = os.path.join(weights_dir, "valid_metrics.csv")
        self.csv_test  = os.path.join(weights_dir, "test_metrics.csv")
        self._init_csv(self.csv_train)
        self._init_csv(self.csv_val)
        self._init_test_csv(self.csv_test)

        self.hist_ep, self.hist_tr_tot, self.hist_tr_seg = [], [], []
        self.hist_va_tot, self.hist_va_seg = [], []

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

    def _append_csv(self, path, epoch, tot_loss, seg_loss, miou, per_weather, class_iou):
        row = [epoch, tot_loss, seg_loss, miou]
        for wid in range(4):
            row += [per_weather[wid]['loss'], per_weather[wid]['mIoU']]
        row += [class_iou.get(i, 0.) for i in range(self.n_classes)]
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _append_test_csv(self, epoch, tot_loss, seg_loss, miou, pixel_acc, class_acc, per_weather, class_iou):
        row = [epoch, tot_loss, seg_loss, miou, pixel_acc, class_acc]
        for wid in range(4):
            row += [per_weather[wid]['loss'], per_weather[wid]['mIoU']]
        row += [class_iou.get(i, 0.) for i in range(self.n_classes)]
        with open(self.csv_test, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _save_ckpt(self, epoch):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict()
        }, os.path.join(self.weights_dir, f"epoch_{epoch:04d}.pth"))

    def _load_ckpt(self, path):
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        if "optimizer_state_dict" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_ep = ckpt.get("epoch", 0)
        print(f"✓ Resumed from {path} (epoch {start_ep})")
        return start_ep

    def _epoch(self, loader, training: bool):
        self.model.train() if training else self.model.eval()
        self.metrics.reset()
        weather_stats = {wid: WeatherStat(self.n_classes) for wid in range(4)}
        tot_loss_sum = seg_loss_sum = 0.

        for _, clean, seg, wid, _ in tqdm(loader, desc="[Train]" if training else "[Val ]", unit="batch"):
            img = clean.to(self.device)
            seg = seg.to(self.device)
            if training: self.opt.zero_grad()

            out = self.model(img)
            if out.shape[-2:] != seg.shape[-2:]:
                out = torch.nn.functional.interpolate(out, size=seg.shape[-2:], mode='bilinear', align_corners=False)

            seg_loss = self.crit_seg(out, seg)
            tot_loss = seg_loss
            if training:
                tot_loss.backward()
                self.opt.step()

            tot_loss_sum += tot_loss.item()
            seg_loss_sum += seg_loss.item()

            preds = out.argmax(1).cpu().numpy()
            labels = seg.cpu().numpy()
            self.metrics.update(labels, preds)
            for b in range(img.size(0)):
                w = int(wid[b])
                weather_stats[w].update(tot_loss.item(), labels[b:b+1], preds[b:b+1], 1)

        m = self.metrics.get_results()
        per_w = {wid: dict(zip(["loss","mIoU","class_iou"], weather_stats[wid].summary())) for wid in range(4)}
        return tot_loss_sum/len(loader), seg_loss_sum/len(loader), m['Mean IoU'], m['Class IoU'], per_w

    @torch.no_grad()
    def _test(self, loader):
        self.model.eval()
        self.metrics.reset()
        weather_stats = {wid: WeatherStat(self.n_classes) for wid in range(4)}
        tot_sum = seg_sum = 0.

        for _, clean, seg, wid, _ in tqdm(loader, desc="[Test]", unit="batch"):
            img = clean.to(self.device)
            seg = seg.to(self.device)
            out = resize_inference(self.model, img)
            seg_loss = self.crit_seg(out, seg)
            tot_sum += seg_loss.item()
            seg_sum += seg_loss.item()

            preds = out.argmax(1).cpu().numpy()
            labels = seg.cpu().numpy()
            self.metrics.update(labels, preds)
            for b in range(img.size(0)):
                w = int(wid[b])
                weather_stats[w].update(seg_loss.item(), labels[b:b+1], preds[b:b+1], 1)

        m = self.metrics.get_results()
        per_w = {wid: dict(zip(["loss","mIoU","class_iou"], weather_stats[wid].summary())) for wid in range(4)}
        return tot_sum/len(loader), seg_sum/len(loader), m['Mean IoU'], m['Class IoU'], per_w

    def train(self, train_loader, val_loader, test_loader, resume_ckpt=None):
        start_ep = self._load_ckpt(resume_ckpt) if resume_ckpt else 0

        for ep in range(start_ep, self.num_epochs):
            tr_tot, tr_seg, tr_miou, tr_ciou, tr_w = self._epoch(train_loader, True)
            vl_tot, vl_seg, vl_miou, vl_ciou, vl_w = self._epoch(val_loader, False)

            # CSV 기록
            self._append_csv(self.csv_train, ep+1, tr_tot, tr_seg, tr_miou, tr_w, tr_ciou)
            self._append_csv(self.csv_val,   ep+1, vl_tot, vl_seg, vl_miou, vl_w, vl_ciou)

            # 최적 모델 저장
            if vl_tot < self.best_val_loss:
                self.best_val_loss = vl_tot
                torch.save(self.model.state_dict(), self.best_loss_path)
                print(f"✓ [Best Loss] Val Loss↓ {vl_tot:.4f} → {self.best_loss_path}")
            if vl_miou > self.best_val_miou:
                self.best_val_miou = vl_miou
                torch.save(self.model.state_dict(), self.best_miou_path)
                print(f"✓ [Best mIoU] Val mIoU↑ {vl_miou:.4f} → {self.best_miou_path}")

            # 마지막 체크포인트 갱신
            torch.save({
                "epoch": ep+1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict()
            }, self.last_ckpt_path)
            print(f"[Epoch {ep+1}/{self.num_epochs}] Train mIoU {tr_miou:.3f} | Val mIoU {vl_miou:.3f}")

        # 학습 완료 후, 두 가지 체크포인트로 최종 테스트
        for label, pth in [("best_val_loss", self.best_loss_path), ("best_val_miou", self.best_miou_path)]:
            self._load_ckpt(pth)
            ts_tot, ts_seg, ts_miou, ts_ciou, ts_w = self._test(test_loader)

            # 픽셀·클래스 정확도 계산
            all_tp = torch.zeros(self.n_classes, device=self.device)
            all_fp = torch.zeros(self.n_classes, device=self.device)
            with torch.no_grad():
                for _, clean, seg, _, _ in tqdm(test_loader, desc=f"[Test {label}]", leave=False):
                    img = clean.to(self.device)
                    seg = seg.to(self.device)
                    out = resize_inference(self.model, img)
                    preds = out.argmax(1)
                    tp, fp, fn, tn = get_stats(preds, seg, mode="multiclass",
                                               num_classes=self.n_classes,
                                               ignore_index=-1)
                    if tp.dim() == 2:
                        tp = tp.sum(dim=0); fp = fp.sum(dim=0)
                    all_tp += tp.to(self.device)
                    all_fp += fp.to(self.device)

            pixel_acc = float(all_tp.sum() / (all_tp.sum() + all_fp.sum()).clamp(min=1))
            valid = (all_tp + all_fp) > 0
            class_acc = float((all_tp[valid] / (all_tp[valid] + all_fp[valid])).mean()) if valid.any() else 0.0

            # 결과 출력 및 CSV 기록
            print(f"\n=== Test on [{label}] ===")
            print(f" Total Loss: {ts_tot:.4f}, Seg Loss: {ts_seg:.4f}")
            print(f" mIoU: {ts_miou:.4f}, Pixel Acc: {pixel_acc:.4f}, Class Acc: {class_acc:.4f}")
            for wid in range(4):
                print(f"  - {WEATHER_NAMES[wid]} | loss {ts_w[wid]['loss']:.4f}, mIoU {ts_w[wid]['mIoU']:.4f}")
            for cls, iou in ts_ciou.items():
                print(f"  - class {cls} IoU: {iou:.4f}")

            self._append_test_csv(
                epoch=self.num_epochs,
                tot_loss=ts_tot,
                seg_loss=ts_seg,
                miou=ts_miou,
                pixel_acc=pixel_acc,
                class_acc=class_acc,
                per_weather=ts_w,
                class_iou=ts_ciou
            )
