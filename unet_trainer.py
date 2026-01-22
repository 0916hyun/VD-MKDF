import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, csv, numpy as np, torch, matplotlib.pyplot as plt
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


def _extract_logits(output):
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        if "sem_seg" in output:
            return output["sem_seg"]
        if "out" in output:
            return output["out"]
        raise KeyError("Dictionary output missing 'logits'/'sem_seg' keys.")
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def resize_inference(model, img, dst=(352, 480)):
    B, _, H, W = img.shape
    img_rs = torch.nn.functional.interpolate(img, dst, mode="bilinear", align_corners=False)
    out_rs = model(img_rs)
    logits = _extract_logits(out_rs)
    return torch.nn.functional.interpolate(logits, (H, W), mode="bilinear", align_corners=False)


class Trainer(nn.Module):
    def __init__(self, model: nn.Module, save_root: str, weights_dir: str, results_dir: str,
                 device: torch.device, lr: float, num_epochs: int, n_classes: int,
                 ignore_index: int = 11,
                 test_after: int = 400):  # ★ 변경: test_after 추가
        super().__init__()
        self.device, self.lr, self.num_epochs = device, lr, num_epochs
        self.model = model.to(self.device)
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.test_after = int(test_after)   # ★ 변경

        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        self.weights_dir, self.results_dir = weights_dir, results_dir

        self.last_ckpt_path = os.path.join(weights_dir, "last_epoch.pth")

        # self.model = smp.UnetPlusPlus(encoder_name="resnet50",
        #                               encoder_weights="imagenet",
        #                               classes=n_classes,
        #                               activation=None).to(device)
        self.model = model
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.crit_seg = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
        self.metrics = StreamSegMetrics(n_classes)

        self.best_val_loss = float('inf')
        self.best_val_miou = 0.0
        self.best_loss_path = os.path.join(weights_dir, "best_val_loss.pth")
        self.best_miou_path = os.path.join(weights_dir, "best_val_miou.pth")

        self.model_supports_targets = getattr(self.model, "supports_target_input", False)

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
        header = [
            "epoch", "total_loss", "seg_loss", "mIoU", "pixel_acc", "class_acc"
        ]
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

    def _merge_losses(self, loss_value):
        if loss_value is None:
            return None
        if isinstance(loss_value, torch.Tensor):
            return loss_value
        if isinstance(loss_value, dict):
            items = [v for v in loss_value.values() if v is not None]
        elif isinstance(loss_value, (list, tuple)):
            items = [v for v in loss_value if v is not None]
        else:
            return torch.tensor(loss_value, device=self.device, dtype=torch.float32)
        if not items:
            return None
        total = items[0]
        for item in items[1:]:
            total = total + item
        return total

    def _parse_model_output(self, output):
        if isinstance(output, dict):
            logits = output.get("logits")
            if logits is None:
                raise KeyError("Dictionary output must contain 'logits'.")
            loss_value = self._merge_losses(output.get("loss"))
            aux_losses = self._merge_losses(output.get("aux_losses"))
            if aux_losses is not None:
                loss_value = aux_losses if loss_value is None else loss_value + aux_losses
            return logits, loss_value
        logits = _extract_logits(output)
        return logits, None

    def _save_ckpt(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict()
            },
            os.path.join(self.weights_dir, f"epoch_{epoch:04d}.pth")
        )

    def _load_ckpt(self, path):
        ckpt = torch.load(path, map_location=self.device)
        # plain state_dict인지, checkpoint dict인지 구분
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt  # 그냥 바로 state_dict
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            hint = (
                "\n[힌트] 모델 설정(HGFormer config, 모델 variant, num_classes 등)이 "
                "학습 때와 다르면 가중치 키/차원이 맞지 않아 로드가 실패할 수 있습니다. "
                "train.py에서 사용한 --model_variant / --hgformer_config / --hgformer_opts와 "
                "동일한 값으로 tester.py 또는 test.py를 실행했는지 확인하세요."
            )
            raise RuntimeError(str(e) + hint) from e

        # 옵티마이저 복원도 비슷하게
        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer_state_dict"])

        start_ep = ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0
        print(f"✓ Resumed from {path} (epoch {start_ep})")
        return start_ep

    def _epoch(self, loader, training: bool):
        self.model.train() if training else self.model.eval()
        self.metrics.reset()
        weather_stats = {wid: WeatherStat(self.n_classes) for wid in range(4)}

        tot_loss_sum = seg_loss_sum = 0.
        for inp, _, seg, wid, _ in tqdm(loader, desc="[Train]" if training else "[Val ]", unit="batch"):
            inp, seg = inp.to(self.device), seg.to(self.device)

            if training:
                self.opt.zero_grad()

            if self.model_supports_targets:
                out = self.model(inp, seg)
            else:
                out = self.model(inp)
            logits, provided_loss = self._parse_model_output(out)

            if logits.shape[-2:] != seg.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=seg.shape[-2:],  # (H, W)
                    mode='bilinear', align_corners=False
                )

            seg_loss = provided_loss if provided_loss is not None else self.crit_seg(logits, seg)
            tot_loss = seg_loss  # + kd_loss(0)

            if training:
                tot_loss.backward()
                self.opt.step()

            bsz = inp.size(0)
            tot_loss_sum += tot_loss.item()
            seg_loss_sum += seg_loss.item()

            preds  = logits.argmax(1).cpu().numpy()
            labels = seg.cpu().numpy()
            self.metrics.update(labels, preds)

            for b in range(bsz):
                w = int(wid[b])
                weather_stats[w].update(tot_loss.item(), labels[b:b+1], preds[b:b+1], 1)

        m = self.metrics.get_results()
        tot_avg = tot_loss_sum / len(loader)
        seg_avg = seg_loss_sum / len(loader)
        per_w = {
            wid: dict(zip(["loss", "mIoU", "class_iou"], weather_stats[wid].summary()))
            for wid in range(4)
        }
        return tot_avg, seg_avg, m['Mean IoU'], m['Class IoU'], per_w

    @torch.no_grad()
    def _test(self, loader):
        self.model.eval()
        self.metrics.reset()
        weather_stats = {wid: WeatherStat(self.n_classes) for wid in range(4)}

        tot_sum = seg_sum = 0.
        for inp, _, seg, wid, _ in tqdm(loader, desc="[Test]", unit="batch"):
            inp, seg = inp.to(self.device), seg.to(self.device)
            logits = resize_inference(self.model, inp)

            seg_loss = self.crit_seg(logits, seg)
            tot_loss = seg_loss  # + kd_loss(0)

            bsz = inp.size(0)
            tot_sum += tot_loss.item()
            seg_sum += seg_loss.item()

            preds  = logits.argmax(1).cpu().numpy()
            labels = seg.cpu().numpy()
            self.metrics.update(labels, preds)

            for b in range(bsz):
                w = int(wid[b])
                weather_stats[w].update(tot_loss.item(), labels[b:b+1], preds[b:b+1], 1)

        m = self.metrics.get_results()
        per_w = {
            wid: dict(zip(["loss", "mIoU", "class_iou"], weather_stats[wid].summary()))
            for wid in range(4)
        }
        return (tot_sum/len(loader), seg_sum/len(loader), m['Mean IoU'], m['Class IoU'], per_w)

    def train(self, train_loader, val_loader, test_loader, resume_ckpt=None):
        start_ep = 0
        if resume_ckpt:
            start_ep = self._load_ckpt(resume_ckpt)

        # best_test_miou 트래킹(초기화)
        self.best_test_miou = getattr(self, "best_test_miou", -1.0)

        for ep in range(start_ep, self.num_epochs):
            cur_epoch = ep + 1

            # ── Train / Valid
            tr_tot, tr_seg, tr_miou, tr_ciou, tr_w = self._epoch(train_loader, True)
            vl_tot, vl_seg, vl_miou, vl_ciou, vl_w = self._epoch(val_loader, False)

            # ── CSV 기록
            self._append_csv(self.csv_train, cur_epoch, tr_tot, tr_seg, tr_miou, tr_w, tr_ciou)
            self._append_csv(self.csv_val,   cur_epoch, vl_tot, vl_seg, vl_miou, vl_w, vl_ciou)

            self.hist_ep.append(cur_epoch)
            self.hist_tr_tot.append(tr_tot); self.hist_tr_seg.append(tr_seg)
            self.hist_va_tot.append(vl_tot); self.hist_va_seg.append(vl_seg)

            # ── Best 체크포인트 갱신 (오직 2개)
            if vl_tot < self.best_val_loss:
                self.best_val_loss = vl_tot
                torch.save(self.model.state_dict(), self.best_loss_path)
                print(f"✓ [Best Loss] Val Loss↓ {vl_tot:.4f} → {self.best_loss_path}")

            if vl_miou > self.best_val_miou:
                self.best_val_miou = vl_miou
                torch.save(self.model.state_dict(), self.best_miou_path)
                print(f"✓ [Best mIoU] Val mIoU↑ {vl_miou:.4f} → {self.best_miou_path}")

            print(f"Epoch {cur_epoch}/{self.num_epochs} | Train mIoU {tr_miou:.3f} || Val mIoU {vl_miou:.3f}")

            # ── 매 에포크 테스트(메모리 상 현재 모델로) ─────────────────
            if cur_epoch >= self.test_after:
                # 현재 모델로 바로 테스트 (ckpt 리로드 없음)
                ts_tot, ts_seg, ts_miou, ts_ciou, ts_w = self._test(test_loader)

                # pixel_acc / class_acc 계산 (confusion 기반)
                all_tp = torch.zeros(self.n_classes, device=self.device)
                all_fp = torch.zeros(self.n_classes, device=self.device)
                self.model.eval()
                with torch.no_grad():
                    for inp, _, seg, _, _ in tqdm(test_loader, desc=f"[Test epoch_{cur_epoch:04d}]", leave=False):
                        inp, seg = inp.to(self.device), seg.to(self.device)
                        out = resize_inference(self.model, inp)
                        preds = out.argmax(1)
                        tp, fp, fn, tn = get_stats(
                            preds, seg, mode="multiclass",
                            num_classes=self.n_classes, ignore_index=-1
                        )
                        if tp.dim() == 2:
                            tp = tp.sum(dim=0); fp = fp.sum(dim=0)
                        all_tp += tp.to(self.device)
                        all_fp += fp.to(self.device)

                pixel_acc = float(all_tp.sum() / (all_tp.sum() + all_fp.sum()).clamp(min=1))
                valid = (all_tp + all_fp) > 0
                class_acc = float((all_tp[valid] / (all_tp[valid] + all_fp[valid])).mean()) if valid.any() else 0.0

                print(f"\n=== Test on [epoch_{cur_epoch:04d}] ===")
                print(f" Total Loss: {ts_tot:.4f}, Seg Loss: {ts_seg:.4f}")
                print(f" mIoU: {ts_miou:.4f}, Pixel Acc: {pixel_acc:.4f}, Class Acc: {class_acc:.4f}")
                for wid in range(4):
                    print(f" - {WEATHER_NAMES[wid]} | loss {ts_w[wid]['loss']:.4f}, mIoU {ts_w[wid]['mIoU']:.4f}")
                for cls, iou in ts_ciou.items():
                    print(f" - class {cls} IoU: {iou:.4f}")

                # CSV 기록 (해당 에포크 번호로 기록) — 유지
                self._append_test_csv(
                    epoch=cur_epoch, tot_loss=ts_tot, seg_loss=ts_seg, miou=ts_miou,
                    pixel_acc=pixel_acc, class_acc=class_acc, per_weather=ts_w, class_iou=ts_ciou
                )

                # Best Test mIoU 모델 저장 (세 번째 파일)
                if ts_miou > getattr(self, "best_test_miou", -1.0):
                    self.best_test_miou = ts_miou
                    torch.save(self.model.state_dict(), os.path.join(self.weights_dir, "best_test_mIoU.pth"))
                    print(f"✓ [Best Test mIoU] Test mIoU↑ {ts_miou:.4f} → {os.path.join(self.weights_dir, 'best_test_mIoU.pth')}")

        # ── 전체 학습 종료 후: VALID 베스트 2개 각각 1회 테스트(유지) ─────
        for label, pth in [
                        ("best_val_miou", self.best_miou_path)]:  #("best_val_loss", self.best_loss_path),
            if not os.path.exists(pth):
                continue
            _ = self._load_ckpt(pth)
            ts_tot, ts_seg, ts_miou, ts_ciou, ts_w = self._test(test_loader)

            all_tp = torch.zeros(self.n_classes, device=self.device)
            all_fp = torch.zeros(self.n_classes, device=self.device)
            self.model.eval()
            with torch.no_grad():
                for inp, _, seg, _, _ in tqdm(test_loader, desc=f"[Final Test {label}]", leave=False):
                    inp, seg = inp.to(self.device), seg.to(self.device)
                    out = resize_inference(self.model, inp)
                    preds = out.argmax(1)
                    tp, fp, fn, tn = get_stats(
                        preds, seg, mode="multiclass",
                        num_classes=self.n_classes, ignore_index=-1
                    )
                    if tp.dim() == 2:
                        tp = tp.sum(dim=0); fp = fp.sum(dim=0)
                    all_tp += tp.to(self.device)
                    all_fp += fp.to(self.device)

            pixel_acc = float(all_tp.sum() / (all_tp.sum() + all_fp.sum()).clamp(min=1))
            valid = (all_tp + all_fp) > 0
            class_acc = float((all_tp[valid] / (all_tp[valid] + all_fp[valid])).mean()) if valid.any() else 0.0

            print(f"\n=== Final Test on [{label}] ===")
            print(f" Total Loss: {ts_tot:.4f}, Seg Loss: {ts_seg:.4f}")
            print(f" mIoU: {ts_miou:.4f}, Pixel Acc: {pixel_acc:.4f}, Class Acc: {class_acc:.4f}")
            for wid in range(4):
                print(f" - {WEATHER_NAMES[wid]} | loss {ts_w[wid]['loss']:.4f}, mIoU {ts_w[wid]['mIoU']:.4f}")
            for cls, iou in ts_ciou.items():
                print(f" - class {cls} IoU: {iou:.4f}")

            # 최종 결과도 같은 CSV에 기록(유지)
            self._append_test_csv(
                epoch=self.num_epochs, tot_loss=ts_tot, seg_loss=ts_seg, miou=ts_miou,
                pixel_acc=pixel_acc, class_acc=class_acc, per_weather=ts_w, class_iou=ts_ciou
            )