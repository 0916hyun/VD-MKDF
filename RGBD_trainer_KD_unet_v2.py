# RGBD_trainer_KD_unet_v1.py
# -*- coding: utf-8 -*-
import os, csv, math
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

from utils.stream_metrics import StreamSegMetrics
from segmentation_models_pytorch.metrics.functional import get_stats


# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────
def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def _to_scalar(x): return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)


# ─────────────────────────────────────────────────────────────
# 메인 티쳐(EncUpdate) 피처/로짓/게이트 추출 (v1 확장)
#  - fused_mean[i] = (xr + xd)/2
#  - gates[i] = PixelGatingModule의 g
# ─────────────────────────────────────────────────────────────
def _get_attr(m: nn.Module, *names):
    for n in names:
        if hasattr(m, n): return getattr(m, n)
    raise AttributeError(f"Missing any of attributes: {names}")

@torch.no_grad()
def extract_teacher_main_feats(teacher: nn.Module, rgb: torch.Tensor, depth: torch.Tensor
                              ) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
    teacher.eval()

    depth_proj = _get_attr(teacher, "depth_proj")
    rgb_full   = _get_attr(teacher, "rgb")
    dep_full   = _get_attr(teacher, "dep")
    rgb_enc    = getattr(rgb_full, "encoder", rgb_full)
    dep_enc    = getattr(dep_full, "encoder", dep_full)

    rgb_attn   = _get_attr(teacher, "rgb_attn", "rgb_attns")
    dep_attn   = _get_attr(teacher, "dep_attn", "depth_attns")
    conv_r     = _get_attr(teacher, "conv_r", "fuse_conv_r")
    conv_d     = _get_attr(teacher, "conv_d", "fuse_conv_d")
    px_gate    = _get_attr(teacher, "px_gate", "px_gates")
    fuse_set   = set(_get_attr(teacher, "fuse", "fuse_stages"))
    _run_stage = _get_attr(teacher, "_run_stage")
    decode     = _get_attr(teacher, "decode_head")

    x_d = depth_proj(depth)

    xr_in, xd_in = rgb, x_d
    xr_list, xd_list, fused_mean_list, gates_list = [], [], [], []
    for i in range(4):
        xr = _run_stage(rgb_enc, xr_in, i)
        xd = _run_stage(dep_enc, xd_in, i)

        if i in fuse_set:
            or_out, *_ = rgb_attn[i](xr)
            od_out, *_ = dep_attn[i](xd)
            cat = torch.cat([or_out, od_out], 1)
            ar  = conv_r[i](cat)
            ad  = conv_d[i](cat)
            or_new, od_new, g = px_gate[i](ar, ad)   # ← g 받기
            xr = (xr + or_new)/2
            xd = (xd + od_new)/2
            gates_list.append(g)                     # stage i의 g 저장
        else:
            gates_list.append(torch.zeros((xr.size(0), 1, xr.size(2), xr.size(3)),
                                          device=xr.device, dtype=xr.dtype))

        fused_mean_list.append((xr + xd)/2)
        xr_list.append(xr); xd_list.append(xd)
        xr_in, xd_in = xr, xd

    logits = decode([xr + xd for xr, xd in zip(xr_list, xd_list)])
    return {"xr": xr_list, "xd": xd_list, "fused_mean": fused_mean_list, "gates": gates_list, "logits": logits}


# ─────────────────────────────────────────────────────────────
# HF SegFormer(B1/B5 등)에서 hidden_states + logits 추출
# ─────────────────────────────────────────────────────────────
def extract_hf_feats_logits(model: nn.Module, rgb: torch.Tensor
                           ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    target = getattr(model, "model", model)
    out = target(rgb, output_hidden_states=True, return_dict=True)
    feats = list(out.hidden_states)  # 4 stages 예상
    logits = out.logits
    if logits is None and torch.is_tensor(out):
        logits = out
    return feats, logits


# ─────────────────────────────────────────────────────────────
# 학생: smp.Unet(ResNet50) + 1×1 PEA(채널 정렬) 래퍼
# ─────────────────────────────────────────────────────────────
class SMPUnetKDWrapper(nn.Module):
    TEACHER_DIMS = [64, 128, 320, 512]

    def __init__(self, num_classes: int, encoder_name: str = "resnet50",
                 encoder_weights: str | None = "imagenet"):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None
        )
        ch = list(self.net.encoder.out_channels)  # e.g. [3,64,256,512,1024,2048]
        assert len(ch) >= 6, f"Unexpected encoder channels: {ch}"
        self.student_dims = [ch[-4], ch[-3], ch[-2], ch[-1]]  # 256,512,1024,2048

        # PEA: student_ch -> teacher_ch
        self.pea = nn.ModuleList([
            nn.Identity() if s == t else nn.Conv2d(s, t, 1, bias=False)
            for s, t in zip(self.student_dims, self.TEACHER_DIMS)
        ])

    def _decode(self, feats):
        try:
            return self.net.decoder(*feats)    # 신버전
        except TypeError:
            return self.net.decoder(feats)     # 구버전

    def forward(self, x: torch.Tensor):
        feats = self.net.encoder(x)
        dec   = self._decode(feats)
        logits= self.net.segmentation_head(dec)
        return logits

    def feats_and_logits(self, x: torch.Tensor
                        ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        feats = self.net.encoder(x)
        s_feats_raw = [feats[-4], feats[-3], feats[-2], feats[-1]]  # 256,512,1024,2048
        dec   = self._decode(feats)
        logits= self.net.segmentation_head(dec)
        return s_feats_raw, logits

    def project_to_teacher(self, s_feats: List[torch.Tensor],
                           t_ref_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        proj = []
        for i, (sf, tf) in enumerate(zip(s_feats, t_ref_feats)):
            sf = self.pea[i](sf)
            if sf.shape[-2:] != tf.shape[-2:]:
                sf = F.interpolate(sf, size=tf.shape[-2:], mode="bilinear", align_corners=False)
            proj.append(sf)
        return proj



# ─────────────────────────────────────────────────────────────
# KD 트레이너 (v1: CE + feature L1 + logit L1) + Gating Distill 추가
# + CSV 로깅 강화(원본 raw 손실 및 람다 기록, Val/Test에 per-class IoU)
# ─────────────────────────────────────────────────────────────
class KDTrainerUNetV2(nn.Module):
    def __init__(
        self,
        student: SMPUnetKDWrapper,
        teacher_main: nn.Module,
        teachers_sub: List[nn.Module],
        classifier: nn.Module,
        device: str,

        # 학습/로그/테스트
        n_classes: int,
        ignore_index: int,
        lr: float,
        num_epochs: int,
        weights_dir: str,
        results_dir: str,
        test_after: int,
        test_infer_size: Tuple[int, int],

        # gating distill
        lambda_ce: float = 1.0,
        lambda_logit: float = 1.0,
        lambda_feat: float = 1.0,
        lambda_gate: float = 0.10,
        gate_stage: int = 3,
    ):
        super().__init__()
        self.student       = student
        self.teacher_main  = teacher_main
        self.teachers_sub  = teachers_sub
        self.classifier    = classifier

        self.device        = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_classes     = n_classes
        self.ignore_index  = ignore_index
        self.lr            = lr
        self.num_epochs    = num_epochs
        self.weights_dir   = weights_dir
        self.results_dir   = results_dir
        self.test_after    = test_after
        self.test_infer_hw = test_infer_size

        self.lambda_ce = float(lambda_ce)
        self.lambda_logit = float(lambda_logit)
        self.lambda_feat = float(lambda_feat)
        self.lambda_gate = float(lambda_gate)
        self.gate_stage = int(max(0, min(3, gate_stage)))

        # 옵티마이저: 학생(UNet + PEA)만 학습
        self.opt = torch.optim.Adam(self.student.parameters(), lr=self.lr)

        # 메트릭
        self.metric_tr = StreamSegMetrics(self.n_classes)
        self.metric_vl = StreamSegMetrics(self.n_classes)
        self.metric_ts = StreamSegMetrics(self.n_classes)

        # 경로 & 체크포인트
        _ensure_dir(self.weights_dir); _ensure_dir(self.results_dir)
        self.best_val_loss_path = os.path.join(self.weights_dir, "best_val_loss.pth")
        self.best_val_miou_path = os.path.join(self.weights_dir, "best_val_mIoU.pth")
        self.best_test_miou_path= os.path.join(self.weights_dir, "best_test_mIoU.pth")
        self.last_ckpt_path     = os.path.join(self.weights_dir, "last_ckpt.pth")

        self.csv_train    = os.path.join(self.results_dir, "train.csv")
        self.csv_val      = os.path.join(self.results_dir, "val.csv")
        self.csv_test_all = os.path.join(self.results_dir, "test_all.csv")
        self.csv_test_log = os.path.join(self.results_dir, "test_best.csv")

        self.best_val_loss = float("inf")
        self.best_val_miou = -1.0
        self.best_test_miou= -1.0

        self._init_csv(self.csv_train)
        self._init_test_csv(self.csv_val)
        self._init_test_csv(self.csv_test_log)
        self._init_test_csv(self.csv_test_all)

        # ImageNet 정규화(분류기용)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
        self.register_buffer("imnet_mean", mean, persistent=False)
        self.register_buffer("imnet_std",  std,  persistent=False)

    # CSV 헤더(Train/Val/Test)
    def _init_csv(self, path):
        header = [
            "epoch",
            # raw losses (λ 미적용)
            "ce_raw",
            "logit_main_raw","logit_sub0_raw","logit_sub1_raw","logit_sub2_raw","logit_sub3_raw","logit_kd_raw",
            "feat_main_raw","feat_sub0_raw","feat_sub1_raw","feat_sub2_raw","feat_sub3_raw","feat_kd_raw",
            "gate_kd_raw",
            # totals
            "total_raw","total_loss",
            # hyper-params used this epoch (현재 조절 가능: lambda_gate)
            "lambda_ce","lambda_logit","lambda_feat","lambda_gate",
            # quick metric
            "mIoU"
        ]
        with open(path, "w", newline="") as f: csv.writer(f).writerow(header)

    def _init_test_csv(self, path):
        header = ["epoch", "seg_loss", "mIoU", "pixAcc", "clsAcc"] + [f"classIoU_{c}" for c in range(self.n_classes)]
        with open(path, "w", newline="") as f: csv.writer(f).writerow(header)

    def _append_csv(self, path, row):
        with open(path, "a", newline="") as f: csv.writer(f).writerow(row)

    # 배치 언팩 (로더 규약: (inp, tar, depth, seg, wid))
    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) >= 5:
            rgb, depth, label = batch[0], batch[2], batch[3]      # ← 규약 유지
            if label.dim() == 4 and label.size(1) == 1: label = label.squeeze(1)
            return rgb, depth, label.long()
        if isinstance(batch, dict):
            rgb   = batch.get("rgb")   or batch.get("image")
            depth = batch.get("depth") or batch.get("dep")
            label = batch.get("label") or batch.get("seg") or batch.get("mask")
            if label.dim() == 4 and label.size(1) == 1: label = label.squeeze(1)
            return rgb, depth, label.long()
        raise TypeError("Unknown batch type from DataLoader.")

    @staticmethod
    def _student_conf_from_logits(s_up: torch.Tensor, n_classes: int) -> torch.Tensor:
        """학생 로짓(업샘플) → softmax → 엔트로피 정규화 → confidence=1−H/lnC"""
        p = F.softmax(s_up, dim=1)
        H = -(p * (p.clamp_min(1e-8)).log()).sum(dim=1, keepdim=True)  # [B,1,H,W]
        H = H / math.log(max(2, n_classes))
        conf = 1.0 - H
        return conf

    # ───────────── Train/Val 한 에폭 ─────────────
    def _epoch(self, loader: DataLoader, is_train: bool, epoch: int):
        """
        Train:  KD(v1) = CE + featKD + logitKD (+ gateKD 추가)
        Valid:  CE + mIoU + per-class IoU/Acc (KD 완전 비활성화)
        """
        self.student.train(is_train)

        if is_train:
            # Train용 간단 mIoU 추적
            metric = self.metric_tr
            metric.reset()
            pbar = tqdm(loader, ncols=110, desc=f"Train [{epoch + 1}/{self.num_epochs}]")

            ce_sum = fk_sum = lk_sum = gk_sum = 0.0
            # raw breakdown (λ 미적용 누적)
            logit_main_sum = 0.0
            logit_sub_sum = [0.0, 0.0, 0.0, 0.0]
            feat_main_sum  = 0.0
            feat_sub_sum   = [0.0, 0.0, 0.0, 0.0]
            cnt = 0

            alpha_main = 0.5

            for batch in pbar:
                rgb, depth, label = self._unpack_batch(batch)
                rgb = rgb.to(self.device, non_blocking=True)
                depth = depth.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                if depth.dim() != 4 or depth.size(1) != 1:
                    raise RuntimeError(f"Depth must be 1ch (got {tuple(depth.shape)})")

                B = rgb.size(0)
                Ht, Wt = label.shape[-2], label.shape[-1]

                # 1) 메인 티쳐 (feat/logits/gates)
                t_main = extract_teacher_main_feats(self.teacher_main, rgb, depth)
                t_up = F.interpolate(t_main["logits"], size=(Ht, Wt), mode="bilinear", align_corners=False)

                # 게이트 맵(g) 선택 스테이지 → 업샘플
                g_list = t_main.get("gates", None)
                if g_list is not None and len(g_list) > 0:
                    g_teacher = g_list[self.gate_stage]
                    g_up = F.interpolate(g_teacher, size=(Ht, Wt), mode="bilinear", align_corners=False).detach()
                else:
                    g_up = torch.zeros((B,1,Ht,Wt), device=rgb.device, dtype=t_up.dtype)

                # 2) 학생 (feat + logits)
                s_feats_raw, s_logits = self.student.feats_and_logits(rgb)
                s_up = F.interpolate(s_logits, size=(Ht, Wt), mode="bilinear", align_corners=False)

                # 학생 피처를 교사 피처(fused_mean) 해상도/채널로 정렬
                t_ref_feats = [t_main["fused_mean"][i].detach() for i in range(4)]
                s_feats = self.student.project_to_teacher(s_feats_raw, t_ref_feats)

                # 3) 서브티쳐 (feat + logits, no_grad)
                sub_feats_list = []
                sub_logits_up = []
                with torch.no_grad():
                    for tch in self.teachers_sub:
                        fts, lg = extract_hf_feats_logits(tch, rgb)
                        sub_feats_list.append(fts)
                        sub_logits_up.append(F.interpolate(lg, size=(Ht, Wt), mode="bilinear", align_corners=False))

                # 4) α_sub (분류기 기반) — 선택 0.35, 나머지 0.05
                with torch.no_grad():
                    x = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
                    x = (x - self.imnet_mean) / self.imnet_std
                    sel = self.classifier(x).argmax(1)  # (B,)
                    alpha_sub = torch.full((B, len(self.teachers_sub)), 0.05, device=self.device, dtype=s_up.dtype)
                    alpha_sub.scatter_(1, sel.unsqueeze(1).clamp_max(len(self.teachers_sub) - 1), 0.35)

                # 5) 손실
                ce_loss = F.cross_entropy(s_up, label, ignore_index=self.ignore_index)

                # Logit KD (원본 v1 유지: L1)
                L_logit_main = F.l1_loss(s_up, t_up)  # scalar
                L_logit_sub_total = 0.0
                _per_sub_logit = []  # raw per-sub (배치 평균)
                for k in range(len(self.teachers_sub)):
                    l1k = F.l1_loss(s_up, sub_logits_up[k], reduction='none').mean(dim=(1, 2, 3))  # (B,)
                    l1k_mean = l1k.mean()
                    _per_sub_logit.append(l1k_mean)
                    L_logit_sub_total = L_logit_sub_total + (l1k * alpha_sub[:, k]).mean()
                logit_kd = alpha_main * L_logit_main + L_logit_sub_total

                # Feature KD (원본 v1 유지: L1)
                per_sample_main = 0.0
                for i in range(4):
                    tf = t_ref_feats[i]
                    sf = s_feats[i]
                    per_sample_main = per_sample_main + F.l1_loss(sf, tf, reduction='none').mean(dim=(1, 2, 3))
                L_feat_main = (per_sample_main / 4.0).mean()

                L_feat_sub_total = 0.0
                _per_sub_feat = []  # raw per-sub (배치 평균)
                for k in range(len(self.teachers_sub)):
                    per_k = 0.0
                    for i in range(4):
                        tf = sub_feats_list[k][i].detach()
                        sf = s_feats[i]
                        if sf.shape[-2:] != tf.shape[-2:]:
                            sf = F.interpolate(sf, size=tf.shape[-2:], mode="bilinear", align_corners=False)
                        c = min(sf.size(1), tf.size(1))
                        per_k = per_k + F.l1_loss(sf[:, :c], tf[:, :c], reduction='none').mean(dim=(1, 2, 3))
                    per_k = (per_k / 4.0)
                    _per_sub_feat.append(per_k.mean())
                    L_feat_sub_total = L_feat_sub_total + (per_k * alpha_sub[:, k]).mean()
                feat_kd = alpha_main * L_feat_main + L_feat_sub_total

                # Gating Distill (추가): teacher g ↔ student confidence (원본 로직 불변, 추가만)
                valid = (label != self.ignore_index).float().unsqueeze(1)  # [B,1,H,W]
                s_conf = self._student_conf_from_logits(s_up, self.n_classes)  # grad ON
                gate_kd_map = (s_conf - g_up).abs()                           # L1
                gate_kd = (gate_kd_map * valid).sum() / (valid.sum() + 1e-6)

                # 총 손실 (기존 ce/feat/logit λ=1, gate만 λ 조절)
                total = (
                        self.lambda_ce * ce_loss
                        + self.lambda_feat * feat_kd
                        + self.lambda_logit * logit_kd
                        + self.lambda_gate * gate_kd
                )

                self.opt.zero_grad(set_to_none=True)
                total.backward()
                self.opt.step()

                # mIoU (빠른 추적)
                with torch.no_grad():
                    preds = torch.argmax(s_up, 1)
                    label_for_metric = label.clone()
                    label_for_metric[label_for_metric == self.ignore_index] = -1
                    lt_np = label_for_metric.detach().cpu().numpy()
                    lp_np = preds.detach().cpu().numpy()
                    metric.update(lt_np, lp_np)
                    miou = float(metric.get_results()["Mean IoU"])

                # 누적(평균 계산용)
                ce_sum += float(ce_loss.detach().cpu()) * B
                fk_sum += float(feat_kd) * B
                lk_sum += float(logit_kd) * B
                gk_sum += float(gate_kd.detach().cpu()) * B
                # raw breakdown
                logit_main_sum += float(L_logit_main.detach().cpu()) * B
                for k in range(4): logit_sub_sum[k] += float(_per_sub_logit[k].detach().cpu()) * B
                feat_main_sum  += float(L_feat_main.detach().cpu())  * B
                for k in range(4): feat_sub_sum[k]  += float(_per_sub_feat[k].detach().cpu()) * B

                cnt += B
                pbar.set_postfix(
                    ce=f"{ce_sum / cnt:.3f}",
                    fk=f"{fk_sum / cnt:.3f}",
                    lk=f"{lk_sum / cnt:.3f}",
                    gk=f"{gk_sum / cnt:.3f}",
                    mIoU=f"{miou:.3f}"
                )

            # 에폭 평균 & CSV row 구성
            ce_avg = ce_sum / max(1, cnt)
            fk_avg = fk_sum / max(1, cnt)
            lk_avg = lk_sum / max(1, cnt)
            gk_avg = gk_sum / max(1, cnt)

            # raw breakdown 평균
            logit_main_avg = logit_main_sum / max(1, cnt)
            logit_sub_avg  = [v / max(1, cnt) for v in logit_sub_sum]
            feat_main_avg  = feat_main_sum  / max(1, cnt)
            feat_sub_avg   = [v / max(1, cnt) for v in feat_sub_sum]

            # raw total (참고용): λ 미적용 합
            total_raw = ce_avg + (feat_main_avg + sum(feat_sub_avg)) + (logit_main_avg + sum(logit_sub_avg)) + gk_avg
            # 실제 학습 total (기존 로직 유지)
            res = metric.get_results()
            miou = float(res["Mean IoU"])
            total = 1.0 * ce_avg + 1.0 * fk_avg + 1.0 * lk_avg + self.lambda_gate * gk_avg

            train_row = [
                epoch+1,
                # raw
                ce_avg,
                logit_main_avg, *logit_sub_avg, (logit_main_avg + sum(logit_sub_avg)),
                feat_main_avg,  *feat_sub_avg,  (feat_main_avg  + sum(feat_sub_avg)),
                gk_avg,
                # totals
                total_raw, total,
                # lambdas (조절 가능한 하이퍼파라미터 로깅)
                float(self.lambda_ce), float(self.lambda_logit), float(self.lambda_feat), float(self.lambda_gate),
                miou
            ]
            return total, ce_avg, fk_avg, lk_avg, miou, train_row

        else:
            # VALID: KD 스킵 → seg loss & metrics만
            metric = self.metric_vl
            metric.reset()
            pbar = tqdm(loader, ncols=110, desc=f"Valid [{epoch + 1}/{self.num_epochs}]")

            ce_sum, cnt = 0.0, 0
            tp_sum = torch.zeros(self.n_classes, dtype=torch.long)
            fp_sum = torch.zeros(self.n_classes, dtype=torch.long)
            fn_sum = torch.zeros(self.n_classes, dtype=torch.long)

            for batch in pbar:
                rgb, depth, label = self._unpack_batch(batch)
                rgb = rgb.to(self.device, non_blocking=True)
                depth = depth.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True).long()

                if depth.dim() != 4 or depth.size(1) != 1:
                    raise RuntimeError(f"Depth must be 1ch (got {tuple(depth.shape)})")

                logits = self.student(rgb)
                s_up = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)

                ce_loss = F.cross_entropy(s_up, label, ignore_index=self.ignore_index)
                ce_sum += float(ce_loss.detach().cpu()) * rgb.shape[0]
                cnt    += rgb.shape[0]

                preds = torch.argmax(s_up, 1)
                label_for_metric = label.clone()
                label_for_metric[label_for_metric == self.ignore_index] = -1
                lt_np = label_for_metric.detach().cpu().numpy()
                lp_np = preds.detach().cpu().numpy()
                metric.update(lt_np, lp_np)

                stats = get_stats(preds, label_for_metric, mode="multiclass",
                                  num_classes=self.n_classes, ignore_index=-1)
                if hasattr(stats, "tp"):
                    tp, fp, fn = stats.tp, stats.fp, stats.fn
                else:
                    tp, fp, fn = stats[0], stats[1], stats[2]
                if tp.dim()>1: tp=tp.sum(0); fp=fp.sum(0); fn=fn.sum(0)
                tp_sum += tp.cpu().long(); fp_sum += fp.cpu().long(); fn_sum += fn.cpu().long()

                miou = float(metric.get_results()["Mean IoU"])
                pbar.set_postfix(seg=f"{ce_sum/max(1,cnt):.3f}", mIoU=f"{miou:.3f}")

            res = metric.get_results()
            miou = float(res["Mean IoU"])
            seg_loss_avg = ce_sum / max(1, cnt)
            pix_acc = (tp_sum.sum().float() / ((tp_sum + fp_sum).sum().float() + 1e-6)).item()
            cls_acc = (tp_sum.float() / (tp_sum.float() + fn_sum.float() + 1e-6)).mean().item()
            class_iou = (tp_sum.float() / (tp_sum.float() + fp_sum.float() + fn_sum.float() + 1e-6)).tolist()

            val_row = [epoch+1, seg_loss_avg, miou, pix_acc, cls_acc, *[float(x) for x in class_iou]]
            return val_row

    # ───────────── 테스트 ─────────────
    @torch.no_grad()
    def _test(self, loader: DataLoader, epoch: Optional[int]=None):
        self.student.eval()
        self.metric_ts.reset()

        ce_sum, cnt = 0.0, 0
        tp_sum = torch.zeros(self.n_classes, dtype=torch.long)
        fp_sum = torch.zeros(self.n_classes, dtype=torch.long)
        fn_sum = torch.zeros(self.n_classes, dtype=torch.long)

        ep_tag = f" [{epoch+1}/{self.num_epochs}]" if epoch is not None else ""
        for batch in tqdm(loader, ncols=100, desc="Test"+ep_tag):
            rgb, depth, label = self._unpack_batch(batch)
            rgb   = rgb.to(self.device); depth = depth.to(self.device); label = label.to(self.device).long()

            if all(x>0 for x in self.test_infer_hw):
                Ht, Wt = self.test_infer_hw
                rgb_in = F.interpolate(rgb, size=(Ht,Wt), mode="bilinear", align_corners=False)
            else:
                rgb_in = rgb

            logits = self.student(rgb_in)
            s_up = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)

            ce_loss = F.cross_entropy(s_up, label, ignore_index=self.ignore_index)
            ce_sum += float(ce_loss.detach().cpu()) * rgb.shape[0]
            cnt    += rgb.shape[0]

            preds = torch.argmax(s_up, 1)
            label_for_metric = label.clone(); label_for_metric[label_for_metric==self.ignore_index] = -1
            lt_np = label_for_metric.detach().cpu().numpy()
            lp_np = preds.detach().cpu().numpy()
            self.metric_ts.update(lt_np, lp_np)

            stats = get_stats(preds, label_for_metric, mode="multiclass",
                              num_classes=self.n_classes, ignore_index=-1)
            if hasattr(stats, "tp"):
                tp, fp, fn = stats.tp, stats.fp, stats.fn
            else:
                tp, fp, fn = stats[0], stats[1], stats[2]
            if tp.dim()>1: tp=tp.sum(0); fp=fp.sum(0); fn=fn.sum(0)
            tp_sum += tp.cpu().long(); fp_sum += fp.cpu().long(); fn_sum += fn.cpu().long()

        res = self.metric_ts.get_results(); miou = float(res["Mean IoU"])
        total_correct = tp_sum.sum().float()
        total_pixels  = (tp_sum + fp_sum).sum().float()
        pix_acc = (total_correct/(total_pixels+1e-6)).item()
        cls_acc = (tp_sum.float()/(tp_sum.float()+fn_sum.float()+1e-6)).mean().item()
        class_iou = (tp_sum.float() / (tp_sum.float() + fp_sum.float() + fn_sum.float() + 1e-6)).tolist()

        return ce_sum/max(1,cnt), ce_sum/max(1,cnt), miou, pix_acc, cls_acc, class_iou

    # ───────────── 에폭 루프 ─────────────
    def train_loop(self, train_loader: DataLoader, val_loader: DataLoader,
                   test_loader: DataLoader, resume_ckpt: str=""):
        # resume(optional)
        start_ep = 0
        if resume_ckpt and os.path.isfile(resume_ckpt):
            ckpt = torch.load(resume_ckpt, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.student.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt: self.opt.load_state_dict(ckpt["optimizer_state_dict"])
                start_ep = int(ckpt.get("epoch", 0))
            else:
                self.student.load_state_dict(ckpt)
            print(f"▶ Resumed from {resume_ckpt} @ epoch {start_ep}")

        for ep in range(start_ep, self.num_epochs):
            tr_tot, tr_ce, tr_fk, tr_lk, tr_miou, tr_row = self._epoch(train_loader, True,  ep)
            vl_row = self._epoch(val_loader,   False, ep)

            self._append_csv(self.csv_train, tr_row)
            self._append_csv(self.csv_val,   vl_row)

            if vl_row[1] < getattr(self, "best_val_loss", float("inf")):
                self.best_val_loss = vl_row[1]
                torch.save(self.student.state_dict(), self.best_val_loss_path)
            if vl_row[2] > getattr(self, "best_val_miou", -1.0):
                self.best_val_miou = vl_row[2]
                torch.save(self.student.state_dict(), self.best_val_miou_path)

            if (ep + 1) >= self.test_after:
                ts_tot, ts_ce, ts_miou, pix_acc, cls_acc, cls_iou = self._test(test_loader, epoch=ep)
                self._append_csv(self.csv_test_all, [ep+1, ts_tot, ts_ce, ts_miou, pix_acc, cls_acc, *cls_iou])
                if ts_miou > getattr(self, "best_test_miou", -1.0):
                    self.best_test_miou = ts_miou
                    torch.save(self.student.state_dict(), self.best_test_miou_path)
                    self._append_csv(self.csv_test_log, [ep+1, ts_tot, ts_ce, ts_miou, pix_acc, cls_acc, *cls_iou])
                    print(f"[Epoch {ep+1}] New Best Test mIoU {ts_miou:.4f} → {self.best_test_miou_path}")

            torch.save({"epoch": ep+1,
                        "model_state_dict": self.student.state_dict(),
                        "optimizer_state_dict": self.opt.state_dict()}, self.last_ckpt_path)
