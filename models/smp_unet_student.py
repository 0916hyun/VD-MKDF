# models/smp_unet_student.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import List, Tuple

class SMPUnetKDWrapper(nn.Module):
    """
    Student: SMP Unet (e.g., encoder=resnet50) + PEA(1x1) for KD feature alignment.
    - feats_and_logits(x) -> ( [f2..f5], logits ) with grad
    - project_to_teacher(s_feats, t_ref_feats) -> channel(HxW) aligned for KD
    """
    TEACHER_DIMS = [64, 128, 320, 512]  # target chans of SegFormer teachers

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
        self.student_dims = [ch[-4], ch[-3], ch[-2], ch[-1]]  # f2..f5: 256,512,1024,2048 (resnet50)

        # PEA: student_ch -> teacher_ch
        self.pea = nn.ModuleList([
            nn.Identity() if s == t else nn.Conv2d(s, t, 1, bias=False)
            for s, t in zip(self.student_dims, self.TEACHER_DIMS)
        ])

    def _decode(self, feats):
        """SMP 버전별 decoder 시그니처 차이 대응"""
        try:   return self.net.decoder(*feats)   # 신버전
        except TypeError:
               return self.net.decoder(feats)    # 구버전

    def forward(self, x: torch.Tensor):
        """추론 경로(배포 기준): logits만. PEA는 사용하지 않음."""
        feats = self.net.encoder(x)          # [f0..f5]
        dec   = self._decode(feats)
        logits= self.net.segmentation_head(dec)
        return logits

    def feats_and_logits(self, x: torch.Tensor
                        ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """학습 경로(KD 계산용): f2..f5 + logits (Grad ON)"""
        feats = self.net.encoder(x)
        s_feats_raw = [feats[-4], feats[-3], feats[-2], feats[-1]]
        dec   = self._decode(feats)
        logits= self.net.segmentation_head(dec)
        return s_feats_raw, logits

    def project_to_teacher(self, s_feats: List[torch.Tensor],
                           t_ref_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """학생→교사 정렬: 1×1 채널 정렬 + H×W bilinear 정렬"""
        proj = []
        for i, (sf, tf) in enumerate(zip(s_feats, t_ref_feats)):
            sf = self.pea[i](sf)
            if sf.shape[-2:] != tf.shape[-2:]:
                sf = F.interpolate(sf, size=tf.shape[-2:], mode="bilinear", align_corners=False)
            proj.append(sf)
        return proj
