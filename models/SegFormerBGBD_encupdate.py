import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from custom_module.att import PatchChannelAttentionModule
from custom_module.att_res import PatchSpatialAttentionModule_res
from custom_module.pixel_gating import PixelGatingModule

class Segb5_RGBD_FuseStageSelective_EncUpdate_switch(nn.Module):
    """
    SegFormer-B5 Dual-Encoder RGB-D + Stage-Selective Fusion (Encoder-Update)

    블렌딩 모드:
      - 'avg'       : xr <- (xr + or_new)/2, xd <- (xd + od_new)/2
      - 'learnable' : xr <- α_r·xr + (1-α_r)·or_new,  xd <- α_d·xd + (1-α_d)·od_new
      - 'mulcoeff'  : xr <- xr * (1 + β_r·σ(W_r·cat)), xd <- xd * (1 + β_d·σ(W_d·cat))
                      (cat = concat(att_r, att_d),  W_r/W_d: 1×1 Conv, 채널별 계수)
    """
    def __init__(self,
                 n_classes: int,
                 patch_sizes: list[int],
                 fuse_stages: list[int],
                 blend: str = 'avg'):
        super().__init__()
        assert blend in {'avg', 'learnable', 'mulcoeff'}, f"blend={blend}"
        self.blend = blend

        # 1) HF SegFormer-B5 로드 (디코더/백본 재사용)
        hf = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
            num_labels=n_classes,
            ignore_mismatched_sizes=True,
            return_dict=True
        )
        self.rgb = hf.segformer
        self.dep = copy.deepcopy(hf.segformer)
        self.decode_head = hf.decode_head

        # 2) Depth 1ch → RGB 입력 채널수로 투영
        in_ch = self.rgb.config.num_channels  # 보통 3
        self.depth_proj = nn.Conv2d(1, in_ch, kernel_size=1)

        # 3) 스테이지별 채널 크기
        self.dims = list(self.rgb.config.hidden_sizes)  # [64, 128, 320, 512]
        assert len(patch_sizes) == 4
        self.patch_sizes = patch_sizes
        self.fuse = set(fuse_stages)

        # 4) Base attentions
        self.rgb_attn = nn.ModuleList([
            PatchChannelAttentionModule(self.dims[i], (patch_sizes[i],)*2) for i in range(4)
        ])
        self.dep_attn = nn.ModuleList([
            PatchSpatialAttentionModule_res(self.dims[i], (patch_sizes[i],)*2) for i in range(4)
        ])

        # 5) 1×1 분배 Conv (concat → ar/ad)
        self.conv_r   = nn.ModuleList([nn.Conv2d(2*self.dims[i], self.dims[i], 1) for i in range(4)])
        self.conv_d   = nn.ModuleList([nn.Conv2d(2*self.dims[i], self.dims[i], 1) for i in range(4)])

        # 6) 픽셀 게이팅 (두 분기 ar/ad 입력)
        self.px_gate  = nn.ModuleList([PixelGatingModule(2*self.dims[i]) for i in range(4)])

        # 7) 블렌딩 파라미터(선택)
        if self.blend == 'learnable':
            # stage·branch별 α (스칼라). σ(0)=0.5 → 초기엔 avg와 동일
            self.alpha_r = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(4)])
            self.alpha_d = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(4)])
        elif self.blend == 'mulcoeff':
            # cat(2C) → C 채널 계수, β는 범위 조절 계수(σ로 안정화)
            self.coeff_r = nn.ModuleList([nn.Conv2d(2*self.dims[i], self.dims[i], 1) for i in range(4)])
            self.coeff_d = nn.ModuleList([nn.Conv2d(2*self.dims[i], self.dims[i], 1) for i in range(4)])
            self.beta_r  = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(4)])
            self.beta_d  = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(4)])

        # 8) HF 버전별 인코더 포인터
        self.r_enc = getattr(self.rgb, "encoder", None) or getattr(self.rgb, "mit", None)
        self.d_enc = getattr(self.dep, "encoder", None) or getattr(self.dep, "mit", None)
        if self.r_enc is None or self.d_enc is None:
            raise AttributeError("SegFormer encoder not found (tried 'encoder' and 'mit').")

    # 선택: 일부 브랜치 freeze
    def freeze_rgb(self):
        for p in self.rgb.parameters(): p.requires_grad = False
    def freeze_depth(self):
        for p in self.dep.parameters(): p.requires_grad = False

    # 블록 호출 (※ no-grad 금지: 역전파 위해 grad 유지)
    def _blk_forward(self, blk, x, H, W):
        try:
            return blk(x, H, W)
        except TypeError:
            return blk(x, (H, W))

    def _run_stage(self, enc, x, i):
        # patch embed
        x, H, W = enc.patch_embeddings[i](x)  # x: (B, N, C)
        # transformer blocks
        for blk in enc.block[i]:
            x = self._blk_forward(blk, x, H, W)
            if isinstance(x, tuple):  # 일부 변형 대응
                x = x[0]
        # LN → (B,N,C) → (B,C,H,W)
        x = enc.layer_norm[i](x)
        B, N, C = x.shape
        return x.transpose(1, 2).reshape(B, C, H, W)

    def _blend(self, xr, xd, cat, or_new, od_new, i):
        if self.blend == 'avg':
            xr = (xr + or_new) / 2
            xd = (xd + od_new) / 2
        elif self.blend == 'learnable':
            ar = torch.sigmoid(self.alpha_r[i])
            ad = torch.sigmoid(self.alpha_d[i])
            xr = ar * xr + (1 - ar) * or_new
            xd = ad * xd + (1 - ad) * od_new
        elif self.blend == 'mulcoeff':
            # 채널별 계수: scale = 1 + σ(β)·σ(W(cat))
            scale_r = 1 + torch.sigmoid(self.beta_r[i]) * torch.sigmoid(self.coeff_r[i](cat))
            scale_d = 1 + torch.sigmoid(self.beta_d[i]) * torch.sigmoid(self.coeff_d[i](cat))
            xr = xr * scale_r
            xd = xd * scale_d
        return xr, xd

    def forward(self, x_rgb, x_depth):
        # Depth 1ch → RGB ch
        x_d = self.depth_proj(x_depth)

        fused_feats = []
        xr_in, xd_in = x_rgb, x_d

        # 4-stage encoder with EncUpdate
        for i in range(4):
            xr = self._run_stage(self.r_enc, xr_in, i)
            xd = self._run_stage(self.d_enc, xd_in, i)

            if i in self.fuse:
                # (1) Base attention
                or_out, *_ = self.rgb_attn[i](xr)   # RGB: 채널 중심
                od_out, *_ = self.dep_attn[i](xd)   # Depth: 공간 중심(C채널로 확장)

                # (2) concat → 1×1 분배 conv
                cat = torch.cat([or_out, od_out], dim=1)  # (B, 2C, H, W)
                ar  = self.conv_r[i](cat)                 # (B, C, H, W)
                ad  = self.conv_d[i](cat)                 # (B, C, H, W)

                # (3) 픽셀 게이팅 (교차강조)
                or_new, od_new, _ = self.px_gate[i](ar, ad)

                # (4) 블렌딩 (모드별)
                xr, xd = self._blend(xr, xd, cat, or_new, od_new, i)

            # (5) 디코더 입력 피처
            fused_feats.append(xr + xd)

            # (6) ★ EncUpdate: 다음 스테이지 입력에 갱신본 전달
            xr_in, xd_in = xr, xd

        # (7) SegFormer decode head
        return self.decode_head(fused_feats)
