# import copy
# import torch
# import torch.nn as nn
# from transformers import SegformerForSemanticSegmentation
# from custom_module.att import PatchChannelAttentionModule
# from custom_module.att_res import PatchSpatialAttentionModule_res
# from custom_module.pixel_gating import PixelGatingModule

# class Segb5_RGBD_FuseStageSelective(nn.Module):
#     """
#     SegFormer B5 기반 Dual-Encoder RGB-D 세그멘테이션 모델에
#     Base Attention + Pixel Gating Fusion을 적용합니다.

#     Args:
#         n_classes (int): 예측할 클래스 수.
#         patch_sizes (list[int]): 각 Transformer 블록 출력에 대응하는 패치 크기 리스트 (length=4).
#         fuse_stages (list[int]): Fusion/Attention을 적용할 인코더 블록 인덱스 (1~4).
#     """
#     def __init__(
#         self,
#         n_classes: int,
#         patch_sizes: list[int],
#         fuse_stages: list[int],
#     ):
#         super().__init__()

#         # 1) SegFormer B5 백본 로드 (output_hidden_states=True)
#         hf = SegformerForSemanticSegmentation.from_pretrained(
#             "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
#             num_labels=n_classes,
#             ignore_mismatched_sizes=True,
#             output_hidden_states=True
#         )
#         self.encoder_rgb   = hf.segformer
#         self.encoder_depth = copy.deepcopy(hf.segformer)
#         self.decode_head   = hf.decode_head

#         # 1×1 프로젝션: depth 1채널 → rgb 채널 수 (embed_dim)
#         embed_dim = self.encoder_rgb.config.num_channels  # 보통 3
#         self.depth_proj = nn.Conv2d(1, embed_dim, kernel_size=1)

#         # 2) 블록별 채널 크기 (hidden_states[1]~[4])
#         dims = self.encoder_rgb.config.hidden_sizes  # e.g. [64, 128, 320, 512]
#         assert len(patch_sizes) == len(dims), (
#             f"patch_sizes 길이({len(patch_sizes)})는 블록 수({len(dims)})와 같아야 합니다."
#         )

#         self.patch_sizes = patch_sizes
#         self.fuse_stages = set(fuse_stages)

#         # 3) Base Attention, 1×1 Fusion Conv, PixelGating 모듈 정의
#         self.rgb_attns   = nn.ModuleList([
#             PatchChannelAttentionModule(dims[i], patch_size=(patch_sizes[i], patch_sizes[i]))
#             for i in range(len(dims))
#         ])
#         self.depth_attns = nn.ModuleList([
#             PatchSpatialAttentionModule_res(dims[i], patch_size=(patch_sizes[i], patch_sizes[i]))
#             for i in range(len(dims))
#         ])
#         self.fuse_conv_r = nn.ModuleList([
#             nn.Conv2d(2 * dims[i], dims[i], kernel_size=1)
#             for i in range(len(dims))
#         ])
#         self.fuse_conv_d = nn.ModuleList([
#             nn.Conv2d(2 * dims[i], dims[i], kernel_size=1)
#             for i in range(len(dims))
#         ])
#         self.pixel_gates = nn.ModuleList([
#             PixelGatingModule(dims[i])
#             for i in range(len(dims))
#         ])

#     def forward(self, x_rgb: torch.Tensor, x_depth: torch.Tensor) -> torch.Tensor:
#         # 1) Depth 1ch → rgb 채널 수로 프로젝션
#         x_d_proj = self.depth_proj(x_depth)

#         # 2) Dual-Encoder 순전파 (hidden_states 추출)
#         out_r = self.encoder_rgb(x_rgb)
#         out_d = self.encoder_depth(x_d_proj)
#         hs_r  = out_r.hidden_states  # tuple length=5: [embed, block1, block2, block3, block4]
#         hs_d  = out_d.hidden_states

#         fused_feats = []
#         # 3) Stage 0 (Patch Embedding) 합산
#         fused_feats.append(hs_r[0] + hs_d[0])  # 채널 768

#         # 4) 스테이지 1~4 Fusion 적용
#         for idx, stage in enumerate(range(1, len(hs_r))):
#             x_r = hs_r[stage]
#             x_d = hs_d[stage]

#             if stage in self.fuse_stages:
#                 # 4-1) Base Attention
#                 o_r, *_, _ = self.rgb_attns[idx](x_r)
#                 o_d, *_, _ = self.depth_attns[idx](x_d)
#                 # 4-2) Concat → 1×1 Conv 분리
#                 cat    = torch.cat([o_r, o_d], dim=1)
#                 att_r  = self.fuse_conv_r[idx](cat)
#                 att_d  = self.fuse_conv_d[idx](cat)
#                 # 4-3) Pixel Gating + Residual 평균
#                 o_r_new, o_d_new, _ = self.pixel_gates[idx](att_r, att_d)
#                 x_r = (x_r + o_r_new) / 2
#                 x_d = (x_d + o_d_new) / 2

#             # 4-4) RGB+Depth 합산 → 디코더 입력
#             fused_feats.append(x_r + x_d)

#         # 5) Decode Head로 최종 Segmentation 예측
#         seg_logits = self.decode_head(fused_feats)  # expects list of 5 feature maps
#         return seg_logits
import copy
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from custom_module.att import PatchChannelAttentionModule
from custom_module.att_res import PatchSpatialAttentionModule_res
from custom_module.pixel_gating import PixelGatingModule

class Segb5_RGBD_FuseStageSelective(nn.Module):
    """
    SegFormer B5 기반 Dual-Encoder RGB-D 세그멘테이션 모델에
    Base Attention + Pixel Gating Fusion을 적용합니다.

    Args:
        n_classes (int): 예측할 클래스 수.
        patch_sizes (list[int]): 각 Transformer 블록 출력 채널에 대응하는 패치 크기 리스트 (length=4).
        fuse_stages (list[int]): Fusion/Attention을 적용할 블록 인덱스 (0~3에 매핑).
    """
    def __init__(self, n_classes: int, patch_sizes: list[int], fuse_stages: list[int]):
        super().__init__()

        # SegFormer B5 백본 로드 (output_hidden_states=True)
        hf = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
            num_labels=n_classes,
            ignore_mismatched_sizes=True,
            output_hidden_states=True
        )
        self.encoder_rgb   = hf.segformer
        self.encoder_depth = copy.deepcopy(hf.segformer)
        self.decode_head   = hf.decode_head

        # Depth 1ch → RGB 입력 채널 수로 프로젝션
        in_ch = self.encoder_rgb.config.num_channels
        self.depth_proj = nn.Conv2d(1, in_ch, kernel_size=1)

        # Transformer 블록 출력 채널 [blk1..blk4]
        self.dims = self.encoder_rgb.config.hidden_sizes  # [64, 128, 320, 512]
        assert len(patch_sizes) == len(self.dims), (
            f"patch_sizes 길이 {len(patch_sizes)} != 블록 수 {len(self.dims)}"
        )

        self.patch_sizes = patch_sizes
        self.fuse_stages = set(fuse_stages)
        self.fuse_stages_list = list(fuse_stages)

        # Base Attention, Fusion Conv, Pixel Gating 모듈 초기화
        self.rgb_attns = nn.ModuleList([
            PatchChannelAttentionModule(self.dims[i], patch_size=(patch_sizes[i], patch_sizes[i]))
            for i in range(len(self.dims))
        ])
        self.depth_attns = nn.ModuleList([
            PatchSpatialAttentionModule_res(self.dims[i], patch_size=(patch_sizes[i], patch_sizes[i]))
            for i in range(len(self.dims))
        ])
        self.fuse_conv_r = nn.ModuleList([
            nn.Conv2d(2 * self.dims[i], self.dims[i], kernel_size=1)
            for i in range(len(self.dims))
        ])
        self.fuse_conv_d = nn.ModuleList([
            nn.Conv2d(2 * self.dims[i], self.dims[i], kernel_size=1)
            for i in range(len(self.dims))
        ])
        self.pixel_gates = nn.ModuleList([
            PixelGatingModule(2 * self.dims[i])
            for i in range(len(self.dims))
        ])

    def forward(self, x_rgb: torch.Tensor, x_depth: torch.Tensor) -> torch.Tensor:


        # Depth 브랜치 프로젝션 (1ch → RGB ch)
        x_d = self.depth_proj(x_depth)

        # Dual-Encoder 순전파
        out_r = self.encoder_rgb(x_rgb)
        out_d = self.encoder_depth(x_d)
        hs_r  = out_r.hidden_states  # tuple length=4: [blk1, blk2, blk3, blk4]
        hs_d  = out_d.hidden_states

        fused_feats = []
        # 블록 단위 처리
        for i, (xr, xd) in enumerate(zip(hs_r, hs_d)):

            if i in self.fuse_stages:
                # Attention 모듈 출력 (첫 번째 반환값만 사용)
                or_out, *_, _ = self.rgb_attns[i](xr)
                od_out, *_, _ = self.depth_attns[i](xd)

                # Concat 및 Fusion Conv
                cat = torch.cat([or_out, od_out], dim=1)

                ar = self.fuse_conv_r[i](cat)
                ad = self.fuse_conv_d[i](cat)
                or_new, od_new, _ = self.pixel_gates[i](ar, ad)
                xr = (xr + or_new) / 2
                xd = (xd + od_new) / 2

            fused_feats.append(xr + xd)

        # Decode Head 적용
        seg_logits = self.decode_head(fused_feats)
        return seg_logits
    




class Segb5_RGBD_FuseStageSelective_EncUpdate(nn.Module):
    def __init__(self, n_classes: int, patch_sizes: list[int], fuse_stages: list[int]):
        super().__init__()
        hf = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
            num_labels=n_classes, ignore_mismatched_sizes=True, return_dict=True,
        )
        self.rgb = hf.segformer
        self.dep = copy.deepcopy(hf.segformer)
        self.decode_head = hf.decode_head

        in_ch = self.rgb.config.num_channels  # 3
        self.depth_proj = nn.Conv2d(1, in_ch, kernel_size=1)

        self.dims = list(self.rgb.config.hidden_sizes)  # [64,128,320,512]
        assert len(patch_sizes) == 4
        self.patch_sizes = patch_sizes
        self.fuse = set(fuse_stages)

        self.rgb_attn = nn.ModuleList([PatchChannelAttentionModule(self.dims[i], (patch_sizes[i],)*2) for i in range(4)])
        self.dep_attn = nn.ModuleList([PatchSpatialAttentionModule_res(self.dims[i], (patch_sizes[i],)*2) for i in range(4)])
        self.conv_r   = nn.ModuleList([nn.Conv2d(2*self.dims[i], self.dims[i], 1) for i in range(4)])
        self.conv_d   = nn.ModuleList([nn.Conv2d(2*self.dims[i], self.dims[i], 1) for i in range(4)])
        self.px_gate  = nn.ModuleList([PixelGatingModule(2*self.dims[i]) for i in range(4)])

        # 편의 포인터 (버전에 따라 이름이 'encoder' 또는 'mit'일 수 있음)
        self.r_enc = getattr(self.rgb, "encoder", None)
        if self.r_enc is None:
            self.r_enc = getattr(self.rgb, "mit", None)
        if self.r_enc is None:
            raise AttributeError("SegFormer encoder not found on self.rgb (tried 'encoder' and 'mit').")

        self.d_enc = getattr(self.dep, "encoder", None)
        if self.d_enc is None:
            self.d_enc = getattr(self.dep, "mit", None)
        if self.d_enc is None:
            raise AttributeError("SegFormer encoder not found on self.dep (tried 'encoder' and 'mit').")

    def freeze_rgb(self):
        for p in self.rgb.parameters(): p.requires_grad = False
    def freeze_depth(self):
        for p in self.dep.parameters(): p.requires_grad = False

    def _run_stage(self, enc, x, i):
        x, H, W = enc.patch_embeddings[i](x)  # x: (B, N, C)

        for blk in enc.block[i]:
            # HF 버전별 시그니처 호환
            try:
                # v4.2x ~ v4.3x 계열: (hidden_states, height, width)
                x = blk(x, H, W)
            except TypeError:
                # 일부 버전: (hidden_states, (H, W))
                x = blk(x, (H, W))

            # 혹시라도 블록이 (Tensor, H, W)를 튜플로 되돌려주는 변형 대응
            if isinstance(x, tuple):
                # 관례적으로 첫 항이 hidden_states
                x = x[0]

        x = enc.layer_norm[i](x)              # (B, N, C)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        return x

    def forward(self, x_rgb, x_depth):
        x_d = self.depth_proj(x_depth)
        fused_feats = []
        xr_in, xd_in = x_rgb, x_d

        for i in range(4):
            xr = self._run_stage(self.r_enc, xr_in, i)
            xd = self._run_stage(self.d_enc, xd_in, i)

            if i in self.fuse:
                or_out, *_ = self.rgb_attn[i](xr)
                od_out, *_ = self.dep_attn[i](xd)
                cat = torch.cat([or_out, od_out], dim=1)
                ar  = self.conv_r[i](cat)
                ad  = self.conv_d[i](cat)
                or_new, od_new, _ = self.px_gate[i](ar, ad)
                xr = (xr + or_new) / 2
                xd = (xd + od_new) / 2

            fused_feats.append(xr + xd)
            xr_in, xd_in = xr, xd  # ★ 다음 스테이지 입력으로 갱신본 전달

        return self.decode_head(fused_feats)
    
