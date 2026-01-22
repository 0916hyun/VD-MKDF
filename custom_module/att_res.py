import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from custom_module.att import (PatchChannelAttentionModule,
                 PatchSpatialAttentionModule)
from custom_module.block_diag_attention import BlockDiagonalSpatialAttention
# cp
from custom_module.cp_attention import CPSpatialAttention

class PatchSpatialAttentionModule_res(nn.Module):
    """
    Depth 브랜치 Spatial Attention을 C 채널로 확장하여 직접 학습하도록 수정한 모듈
    """
    def __init__(self, channels: int, patch_size=(7,7)):
        super().__init__()
        self.base = PatchSpatialAttentionModule(patch_size)
        self.expand = nn.Conv2d(1, channels, kernel_size=1, bias=True)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        # base forward: (out, Sc_patch, cov_patch, L_patch, Ec_map)
        out, Sc_patch, cov_patch, L_patch, Ec_map = self.base(x)
        # 채널 수에 따라 확장
        if Ec_map.shape[1] == 1:
            Ec_res = self.expand(Ec_map)
        else:
            Ec_res = Ec_map
        # residual attention 적용
        attention_map = self.beta * Ec_res + x
        out = x * attention_map
        return out, Sc_patch, cov_patch, L_patch, Ec_res

class BlockDiagonalSpatialAttention_res(nn.Module):
    """
    Depth 브랜치 Block-Diagonal Spatial Attention을 C 채널로 확장한 모듈
    """
    def __init__(self, channels: int, patch_size: int, alpha: float = 0.5):
        super().__init__()
        self.base = BlockDiagonalSpatialAttention(patch_size, alpha)
        self.expand = nn.Conv2d(1, channels, kernel_size=1, bias=True)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        out, Sc_patch, cov_patch, L_patch, Ec_map = self.base(x)
        if Ec_map.shape[1] == 1:
            Ec_res = self.expand(Ec_map)
        else:
            Ec_res = Ec_map
        attention_map = self.beta * Ec_res + x
        out = x * attention_map
        return out, Sc_patch, cov_patch, L_patch, Ec_res

class CPSpatialAttention_res(nn.Module):
    """
    Depth 브랜치 CP Spatial Attention을 C 채널로 확장한 모듈
    """
    def __init__(self, channels: int, rank: int = 8):
        super().__init__()
        self.base = CPSpatialAttention(rank=rank)
        # CP 모듈은 이미 C채널 반환
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        out, Sc_patch, cov_patch, L_patch, Ec_map = self.base(x)
        Ec_res = Ec_map  # 채널 그대로 사용
        attention_map = self.beta * Ec_res + x
        out = x * attention_map
        return out, Sc_patch, cov_patch, L_patch, Ec_res
