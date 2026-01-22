import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelGatingModule(nn.Module):
    """
    두 브랜치(o_r, o_d)의 feature map을 concat한 뒤,
    1×1 Conv + Sigmoid 게이트를 만들어 픽셀별 가중합을 수행.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # in_channels = o_r 채널 수 + o_d 채널 수
        self.gate_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        # bias가 있는 게 좋습니다 (threshold 조정용)
        self.sigmoid = nn.Sigmoid()

    def forward(self, o_r: torch.Tensor, o_d: torch.Tensor):
        # o_r, o_d: [B, C, H, W]
        cat = torch.cat([o_r, o_d], dim=1)         # [B, 2C, H, W]
        g = self.sigmoid(self.gate_conv(cat))     # [B, 1, H, W], 값 ∈ (0,1)
        # 픽셀 단위로 RGB vs Depth 강조
        o_r_new = g * o_r + (1 - g) * o_d
        o_d_new = g * o_d + (1 - g) * o_r
        return o_r_new, o_d_new, g