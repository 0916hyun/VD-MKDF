import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchChannelAttentionModule_g(nn.Module):
    def __init__(self, channels, patch_size=(7,7)):
        super().__init__()
        self.C = channels
        self.ph, self.pw = patch_size
        self.beta = nn.Parameter(torch.zeros(1))
        # 변경: Cross-Gating을 위한 게이트 Conv 추가
        self.gate_conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)

    # 변경: external_L 인자 추가
    def forward(self, x, external_L=None):
        B, C, H, W = x.shape
        P = self.ph * self.pw
        # 1) Unfold → patches
        patches = F.unfold(x, kernel_size=(self.ph, self.pw), stride=(self.ph, self.pw))  # (B, C*P, M)
        M = patches.size(-1)
        patches = patches.view(B, C, P, M).permute(0,3,1,2)  # (B, M, C, P)

        # 2) Sc_patch 계산
        FM_big = patches.reshape(B*M, C, P)
        Sc_big = F.softmax(torch.bmm(FM_big, FM_big.transpose(1,2)), dim=-1)
        Sc_patch = Sc_big.view(B, M, C, C)

        # 3) Mc_patch 계산
        Mc_big = torch.bmm(Sc_big, FM_big)

        # 4) cov_patch 계산
        mean_A = FM_big.mean(dim=-1, keepdim=True)
        mean_M = Mc_big.mean(dim=-1, keepdim=True)
        cov_big = torch.bmm(FM_big-mean_A, (Mc_big-mean_M).transpose(1,2)) / P
        cov_patch = cov_big.view(B, M, C, C)

        # 5) Lc_patch 계산
        Lc_patch = Sc_patch + cov_patch
        # 변경: Cross-Gating 적용
        if external_L is not None:
            # external_L: (B, M, P, P)
            gate_input = external_L.mean(dim=(2,3), keepdim=True)  # (B, M, 1, 1)
            # Conv expects (B, in_C, H, W) so permute
            g = torch.sigmoid(self.gate_conv(gate_input.permute(0,2,1,3)))
            g = g.permute(0,2,1,3)  # (B, M, 1,1)
            Lc_patch = Lc_patch * g

        # 6) Ec_big = Lc·A_big
        Ec_big = torch.bmm(Lc_patch.view(B*M, C, C), FM_big)
        Ec_patch = Ec_big.view(B, M, C, P)

        # 7) Fold back → Ec_map
        Ec_patches = Ec_patch.permute(0,2,3,1).reshape(B, C*P, M)
        Ec_map = F.fold(Ec_patches, output_size=(H, W), kernel_size=(self.ph, self.pw), stride=(self.ph, self.pw))

        # 8) out 계산
        attention_map = self.beta * Ec_map + x
        out = x * attention_map
        return out, Sc_patch, cov_patch, Lc_patch, Ec_map
    
class PatchSpatialAttentionModule_g(nn.Module):
    def __init__(self, patch_size=(7,7)):
        super().__init__()
        self.ph, self.pw = patch_size
        self.P = self.ph * self.pw
        self.beta = nn.Parameter(torch.zeros(1))
        # 변경: Cross-Gating 게이트 Conv 추가
        self.gate_conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)

    # 변경: external_L 인자 추가
    def forward(self, x, external_L=None):
        B, C, H, W = x.shape
        P = self.P
        # 1) Unfold → patches_flat
        patches = F.unfold(x, kernel_size=(self.ph, self.pw), stride=(self.ph, self.pw))
        M = patches.size(-1)
        patches_flat = patches.view(B, C, P, M).permute(0,3,2,1).reshape(B*M, P, C)

        # 2) Sc_patch
        Sc = F.softmax(torch.bmm(patches_flat, patches_flat.transpose(1,2)), dim=-1)
        Sc_patch = Sc.view(B, M, P, P)

        # 3) cov_patch
        Mc_flat = torch.bmm(Sc, patches_flat)
        mean_A = patches_flat.mean(dim=-1, keepdim=True)
        mean_M = Mc_flat.mean(dim=-1, keepdim=True)
        cov = torch.bmm(patches_flat-mean_A, (Mc_flat-mean_M).transpose(1,2)) / C
        cov_patch = cov.view(B, M, P, P)

        # 4) Ls_patch = Sc + cov
        Ls_patch = Sc_patch + cov_patch
        # 변경: Cross-Gating 적용
        if external_L is not None:
            gate_input = external_L.mean(dim=(2,3), keepdim=True)
            g = torch.sigmoid(self.gate_conv(gate_input.permute(0,2,1,3)))
            g = g.permute(0,2,1,3)
            Ls_patch = Ls_patch * g

        # 5) Ec_big = Ls·patches_flat
        Ec_big = torch.bmm(Ls_patch.view(B*M, P, P), patches_flat)
        Ec_patch = Ec_big.view(B, M, P, C)

        # 6) Fold back → Ec_map
        Ec_patches = Ec_patch.permute(0,3,2,1).reshape(B, C*P, M)
        Ec_map = F.fold(Ec_patches, output_size=(H, W), kernel_size=(self.ph, self.pw), stride=(self.ph, self.pw))

        # 7) out
        attention_map = self.beta * Ec_map + x
        out = x * attention_map
        return out, Sc_patch, cov_patch, Ls_patch, Ec_map