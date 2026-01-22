import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
from tensorly.decomposition import parafac


tl.set_backend('pytorch')

class CPChannelAttention(nn.Module):
    def __init__(self, channels, patch_size=(7,7), rank=8):
        super().__init__()
        self.C, (self.ph, self.pw), self.P = channels, patch_size, patch_size[0]*patch_size[1]
        self.rank = rank
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        # 1) unfold
        patches = F.unfold(x, (self.ph, self.pw), stride=(self.ph, self.pw))  # (B, C*P, M)
        M = patches.size(-1)
        FM = patches.view(B, C, self.P, M).permute(0,3,1,2).reshape(B*M, C, self.P)

        # 2) 채널 유사도
        Sc = F.softmax(torch.bmm(FM, FM.transpose(1,2)), dim=-1)  # (B*M, C, C)

        # 3) 공분산
        mean_FM = FM.mean(dim=-1, keepdim=True)
        cov_big = torch.bmm(FM-mean_FM, (FM-mean_FM).transpose(1,2)) / self.P  # (B*M, C, C)

        # === 4) 글로벌 CP-ALS 분해 & 재구성 ===
        cov_global = cov_big.mean(dim=0, keepdim=True)                   # (1, C, C)
        weights, factors = parafac(cov_global, rank=self.rank,
                                   n_iter_max=5, init='random', tol=1e-4)
        # factors: [F0 (1×R), F1 (C×R), F2 (C×R)]
        f0, f1, f2 = factors
        w = weights * f0.squeeze(0)                                     # (R,)
        # cov_cp_global[i,j] = Σ_r w[r] * f1[i,r] * f2[j,r]
        cov_cp_global = (f1 * w.unsqueeze(0)) @ f2.T                     # (C, C)
        cov_cp = cov_cp_global.unsqueeze(0).expand(B*M, C, C)            # (B*M, C, C)

        # 5) 결합 & 강화
        L = Sc + cov_cp
        Ec = torch.bmm(L, FM)                                            # (B*M, C, P)

        # 6) fold back
        Ec = Ec.view(B, M, C, self.P).permute(0,2,3,1).reshape(B, C*self.P, M)
        Ec_map = F.fold(Ec, (H, W), kernel_size=(self.ph, self.pw),
                        stride=(self.ph, self.pw))

        # 7) output
        att_map = self.beta * Ec_map + x
        out     = x * att_map
        return out, Sc.view(B, M, C, C), cov_big.view(B, M, C, C), None, Ec_map


class CPSpatialAttention(nn.Module):
    def __init__(self, patch_size=(7,7), rank=8):
        super().__init__()
        self.ph, self.pw = patch_size
        self.P           = self.ph * self.pw
        self.rank        = rank
        self.beta        = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        # 1) unfold
        patches = F.unfold(x, (self.ph, self.pw), stride=(self.ph, self.pw))  # (B, C*P, M)
        M = patches.size(-1)
        FM = patches.view(B, C, self.P, M).permute(0,3,2,1).reshape(B*M, self.P, C)

        # 2) 공간 유사도
        Sc = F.softmax(torch.bmm(FM, FM.transpose(1,2)), dim=-1)  # (B*M, P, P)

        # 3) 공분산
        mean_FM = FM.mean(dim=-1, keepdim=True)
        cov_big = torch.bmm(FM-mean_FM, (FM-mean_FM).transpose(1,2)) / C  # (B*M, P, P)

        # === 4) 글로벌 CP-ALS 분해 & 재구성 ===
        cov_global = cov_big.mean(dim=0, keepdim=True)                # (1, P, P)
        weights, factors = parafac(cov_global, rank=self.rank,
                                   n_iter_max=5, init='random', tol=1e-4)
        f0, f1, f2 = factors  # f1, f2: (P×R)
        w = weights * f0.squeeze(0)                                     # (R,)
        cov_cp_global = (f1 * w.unsqueeze(0)) @ f2.T                     # (P, P)
        cov_cp = cov_cp_global.unsqueeze(0).expand(B*M, self.P, self.P)  # (B*M, P, P)

        # 5) 결합 & 강화
        L  = Sc + cov_cp
        Ec = torch.bmm(L, FM)                                            # (B*M, P, C)

        # 6) fold back
        Ec = Ec.view(B, M, self.P, C).permute(0,3,2,1).reshape(B, C*self.P, M)
        Ec_map = F.fold(Ec, (H, W), kernel_size=(self.ph, self.pw),
                        stride=(self.ph, self.pw))

        # 7) output
        att_map = self.beta * Ec_map + x
        out     = x * att_map
        return out, Sc.view(B, M, self.P, self.P), cov_big.view(B, M, self.P, self.P), None, Ec_map