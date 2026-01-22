import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockDiagonalChannelAttention(nn.Module):
    def __init__(self, channels, patch_size=(7,7), alpha=0.5):
        """
        블록 대각 근사 joint covariance를 사용하는 채널 어텐션
        channels: 입력 채널 수 C
        patch_size: (패치 높이, 패치 너비)
        alpha: 채널/공간 블록 가중치
        """
        super().__init__()
        self.C     = channels
        self.ph, self.pw = patch_size
        self.P     = self.ph * self.pw
        self.alpha = alpha
        self.beta  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        # 1) unfold → patches (B, M, C, P)
        patches = F.unfold(x, kernel_size=(self.ph, self.pw),
                              stride=(self.ph, self.pw))  # (B, C*P, M)
        M = patches.size(-1)
        patches = patches.view(B, C, self.P, M).permute(0,3,1,2)

        # 2) 채널 유사도 Sc_big (B*M, C, C)
        flat   = patches.reshape(B*M, C, self.P)
        Sc_big = torch.bmm(flat, flat.transpose(1,2))
        Sc_big = F.softmax(Sc_big, dim=-1)

        # 3) 채널 집계 Mc_big (B*M, C, P)
        Mc_big = torch.bmm(Sc_big, flat)

        # 4) 채널 공분산 cov_C (B*M, C, C)
        mean_A = flat.mean(dim=-1, keepdim=True)
        cov_C  = torch.bmm(flat - mean_A,
                          (flat - mean_A).transpose(1,2)) / self.P

        # 5) 공간 공분산 cov_S (B*M, P, P)
        #    패치 내 위치 간 공분산을 이용
        loc_flat = flat.transpose(1,2)  # (B*M, P, C)
        mean_L   = loc_flat.mean(dim=-1, keepdim=True)
        cov_S    = torch.bmm(loc_flat - mean_L,
                            (loc_flat - mean_L).transpose(1,2)) / self.C

        # 6) 블록 대각 joint cov (B*M, C+P, C+P)
        D = C + self.P
        cov_joint = torch.zeros(B*M, D, D,
                                device=x.device, dtype=x.dtype)
        # 좌상: 채널, 우하: 공간
        cov_joint[:, :C, :C] =  self.alpha   * cov_C
        cov_joint[:, C:, C:] = (1-self.alpha) * cov_S

        # 7) block‑diag 형태로 Sc_big padding (B*M, D, D)
        Sc_block = F.pad(Sc_big,
                         (0, self.P, 0, self.P),
                         value=0.0)

        # 8) joint relation L_big = Sc_block + cov_joint
        L_big = Sc_block + cov_joint               # (B*M, D, D)

        # 9) 강화된 feature Ec_big = L_big[:, :C, :C]·flat  (B*M, C, P)
        #    (block‑diag joint cov 중 채널 부분만 사용)
        Ec_big = torch.bmm(L_big[:, :C, :C], flat)

        # 10) fold back to (B, C, H, W)
        Ec_patch = Ec_big.view(B, M, C, self.P)    # (B, M, C, P)
        Ec_flat  = Ec_patch.permute(0,2,3,1).reshape(B, C*self.P, M)
        Ec_map   = F.fold(Ec_flat,
                         output_size=(H, W),
                         kernel_size=(self.ph, self.pw),
                         stride=(self.ph, self.pw))

        # 11) 최종 output
        attention_map = self.beta * Ec_map + x
        out = x * attention_map
        return out, Sc_big.view(B, M, C, C), cov_C.view(B, M, C, C), None, Ec_map


class BlockDiagonalSpatialAttention(nn.Module):
    def __init__(self, patch_size=(7,7), alpha=0.5):
        """
        블록 대각 근사 joint covariance를 사용하는 공간 어텐션
        patch_size: (패치 높이, 패치 너비)
        alpha: 채널/공간 블록 가중치
        """
        super().__init__()
        self.ph, self.pw = patch_size
        self.P           = self.ph * self.pw
        self.alpha       = alpha
        self.beta        = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        # 1) unfold → patches (B, M, P, C)
        patches = F.unfold(x, kernel_size=(self.ph, self.pw),
                             stride=(self.ph, self.pw))  # (B, C*P, M)
        M = patches.size(-1)
        patches = patches.view(B, C, self.P, M).permute(0,3,2,1)
        flat    = patches.reshape(B*M, self.P, C)

        # 2) 위치 유사도 Sc_big (B*M, P, P)
        Sc_big = torch.bmm(flat, flat.transpose(1,2))
        Sc_big = F.softmax(Sc_big, dim=-1)

        # 3) 위치 집계 Mc_big (B*M, P, C)
        Mc_big = torch.bmm(Sc_big, flat)

        # 4) 위치 공분산 cov_S (B*M, P, P)
        mean_A = flat.mean(dim=-1, keepdim=True)
        cov_S  = torch.bmm(flat-mean_A,
                          (flat-mean_A).transpose(1,2)) / C

        # 5) 채널 공분산 cov_C (B*M, C, C)
        loc_flat = flat.transpose(1,2)  # (B*M, C, P)
        mean_L   = loc_flat.mean(dim=-1, keepdim=True)
        cov_C    = torch.bmm(loc_flat-mean_L,
                            (loc_flat-mean_L).transpose(1,2)) / self.P

        # 6) 블록 대각 joint cov (B*M, P+C, P+C)
        D = self.P + C
        cov_joint = torch.zeros(B*M, D, D,
                                device=x.device, dtype=x.dtype)
        cov_joint[:, :self.P, :self.P] = self.alpha   * cov_S
        cov_joint[:, self.P:, self.P:] = (1-self.alpha)* cov_C

        # 7) Sc padding → Sc_block (B*M, D, D)
        Sc_block = F.pad(Sc_big,
                         (0, C, 0, C),
                         value=0.0)

        # 8) L_big = Sc_block + cov_joint (B*M, D, D)
        L_big = Sc_block + cov_joint

        # 9) Ec_big = L_big[:P,:P]·flat  (B*M, P, C)
        Ec_big = torch.bmm(L_big[:, :self.P, :self.P], flat)

        # 10) fold back to (B, C, H, W)
        Ec_patch = Ec_big.view(B, M, self.P, C).permute(0,3,2,1)
        Ec_flat  = Ec_patch.reshape(B, C*self.P, M)
        Ec_map   = F.fold(Ec_flat,
                         output_size=(H, W),
                         kernel_size=(self.ph, self.pw),
                         stride=(self.ph, self.pw))

        # 11) 최종 output
        attention_map = self.beta * Ec_map + x
        out = x * attention_map
        return out, Sc_big.view(B, M, self.P, self.P), cov_S.view(B, M, self.P, self.P), None, Ec_map
