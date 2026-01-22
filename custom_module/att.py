import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchChannelAttentionModule(nn.Module):
    def __init__(self, channels, patch_size=(7,7)):
        """
        channels: 입력 채널 수 C
        patch_size: (패치 높이, 패치 너비)
        """
        super().__init__()
        self.C = channels
        self.ph, self.pw = patch_size
        
        # attention 강화 계수
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        #----------------------------------------------------------------------
        # 1) 패치 단위로 Unfold => (B, C*P, M)  // P=ph*pw, M=패치 개수
        #----------------------------------------------------------------------
        patches = F.unfold(x, kernel_size=(self.ph, self.pw),
                                stride=(self.ph, self.pw))  # (B, C*P, M)
        P = self.ph * self.pw
        M = patches.size(-1)
        
        # (B, C, P, M) -> (B, M, C, P)
        patches = patches.view(B, C, P, M).permute(0, 3, 1, 2)
        # 이제 patches.shape == (B, M, C, P)
        
        #----------------------------------------------------------------------
        # 2) 패치별 채널 유사도 Sc_patch 계산
        #    Sc_patch.shape = (B, M, C, C)
        #----------------------------------------------------------------------
        FM = patches  # alias: (B, M, C, P)
        FM_big = FM.reshape(B*M, C, P)            # (B*M, C, P)
        Sc_big = torch.bmm(FM_big, FM_big.transpose(1,2))  # (B*M, C, C)
        Sc_big = F.softmax(Sc_big, dim=-1)
        Sc_patch = Sc_big.view(B, M, C, C)
        
        #----------------------------------------------------------------------
        # 3) 패치별 집계 feature Mc_patch
        #    Mc_patch.shape = (B, M, C, P)
        #----------------------------------------------------------------------
        Mc_big = torch.bmm(Sc_big, FM_big)        # (B*M, C, P)
        Mc_patch = Mc_big.view(B, M, C, P)
        
        #----------------------------------------------------------------------
        # 4) 패치별 공분산 cov_patch 계산
        #    cov_patch.shape = (B, M, C, C)
        #----------------------------------------------------------------------
        A_big = FM_big                             # (B*M, C, P)
        M_big = Mc_big                             # (B*M, C, P)
        mean_A = A_big.mean(dim=-1, keepdim=True)  # (B*M, C, 1)
        mean_M = M_big.mean(dim=-1, keepdim=True)  # (B*M, C, 1)
        cov_big = torch.bmm(
            A_big - mean_A,
            (M_big - mean_M).transpose(1,2)
        ) / P                                       # (B*M, C, C)
        cov_patch = cov_big.view(B, M, C, C)
        
        #----------------------------------------------------------------------
        # 5) 패치별 복합 채널 관계 Lc_patch = Sc_patch + cov_patch
        #    shape = (B, M, C, C)
        #----------------------------------------------------------------------
        Lc_patch = Sc_patch + cov_patch
        
        #----------------------------------------------------------------------
        # 6) 패치별 강화된 feature Ec_patch = Lc_patch · A_patch
        #    Ec_big.shape = (B*M, C, P)
        #    → Ec_patch.shape = (B, M, C, P)
        #----------------------------------------------------------------------
        Ec_big = torch.bmm(Lc_patch.view(B*M, C, C), A_big)  # (B*M, C, P)
        Ec_patch = Ec_big.view(B, M, C, P)
        
        #----------------------------------------------------------------------
        # 7) 패치들을 다시 Fold(역 Unfold) → (B, C, H, W)
        #    attention_map = β·fold(Ec_patch) + x
        #    out = x * attention_map
        #----------------------------------------------------------------------
        # (B, C*P, M)
        Ec_patches = Ec_patch.permute(0,2,3,1).reshape(B, C*P, M)
        Ec_map = F.fold(Ec_patches,
                        output_size=(H, W),
                        kernel_size=(self.ph, self.pw),
                        stride=(self.ph, self.pw))
        
        attention_map = self.beta * Ec_map + x
        out = x * attention_map
        
        return out, Sc_patch, cov_patch, Lc_patch, Ec_map
    
class PatchSpatialAttentionModule(nn.Module):
    def __init__(self, patch_size=(7,7)):
        """
        patch_size: (패치 높이, 패치 너비)
        """
        super().__init__()
        self.ph, self.pw = patch_size
        self.beta = nn.Parameter(torch.zeros(1))  # 어텐션 강화 계수

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns:
            out            : (B, C, H, W) — 패치 단위 spatial attention 적용된 출력
            Sc_patch       : (B, M, P, P) — 패치별 1차 유사도 행렬
            cov_patch      : (B, M, P, P) — 패치별 2차 공분산 행렬
            L_patch        : (B, M, P, P) — 패치별 복합 위치 관계 행렬
            Ec_map         : (B, C, H, W) — 패치단위 강화된 spatial map
        """
        B, C, H, W = x.shape
        # P = patch 내 위치 수, M = 패치 개수
        P = self.ph * self.pw
        # 1) 패치 단위로 unfold → (B, C*P, M)
        patches = F.unfold(x, kernel_size=(self.ph, self.pw),
                             stride=(self.ph, self.pw))  # (B, C*P, M)
        M = patches.size(-1)

        # 2) (B, C*P, M) → (B, M, C, P) → (B*M, P, C)
        patches = patches.view(B, C, P, M).permute(0, 3, 2, 1)  # (B, M, P, C)
        patches_flat = patches.reshape(B*M, P, C)

        # 3) 패치별 1차 유사도 Sc_patch: (B*M, P, P) → (B, M, P, P)
        Sc = torch.bmm(patches_flat, patches_flat.transpose(1,2))
        Sc = F.softmax(Sc, dim=-1)
        Sc_patch = Sc.view(B, M, P, P)

        # 4) 패치별 집계 feature Mc_flat: (B*M, P, C)
        Mc_flat = torch.bmm(Sc, patches_flat)  # 각 위치별 채널 feature 집계

        # 5) 패치별 공분산 cov_patch: (B*M, P, P) → (B, M, P, P)
        mean_A = patches_flat.mean(dim=1, keepdim=True)  # (B*M, 1, C)
        mean_M = Mc_flat.mean(dim=1, keepdim=True)       # (B*M, 1, C)
        cov = torch.bmm(
            (patches_flat - mean_A).transpose(1,2),      # (B*M, C, P)
            (Mc_flat - mean_M)                           # (B*M, P, C)
        ) / P                                            # (B*M, C, C)? → 위치끼리 공분산
        # 사실 spatial 공분산은 위치×위치(P×P)이므로
        # patches_flat: (P,C) → cov_dim=(P,P) 로 계산하려면 transpose 순서 변경:
        cov = torch.bmm(
            (patches_flat - mean_A),                     # (B*M, P, C)
            (Mc_flat - mean_M).transpose(1,2)            # (B*M, C, P)
        ) / P                                            # (B*M, P, P)
        cov_patch = cov.view(B, M, P, P)

        # 6) 패치별 복합 위치 관계 L = Sc + cov
        L_patch = Sc_patch + cov_patch  # (B, M, P, P)

        # 7) patch별 강화된 위치 표현 Ec_flat = L · 원본 위치 벡터
        # Ec_flat = torch.bmm(
        #     L_patch.view(B*M, P, P),
        #     patches_flat.transpose(1,2)                   # (B*M, C, P)
        # )  # 결과: (B*M, P, P)×(B*M, C, P)? → 사실 (B*M, P, C)
        Ec_flat = torch.bmm(
            L_patch.view(B*M, P, P),
            patches_flat
        )  # (B*M, P, C)
        # 다시 (B, M, P, C) → (B, C*P, M) 형태로
        Ec_patches = Ec_flat.view(B, M, P, C).permute(0, 3, 2, 1).reshape(B, C*P, M)

        # 8) fold → Ec_map (B, C, H, W)
        Ec_map = F.fold(Ec_patches,
                        output_size=(H, W),
                        kernel_size=(self.ph, self.pw),
                        stride=(self.ph, self.pw))

        # 9) 최종 spatial attention
        attention_map = self.beta * Ec_map + x
        out = x * attention_map

        return out, Sc_patch, cov_patch, L_patch, Ec_map


