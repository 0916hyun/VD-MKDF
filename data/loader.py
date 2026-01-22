# import os
# import random
# from PIL import Image
# import torch
# import torchvision.transforms as T
# import torchvision.transforms.functional as TF
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import torchvision.transforms.functional as F
# import random
# from pathlib import Path
# import cv2
#
# WEATHER2ID = {'rain': 0, 'snow': 1, 'fog': 2, 'flare': 3}
#
# class TrainLoadDatasetMulti(Dataset):
#     def __init__(self, sample_list, crop_size=256):
#         self.samples = sample_list
#         self.crop_size = crop_size
#
#     def __len__(self):
#         return len(self.samples)          # 1400
#
#     def __getitem__(self, idx):
#         s = self.samples[idx]
#         inp_img = Image.open(s['inp']).convert("RGB")
#         tar_img = Image.open(s['tar']).convert("RGB")
#         seg_img = Image.open(s['seg']).convert("L")
#
#         i,j,h,w = T.RandomCrop.get_params(inp_img,
#                                           (self.crop_size, self.crop_size))
#         F_crop  = lambda img: F.crop(img, i,j,h,w)
#         inp_img, tar_img, seg_img = map(F_crop, (inp_img, tar_img, seg_img))
#         if random.random() > 0.5:
#             inp_img, tar_img, seg_img = map(F.hflip, (inp_img, tar_img, seg_img))
#
#         inp_img = F.to_tensor(inp_img)
#         tar_img = F.to_tensor(tar_img)
#         seg_img = torch.from_numpy(np.array(seg_img)).long()
#
#         # weather 라벨 (classifier supervision용)
#         weather_id = WEATHER2ID[s['weather']]
#
#         return inp_img, tar_img, seg_img, weather_id, s['weather']
#
# class ValLoadDatasetMulti(Dataset):
#     def __init__(self, sample_list, crop_size=256):
#         self.samples    = sample_list
#         self.crop_size  = crop_size            # 256 유지
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         s = self.samples[idx]
#
#         inp_img = Image.open(s['inp']).convert('RGB')
#         tar_img = Image.open(s['tar']).convert('RGB')
#         seg_img = Image.open(s['seg']).convert('L')
#
#         i, j, h, w = T.RandomCrop.get_params(
#                         inp_img,
#                         output_size=(self.crop_size, self.crop_size))
#         F_crop = lambda img: F.crop(img, i, j, h, w)
#         inp_img, tar_img, seg_img = map(F_crop, (inp_img, tar_img, seg_img))
#
#         inp_img = F.to_tensor(inp_img)
#         tar_img = F.to_tensor(tar_img)
#         seg_img = torch.from_numpy(np.array(seg_img)).long()
#
#         weather_id = WEATHER2ID[s['weather']]
#         filename   = Path(s['tar']).stem              # scene000
#
#         return inp_img, tar_img, seg_img, weather_id, filename
#
#
# class TestLoadDatasetMulti(Dataset):
#
#     def __init__(self, sample_list):
#         self.samples = sample_list
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         s = self.samples[idx]
#
#         inp_img = Image.open(s['inp']).convert('RGB')
#         tar_img = Image.open(s['tar']).convert('RGB')
#         seg_img = Image.open(s['seg']).convert('L')
#
#         inp_img = F.to_tensor(inp_img)
#         tar_img = F.to_tensor(tar_img)
#         seg_img = torch.from_numpy(np.array(seg_img)).long()
#
#         weather_id = WEATHER2ID[s['weather']]
#         filename   = Path(s['tar']).stem
#
#         return inp_img, tar_img, seg_img, weather_id, filename

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

WEATHER2ID = {'rain': 0, 'snow': 1, 'fog': 2, 'flare': 3}


def _normalize_crop_size(crop_size):
    """int 또는 (H, W) 모두 지원 → (th, tw)로 정규화"""
    if isinstance(crop_size, (tuple, list)):
        th, tw = int(crop_size[0]), int(crop_size[1])
    else:
        th = tw = int(crop_size)
    return th, tw


def _pad_if_needed(img: Image.Image, th: int, tw: int, fill=0):
    """PIL.Image 입력. (H, W) 기준 부족분을 오른쪽/아래로 패딩"""
    W, H = img.size  # PIL: (W, H)
    pad_h = max(0, th - H)
    pad_w = max(0, tw - W)
    if pad_h > 0 or pad_w > 0:
        img = TF.pad(img, padding=(0, 0, pad_w, pad_h), fill=fill)
    return img


class TrainLoadDatasetMulti(Dataset):
    def __init__(self, sample_list, crop_size=256, ignore_index=255):
        self.samples = sample_list
        self.crop_size = crop_size
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        inp_img = Image.open(s['inp']).convert("RGB")
        tar_img = Image.open(s['tar']).convert("RGB")
        seg_img = Image.open(s['seg']).convert("L")

        # 1) 크롭 사이즈 정규화 + 작은 이미지 패딩
        th, tw = _normalize_crop_size(self.crop_size)
        inp_img = _pad_if_needed(inp_img, th, tw, fill=0)
        tar_img = _pad_if_needed(tar_img, th, tw, fill=0)
        seg_img = _pad_if_needed(seg_img, th, tw, fill=self.ignore_index)

        # 2) 랜덤 크롭 파라미터 샘플링 (PIL 기준)
        i, j, h, w = T.RandomCrop.get_params(inp_img, (th, tw))

        # 3) 동일 파라미터로 모든 항목 크롭
        crop = lambda im: TF.crop(im, i, j, h, w)
        inp_img, tar_img, seg_img = map(crop, (inp_img, tar_img, seg_img))

        # 4) 좌우 플립(동일하게)
        if random.random() > 0.5:
            inp_img, tar_img, seg_img = map(TF.hflip, (inp_img, tar_img, seg_img))

        # 5) Tensor 변환
        inp_img = TF.to_tensor(inp_img)                       # [3,H,W], float32 in [0,1]
        tar_img = TF.to_tensor(tar_img)
        seg_img = torch.from_numpy(np.array(seg_img)).long()  # [H,W], Long

        weather_id = WEATHER2ID[s['weather']]
        return inp_img, tar_img, seg_img, weather_id, s['weather']


class ValLoadDatasetMulti(Dataset):
    def __init__(self, sample_list, crop_size=256, ignore_index=255):
        self.samples = sample_list
        self.crop_size = crop_size
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        inp_img = Image.open(s['inp']).convert('RGB')
        tar_img = Image.open(s['tar']).convert('RGB')
        seg_img = Image.open(s['seg']).convert('L')

        # 1) 정규화 + 패딩
        th, tw = _normalize_crop_size(self.crop_size)
        inp_img = _pad_if_needed(inp_img, th, tw, fill=0)
        tar_img = _pad_if_needed(tar_img, th, tw, fill=0)
        seg_img = _pad_if_needed(seg_img, th, tw, fill=self.ignore_index)

        # 2) 크롭(검증은 플립 없음)
        i, j, h, w = T.RandomCrop.get_params(inp_img, (th, tw))
        crop = lambda im: TF.crop(im, i, j, h, w)
        inp_img, tar_img, seg_img = map(crop, (inp_img, tar_img, seg_img))

        # 3) Tensor 변환
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        seg_img = torch.from_numpy(np.array(seg_img)).long()

        weather_id = WEATHER2ID[s['weather']]
        filename = Path(s['tar']).stem
        return inp_img, tar_img, seg_img, weather_id, filename


class TestLoadDatasetMulti(Dataset):
    def __init__(self, sample_list):
        self.samples = sample_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        inp_img = Image.open(s['inp']).convert('RGB')
        tar_img = Image.open(s['tar']).convert('RGB')
        seg_img = Image.open(s['seg']).convert('L')

        # 테스트는 원본 그대로(필요 시 프로젝트 규칙에 맞춰 resize/pad 추가 가능)
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        seg_img = torch.from_numpy(np.array(seg_img)).long()

        weather_id = WEATHER2ID[s['weather']]
        filename = Path(s['tar']).stem
        return inp_img, tar_img, seg_img, weather_id, filename
