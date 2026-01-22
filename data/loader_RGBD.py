import os
import random
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms.functional as F
from pathlib import Path
import cv2

WEATHER2ID = {'rain': 0, 'snow': 1, 'fog': 2, 'flare': 3}

import os
import random
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms.functional as F
from pathlib import Path
import cv2

WEATHER2ID = {'rain': 0, 'snow': 1, 'fog': 2, 'flare': 3}

def _normalize_crop_size(size):
    """int 또는 (H,W) → (th, tw)로 정규화"""
    if isinstance(size, (tuple, list)):
        return int(size[0]), int(size[1])
    return int(size), int(size)

def _pad_if_needed(img, th, tw, fill=0):
    """PIL.Image 입력 기준 (W,H). 부족분을 오른쪽/아래로 패딩"""
    W, H = img.size
    pad_h = max(0, th - H)
    pad_w = max(0, tw - W)
    if pad_h > 0 or pad_w > 0:
        img = TF.pad(img, (0, 0, pad_w, pad_h), fill=fill)
    return img, pad_h, pad_w


class TrainLoadDatasetMulti_RGBD(Dataset):
    def __init__(self, sample_list, crop_size, base_dir, depth_root):
        self.samples    = sample_list
        self.crop_size  = crop_size
        self.base_dir   = Path(base_dir)
        self.depth_root = Path(depth_root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load RGB & seg
        inp_img = Image.open(s['inp']).convert("RGB")
        tar_img = Image.open(s['tar']).convert("RGB")
        seg_img = Image.open(s['seg']).convert("L")

        # # Random crop + flip
        # i, j, h, w = T.RandomCrop.get_params(inp_img, (self.crop_size, self.crop_size))
        # _crop = lambda img: F.crop(img, i, j, h, w)
        # inp_img, tar_img, seg_img = map(_crop, (inp_img, tar_img, seg_img))
        # do_flip = (random.random() > 0.5)
        # if do_flip:
        #     inp_img, tar_img, seg_img = map(F.hflip, (inp_img, tar_img, seg_img))

        # # Depth load & normalize via PIL crop + flip
        # rel = Path(s['inp']).relative_to(self.base_dir)
        # depth_path = self.depth_root / rel.with_suffix('.png')
        # depth_img = Image.open(depth_path)    # PIL로 16-bit depth 읽기

        # # RGB/seg와 동일한 i, j, h, w로 crop
        # depth_img = TF.crop(depth_img, i, j, h, w)
        # if do_flip:
        #     depth_img = TF.hflip(depth_img)

        # --- (1) crop_size 정규화: int 또는 (H, W) 모두 지원 ---
        size = self.crop_size
        if isinstance(size, (tuple, list)):
            th, tw = int(size[0]), int(size[1])   # H, W
        else:
            th = tw = int(size)

        # --- (2) 작은 이미지 패딩 가드 (PIL 기준: .size = (W, H)) ---
        W, H = inp_img.size
        pad_h = max(0, th - H)
        pad_w = max(0, tw - W)
        if pad_h > 0 or pad_w > 0:
            pad = (0, 0, pad_w, pad_h)  # (left, top, right, bottom)
            inp_img = TF.pad(inp_img, pad, fill=0)
            tar_img = TF.pad(tar_img, pad, fill=0)
            seg_fill = 255  # 프로젝트 ignore_index에 맞추어 조정 가능
            seg_img = TF.pad(seg_img, pad, fill=seg_fill)
            # H, W 업데이트 (필요 시)
            # W, H = inp_img.size

        # --- (3) 랜덤 크롭 + 좌우 플립 ---
        i, j, h, w = T.RandomCrop.get_params(inp_img, (th, tw))
        _crop = lambda img: TF.crop(img, i, j, h, w)
        inp_img, tar_img, seg_img = map(_crop, (inp_img, tar_img, seg_img))

        do_flip = (random.random() > 0.5)
        if do_flip:
            inp_img, tar_img, seg_img = map(TF.hflip, (inp_img, tar_img, seg_img))

        # --- (4) Depth: 동일 패딩/크롭/플립 적용 ---
        rel = Path(s['inp']).relative_to(self.base_dir)
        depth_path = self.depth_root / rel.with_suffix('.png')
        depth_img = Image.open(depth_path)   # 16-bit depth

        # depth에도 동일 패딩(있었다면) 적용
        if pad_h > 0 or pad_w > 0:
            pad = (0, 0, pad_w, pad_h)
            depth_img = TF.pad(depth_img, pad, fill=0)

        depth_img = TF.crop(depth_img, i, j, h, w)
        if do_flip:
            depth_img = TF.hflip(depth_img)

            """
            여기까지 임시수정
            """

        # numpy 변환 → [0,1] 스케일링 → tensor
        depth_np = np.array(depth_img).astype(np.float32) / 65535.0
        depth = torch.from_numpy(depth_np).unsqueeze(0)  # [1,H,W], float32

        # RGB: [0,1] float32
        inp = TF.to_tensor(inp_img)   # [3,H,W]
        tar = TF.to_tensor(tar_img)

        # Seg: Long
        seg = torch.from_numpy(np.array(seg_img, dtype=np.int64))  # [H,W]

        weather_id = WEATHER2ID[s['weather']]
        return inp, tar, depth, seg, weather_id


class ValLoadDatasetMulti_RGBD(Dataset):
    def __init__(self, sample_list, crop_size, base_dir, depth_root, ignore_index=255):
        self.samples    = sample_list
        self.crop_size  = crop_size
        self.base_dir   = Path(base_dir)
        self.depth_root = Path(depth_root)
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        inp_img = Image.open(s['inp']).convert("RGB")
        tar_img = Image.open(s['tar']).convert("RGB")
        seg_img = Image.open(s['seg']).convert("L")

        # # Crop only
        # i, j, h, w = T.RandomCrop.get_params(inp_img, (self.crop_size, self.crop_size))
        # _crop = lambda img: F.crop(img, i, j, h, w)
        # inp_img, tar_img, seg_img = map(_crop, (inp_img, tar_img, seg_img))

        # # Depth load & normalize via PIL crop
        # rel = Path(s['inp']).relative_to(self.base_dir)
        # depth_path = self.depth_root / rel.with_suffix('.png')
        # depth_img = Image.open(depth_path)                     # PIL로 읽기
        # depth_img = TF.crop(depth_img, i, j, h, w)            # 동일한 crop
        # # no flip in validation

        # --- (1) crop_size 정규화 ---
        th, tw = _normalize_crop_size(self.crop_size)

        inp_img, pad_h, pad_w = _pad_if_needed(inp_img, th, tw, fill=0)
        tar_img, _, _          = _pad_if_needed(tar_img, th, tw, fill=0)
        seg_img, _, _          = _pad_if_needed(seg_img, th, tw, fill=self.ignore_index)

        i, j, h, w = T.RandomCrop.get_params(inp_img, (th, tw))
        crop = lambda im: TF.crop(im, i, j, h, w)
        inp_img, tar_img, seg_img = map(crop, (inp_img, tar_img, seg_img))

        rel = Path(s['inp']).relative_to(self.base_dir)
        depth_path = self.depth_root / rel.with_suffix('.png')
        depth_img = Image.open(depth_path)
        if pad_h > 0 or pad_w > 0:
            depth_img = TF.pad(depth_img, (0, 0, pad_w, pad_h), fill=0)
        depth_img = TF.crop(depth_img, i, j, h, w)

        """
        여기까지 임시수정
        """

        depth_np = np.array(depth_img).astype(np.float32) / 65535.0
        depth = torch.tensor(depth_np, dtype=torch.float32).unsqueeze(0)
        # depth = depth.repeat(3, 1, 1).contiguous() # 3채널 입력용 채널복제

        # To tensor
        inp   = torch.tensor(np.array(inp_img), dtype=torch.float32).permute(2,0,1)
        tar   = torch.tensor(np.array(tar_img), dtype=torch.float32).permute(2,0,1)
        seg   = torch.tensor(np.array(seg_img), dtype=torch.long)

        weather_id = WEATHER2ID[s['weather']]
        return inp.clone(), tar.clone(), depth.clone(), seg.clone(), weather_id


class TestLoadDatasetMulti_RGBD(Dataset):
    def __init__(self, sample_list, base_dir, depth_root):
        self.samples    = sample_list
        self.base_dir   = Path(base_dir)
        self.depth_root = Path(depth_root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        inp_img = Image.open(s['inp']).convert("RGB")
        tar_img = Image.open(s['tar']).convert("RGB")
        seg_img = Image.open(s['seg']).convert("L")

        # Depth load via PIL (no crop/flip for test, or apply full-image logic)
        rel = Path(s['inp']).relative_to(self.base_dir)
        depth_path = self.depth_root / rel.with_suffix('.png')
        depth_img = Image.open(depth_path)
        depth_np  = np.array(depth_img).astype(np.float32) / 65535.0
        depth = torch.tensor(depth_np, dtype=torch.float32).unsqueeze(0)
        # depth = depth.repeat(3, 1, 1).contiguous() # 3채널 입력용 채널복제

        # To tensor
        inp   = torch.tensor(np.array(inp_img), dtype=torch.float32).permute(2,0,1)
        tar   = torch.tensor(np.array(tar_img), dtype=torch.float32).permute(2,0,1)
        seg   = torch.tensor(np.array(seg_img), dtype=torch.long)

        weather_id = WEATHER2ID[s['weather']]
        return inp.clone(), tar.clone(), depth.clone(), seg.clone(), weather_id



class TrainLoadDatasetMulti_RGBD(Dataset):
    def __init__(self, sample_list, crop_size, base_dir, depth_root, ignore_index=255):
        self.samples    = sample_list
        self.crop_size  = crop_size
        self.base_dir   = Path(base_dir)
        self.depth_root = Path(depth_root)
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load RGB & seg
        inp_img = Image.open(s['inp']).convert("RGB")
        tar_img = Image.open(s['tar']).convert("RGB")
        seg_img = Image.open(s['seg']).convert("L")

        # # Random crop + flip
        # i, j, h, w = T.RandomCrop.get_params(inp_img, (self.crop_size, self.crop_size))
        # _crop = lambda img: F.crop(img, i, j, h, w)
        # inp_img, tar_img, seg_img = map(_crop, (inp_img, tar_img, seg_img))
        # do_flip = (random.random() > 0.5)
        # if do_flip:
        #     inp_img, tar_img, seg_img = map(F.hflip, (inp_img, tar_img, seg_img))

        # # Depth load & normalize via PIL crop + flip
        # rel = Path(s['inp']).relative_to(self.base_dir)
        # depth_path = self.depth_root / rel.with_suffix('.png')
        # depth_img = Image.open(depth_path)    # PIL로 16-bit depth 읽기

        # # RGB/seg와 동일한 i, j, h, w로 crop
        # depth_img = TF.crop(depth_img, i, j, h, w)
        # if do_flip:
        #     depth_img = TF.hflip(depth_img)

        # (1) 크롭 사이즈 정규화
        th, tw = _normalize_crop_size(self.crop_size)

        # (2) 작은 이미지 패딩 (RGB/Seg)
        inp_img, pad_h, pad_w = _pad_if_needed(inp_img, th, tw, fill=0)
        tar_img, _, _          = _pad_if_needed(tar_img, th, tw, fill=0)
        seg_img, _, _          = _pad_if_needed(seg_img, th, tw, fill=self.ignore_index)

        # (3) 크롭 파라미터 샘플링 + 동일 크롭
        i, j, h, w = T.RandomCrop.get_params(inp_img, (th, tw))
        crop = lambda im: TF.crop(im, i, j, h, w)
        inp_img, tar_img, seg_img = map(crop, (inp_img, tar_img, seg_img))

        # (4) 좌우 플립(동일)
        do_flip = (random.random() > 0.5)
        if do_flip:
            inp_img, tar_img, seg_img = map(TF.hflip, (inp_img, tar_img, seg_img))

        # (5) Depth도 동일 패딩/크롭/플립
        rel = Path(s['inp']).relative_to(self.base_dir)
        depth_path = self.depth_root / rel.with_suffix('.png')
        depth_img = Image.open(depth_path)  # 16-bit png 가정

        if pad_h > 0 or pad_w > 0:
            depth_img = TF.pad(depth_img, (0, 0, pad_w, pad_h), fill=0)

        depth_img = TF.crop(depth_img, i, j, h, w)
        if do_flip:
            depth_img = TF.hflip(depth_img)

            """
            여기까지 임시수정
            """

        # numpy 변환 → [0,1] 스케일링 → tensor
        depth_np = np.array(depth_img).astype(np.float32) / 65535.0
        depth = torch.tensor(depth_np, dtype=torch.float32).unsqueeze(0)
        # depth = depth.repeat(3, 1, 1).contiguous() # 3채널 입력용 채널복제

        # To tensor (ensuring resizable storage)
        inp = torch.tensor(np.array(inp_img), dtype=torch.float32).permute(2,0,1)
        tar = torch.tensor(np.array(tar_img), dtype=torch.float32).permute(2,0,1)
        seg = torch.tensor(np.array(seg_img), dtype=torch.long)

        inp = inp.clone()
        tar = tar.clone()
        depth = depth.clone()
        seg = seg.clone()

        weather_id = WEATHER2ID[s['weather']]
        return inp, tar, depth, seg, weather_id


class ValLoadDatasetMulti_RGBD(Dataset):
    def __init__(self, sample_list, crop_size, base_dir, depth_root):
        self.samples    = sample_list
        self.crop_size  = crop_size
        self.base_dir   = Path(base_dir)
        self.depth_root = Path(depth_root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        inp_img = Image.open(s['inp']).convert("RGB")
        tar_img = Image.open(s['tar']).convert("RGB")
        seg_img = Image.open(s['seg']).convert("L")

        # # Crop only
        # i, j, h, w = T.RandomCrop.get_params(inp_img, (self.crop_size, self.crop_size))
        # _crop = lambda img: F.crop(img, i, j, h, w)
        # inp_img, tar_img, seg_img = map(_crop, (inp_img, tar_img, seg_img))

        # # Depth load & normalize via PIL crop
        # rel = Path(s['inp']).relative_to(self.base_dir)
        # depth_path = self.depth_root / rel.with_suffix('.png')
        # depth_img = Image.open(depth_path)                     # PIL로 읽기
        # depth_img = TF.crop(depth_img, i, j, h, w)            # 동일한 crop
        # # no flip in validation

        # --- (1) crop_size 정규화 ---
        size = self.crop_size
        if isinstance(size, (tuple, list)):
            th, tw = int(size[0]), int(size[1])
        else:
            th = tw = int(size)

        # --- (2) 작은 이미지 패딩 가드 ---
        W, H = inp_img.size
        pad_h = max(0, th - H)
        pad_w = max(0, tw - W)
        if pad_h > 0 or pad_w > 0:
            pad = (0, 0, pad_w, pad_h)
            inp_img = TF.pad(inp_img, pad, fill=0)
            tar_img = TF.pad(tar_img, pad, fill=0)
            seg_fill = 255  # 프로젝트 ignore_index에 맞추어 조정 가능
            seg_img = TF.pad(seg_img, pad, fill=seg_fill)

        # --- (3) 크롭 (검증은 플립 X) ---
        i, j, h, w = T.RandomCrop.get_params(inp_img, (th, tw))
        _crop = lambda img: TF.crop(img, i, j, h, w)
        inp_img, tar_img, seg_img = map(_crop, (inp_img, tar_img, seg_img))

        # --- (4) Depth에도 동일 패딩/크롭 적용 ---
        rel = Path(s['inp']).relative_to(self.base_dir)
        depth_path = self.depth_root / rel.with_suffix('.png')
        depth_img = Image.open(depth_path)

        if pad_h > 0 or pad_w > 0:
            pad = (0, 0, pad_w, pad_h)
            depth_img = TF.pad(depth_img, pad, fill=0)

        depth_img = TF.crop(depth_img, i, j, h, w)

        """
        여기까지 임시수정
        """

        depth_np = np.array(depth_img).astype(np.float32) / 65535.0
        depth = torch.tensor(depth_np, dtype=torch.float32).unsqueeze(0)
        # depth = depth.repeat(3, 1, 1).contiguous() # 3채널 입력용 채널복제

        # To tensor
        inp   = torch.tensor(np.array(inp_img), dtype=torch.float32).permute(2,0,1)
        tar   = torch.tensor(np.array(tar_img), dtype=torch.float32).permute(2,0,1)
        seg   = torch.tensor(np.array(seg_img), dtype=torch.long)

        weather_id = WEATHER2ID[s['weather']]
        return inp.clone(), tar.clone(), depth.clone(), seg.clone(), weather_id


class TestLoadDatasetMulti_RGBD(Dataset):
    def __init__(self, sample_list, base_dir, depth_root):
        self.samples    = sample_list
        self.base_dir   = Path(base_dir)
        self.depth_root = Path(depth_root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        inp_img = Image.open(s['inp']).convert("RGB")
        tar_img = Image.open(s['tar']).convert("RGB")
        seg_img = Image.open(s['seg']).convert("L")

        # Depth load via PIL (no crop/flip for test, or apply full-image logic)
        rel = Path(s['inp']).relative_to(self.base_dir)
        depth_path = self.depth_root / rel.with_suffix('.png')
        depth_img = Image.open(depth_path)
        depth_np  = np.array(depth_img).astype(np.float32) / 65535.0
        depth = torch.tensor(depth_np, dtype=torch.float32).unsqueeze(0)
        # depth = depth.repeat(3, 1, 1).contiguous() # 3채널 입력용 채널복제

        # To tensor
        inp   = torch.tensor(np.array(inp_img), dtype=torch.float32).permute(2,0,1)
        tar   = torch.tensor(np.array(tar_img), dtype=torch.float32).permute(2,0,1)
        seg   = torch.tensor(np.array(seg_img), dtype=torch.long)

        weather_id = WEATHER2ID[s['weather']]
        return inp.clone(), tar.clone(), depth.clone(), seg.clone(), weather_id
