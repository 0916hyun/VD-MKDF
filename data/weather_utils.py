# weather_utils.py
from pathlib import Path
import random
from collections import defaultdict

# ───────────────────────────────
# 1) 전역 상수
# ───────────────────────────────
WEATHER2ID = {'rain': 0, 'snow': 1, 'fog': 2, 'flare': 3}

# ───────────────────────────────
# 2) scene×weather 샘플 리스트 생성
# ───────────────────────────────
def make_sample_list(root, scene_dir, weather_dirs, seg_dir):
    """
    return: [{'weather':'rain', 'inp':..., 'tar':..., 'seg':...}, ...]
    """
    seg_root = Path(root) / seg_dir
    sample_list = []
    for seg_path in sorted(seg_root.glob('*.png')):
        scene_id = seg_path.stem
        tar_path = Path(root) / scene_dir / f'{scene_id}.png'
        for w, wdir in weather_dirs.items():
            inp_path = Path(root) / wdir / f'{scene_id}.png'
            sample_list.append(
                dict(scene=scene_id, weather=w, inp=inp_path,
                     tar=tar_path, seg=seg_path)
            )
    return sample_list

# ───────────────────────────────
# 3) scene 단위 split 유틸
# ───────────────────────────────
def split_by_scene(sample_list, ratios=(6, 2, 2), seed=1234):
    """
    ratios: (train, val, test) 비율. 합은 아무 값이어도 됨.
    반환: train_samples, val_samples, test_samples  (순서 튜플)
    """
    # 3-1. scene → 샘플들 dict
    bucket = defaultdict(list)
    for item in sample_list:
        bucket[item['scene']].append(item)

    scenes = list(bucket.keys())
    random.seed(seed)
    random.shuffle(scenes)

    # 3-2. 비율에 따라 장면 분할
    n_total   = len(scenes)
    n_train   = int(ratios[0] / sum(ratios) * n_total)
    n_val     = int(ratios[1] / sum(ratios) * n_total)

    train_scenes = scenes[:n_train]
    val_scenes   = scenes[n_train:n_train + n_val]
    test_scenes  = scenes[n_train + n_val:]

    # 3-3. scene 목록을 샘플 리스트로 환산
    train_samples = [s for sc in train_scenes for s in bucket[sc]]
    val_samples   = [s for sc in val_scenes   for s in bucket[sc]]
    test_samples  = [s for sc in test_scenes  for s in bucket[sc]]

    return train_samples, val_samples, test_samples
