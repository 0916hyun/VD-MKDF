# # test_loop_rgbd.py

# import os
# import torch
# import torch.nn.functional as F
# from segmentation_models_pytorch.metrics.functional import get_stats

# from test_RGBD import parse_args, build_dataloaders
# from RGBD_trainer import Trainer, WeatherStat  # WeatherStat for CSV append
# from models.make_model import make_model  # remove helper if unused
# from RGBD_trainer import WEATHER_NAMES

# def run_test(trainer, loader, label, ckpt_path):
#     start_ep = trainer._load_ckpt(ckpt_path)
#     # 기존 _test 메서드와 동일하게 loss, mIoU, per-weather 까지 얻기
#     tot_loss, seg_loss, mean_iou, class_iou_dict, per_weather, pixel_acc, class_acc = trainer._test(loader)

#     # CSV 기록
#     trainer._append_test_csv(
#         path      = trainer.csv_test,
#         epoch     = start_ep,
#         tot_loss  = tot_loss,
#         seg_loss  = seg_loss,
#         miou      = mean_iou,
#         pixel_acc = pixel_acc,
#         class_acc = class_acc,
#         per_weather= per_weather,
#         class_iou  = class_iou_dict
#     )

#     # 결과 출력
#     print(f"\n=== Test on [{label}] (epoch {start_ep}) ===")
#     print(f" Total Loss: {tot_loss:.4f}, Seg Loss: {seg_loss:.4f}")
#     print(f" Mean IoU: {mean_iou:.4f}, Pixel Acc: {pixel_acc:.4f}, Class Acc: {class_acc:.4f}\n")

#     print(">> Weather-wise Metrics:")
#     for wid, stats in per_weather.items():
#         name = WEATHER_NAMES[wid]
#         print(f"  - {name:6s}| Loss {stats[0]:.4f}, mIoU {stats[1]:.4f}")

#     print("\n>> Class IoU:")
#     for cls, iou in class_iou_dict.items():
#         print(f"  - Class {cls:2d}: IoU {iou:.4f}")

# def main():
#     args = parse_args()
#     test_loader = build_dataloaders(args)

#     # RGB-D 모델 생성
#     model = make_model(
#     model_name      = 'unetpp_rgbd_baseline',  # factory 내부에서 UNetPP_RGBD_Baseline을 반환하도록 설정된 이름
#     encoder_name    = 'resnet50',
#     encoder_weights = 'imagenet',
#     in_channels     = 3,
#     depth_channels  = 1,
#     n_classes       = 12,
#     device          = 'cuda',                  # CUDA/CPU 장치
# )

#     trainer = Trainer(
#         model       = model,
#         save_root   = args.data_root,      # unused
#         weights_dir = args.weights_dir,
#         results_dir = args.results_dir,
#         device      = args.device,
#         lr          = 1e-4,
#         num_epochs  = 0,
#         n_classes   = args.num_classes,
#         ignore_index= args.ignore_index
#     ).to(args.device)

#     # best_val_loss → best_val_miou 순으로 테스트
#     for label, ckpt_name in [("best_val_loss", args.ckpt_loss),
#                              ("best_val_miou", args.ckpt_miou)]:
#         ckpt_path = os.path.join(args.weights_dir, ckpt_name)
#         run_test(trainer, test_loader, label, ckpt_path)

# if __name__ == "__main__":
#     main()

# test_loop_rgbd.py

import os
import torch
import torch.nn.functional as F
from segmentation_models_pytorch.metrics.functional import get_stats

from test_RGBD import parse_args, build_dataloaders
from RGBD_trainer import Trainer, WeatherStat  # WeatherStat for CSV append
from models.make_model import make_model  # remove helper if unused
from RGBD_trainer import WEATHER_NAMES

def run_test(trainer, loader, label, ckpt_path):
    start_ep = trainer._load_ckpt(ckpt_path)
    # 기존 _test 메서드와 동일하게 loss, mIoU, per-weather 까지 얻기
    tot_loss, seg_loss, mean_iou, class_iou_dict, per_weather, pixel_acc, class_acc = trainer._test(loader)

    # CSV 기록
    row = [
        start_ep,                      # epoch
        tot_loss, seg_loss,            # total_loss, seg_loss
        mean_iou,                      # mIoU
        pixel_acc, class_acc,          # pixel_acc, class_acc
        # weather-wise (rain, snow, fog, flare 순서로 loss, mIoU)
        per_weather[0][0], per_weather[0][1],
        per_weather[1][0], per_weather[1][1],
        per_weather[2][0], per_weather[2][1],
        per_weather[3][0], per_weather[3][1],
    ] + [class_iou_dict.get(i, 0.0) for i in range(trainer.n_classes)]

    # CSV 기록 파일은 trainer.csv_test_all 사용
    trainer._append_csv(trainer.csv_test_all, row)

    # 결과 출력
    print(f"\n=== Test on [{label}] (epoch {start_ep}) ===")
    print(f" Total Loss: {tot_loss:.4f}, Seg Loss: {seg_loss:.4f}")
    print(f" Mean IoU: {mean_iou:.4f}, Pixel Acc: {pixel_acc:.4f}, Class Acc: {class_acc:.4f}\n")

    print(">> Weather-wise Metrics:")
    for wid, stats in per_weather.items():
        name = WEATHER_NAMES[wid]
        print(f"  - {name:6s}| Loss {stats[0]:.4f}, mIoU {stats[1]:.4f}")

    print("\n>> Class IoU:")
    for cls, iou in class_iou_dict.items():
        print(f"  - Class {cls:2d}: IoU {iou:.4f}")

def main():
    args = parse_args()
    test_loader = build_dataloaders(args)

    # RGB-D 모델 생성
    model = make_model(
        model_name      = 'segb5_rgbd_fuse_stageselective',
        n_classes       = 12,
        patch_sizes     = [7, 7, 5, 3], 
        fuse_stages     = [0,1,2,3], 
        attn_variant    = 'base',
        cp_rank         = 16,
        device          = 'cuda'
    )

    trainer = Trainer(
        model       = model,
        save_root   = args.data_root,      # unused
        weights_dir = args.weights_dir,
        results_dir = args.results_dir,
        device      = args.device,
        lr          = 1e-4,
        num_epochs  = 0,
        n_classes   = args.num_classes,
        ignore_index= args.ignore_index
    ).to(args.device)

    # best_val_loss → best_val_miou 순으로 테스트
    for label, ckpt_name in [("best_val_loss", args.ckpt_loss),
                             ("best_val_miou", args.ckpt_miou)]:
        ckpt_path = os.path.join(args.weights_dir, ckpt_name)
        run_test(trainer, test_loader, label, ckpt_path)

if __name__ == "__main__":
    main()
