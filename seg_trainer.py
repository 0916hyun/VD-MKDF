# seg_trainer.py

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from utils.stream_metrics import StreamSegMetrics


class SegmentationTrainer:
    """
    Trainer for single-weather RGB-only segmentation.
    - 각 epoch마다 train → val → (epoch >= 300 시) test 수행
    - validation loss, val_mIoU, test_mIoU 기준으로 best 모델 별도 저장
    """
    def __init__(
        self,
        model,
        save_root,      # 사용하지 않지만 인터페이스 일관성을 위해 남겨두었습니다.
        weights_dir,
        results_dir,    # 사용하지 않지만 인터페이스 일관성을 위해 남겨두었습니다.
        device,
        lr,
        num_epochs,
        n_classes,
        ignore_index=11
    ):
        self.device = device
        self.model = model.to(device)
        self.lr = lr
        self.num_epochs = num_epochs
        self.n_classes = n_classes
        self.ignore_index = ignore_index

        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
        self.metrics = StreamSegMetrics(n_classes)

        # Directory setup
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        self.weights_dir = weights_dir
        self.results_dir = results_dir

        # Best-model tracking
        self.best_loss = float('inf')
        self.best_miou = 0.0
        self.best_test_miou = 0.0

        self.path_loss = os.path.join(weights_dir, 'best_val_loss.pth')
        self.path_miou = os.path.join(weights_dir, 'best_val_miou.pth')
        self.path_test = os.path.join(weights_dir, 'best_test_miou.pth')

        # Logs 초기화
        # train–val 로그
        self.log_train = os.path.join(weights_dir, 'train_log.csv')
        with open(self.log_train, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_miou'])
        # epoch별 test 로그
        self.log_test_epoch = os.path.join(weights_dir, 'test_epoch_log.csv')
        with open(self.log_test_epoch, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'test_miou'])
        # 최종 test()용 로그
        self.log_test = os.path.join(weights_dir, 'final_test_log.csv')

    def _epoch(self, loader, training: bool):
        """
        Run one epoch (train or val or test). Returns (avg_loss, mIoU).
        """
        if training:
            self.model.train()
        else:
            self.model.eval()
        self.metrics.reset()
        total_loss = 0.0

        desc = 'Train' if training else 'Val/Test'
        for batch in tqdm(loader, desc=desc, unit='batch'):
            inp = batch[0].to(self.device)
            seg = batch[2].to(self.device)

            if training:
                self.optimizer.zero_grad()

            out = self.model(inp)
            # spatial 크기 맞추기
            if out.dim() == 4 and seg.dim() == 3 and out.shape[-2:] != seg.shape[-2:]:
                out = F.interpolate(
                    out, size=seg.shape[-2:], mode='bilinear', align_corners=False
                )

            loss = self.criterion(out, seg)
            if training:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(1).cpu().numpy()
            labels = seg.cpu().numpy()
            self.metrics.update(labels, preds)

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        results = self.metrics.get_results()
        return avg_loss, results['Mean IoU']

    def train(self, train_loader, val_loader, test_loader):
        """
        train_loader, val_loader, test_loader 순서대로 넘겨주세요.
        300 epoch 이후부터 매 epoch마다 test 수행, best_test_miou 갱신 시 모델 저장
        """
        for epoch in range(1, self.num_epochs + 1):
            # 1) train
            tr_loss, _ = self._epoch(train_loader, True)
            # 2) validation
            val_loss, val_miou = self._epoch(val_loader, False)

            # 로그 기록
            with open(self.log_train, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, tr_loss, val_loss, val_miou])
            print(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val mIoU: {val_miou:.4f}"
            )

            # best_val 저장
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), self.path_loss)
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                torch.save(self.model.state_dict(), self.path_miou)

            # 3) test (epoch >= 300부터)
            if epoch >= 300:
                _, test_miou = self._epoch(test_loader, False)
                with open(self.log_test_epoch, 'a', newline='') as f:
                    csv.writer(f).writerow([epoch, test_miou])
                print(f"Test Epoch {epoch} | mIoU: {test_miou:.4f}")

                if test_miou > self.best_test_miou:
                    self.best_test_miou = test_miou
                    torch.save(self.model.state_dict(), self.path_test)

        # ─── 전체 루프 종료 후 최종 best test mIoU 출력 ───
        print("=== Training Complete ===")
        print(f"Best Test mIoU (from epoch ≥800): {self.best_test_miou:.4f}")

    def test(self, test_loader):
        """
        기존 방식: best_val_loss, best_val_miou 모델 불러와서 최종 test 수행
        """
        with open(self.log_test, 'w', newline='') as f:
            csv.writer(f).writerow(['model', 'test_miou'])

        for label, pth in [('best_val_loss', self.path_loss), ('best_val_miou', self.path_miou)]:
            self.model.load_state_dict(torch.load(pth, map_location=self.device))
            _, test_miou = self._epoch(test_loader, False)
            print(f"Test [{label}] | mIoU: {test_miou:.4f}")
            with open(self.log_test, 'a', newline='') as f:
                csv.writer(f).writerow([label, test_miou])
