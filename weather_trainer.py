import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv
import os

class WeatherClassificationTrainer:
    """
    Trainer for 4-class weather classification using MobileNetV2.
    Saves best models based on validation loss and validation accuracy,
    and evaluates both on the test set.
    """
    def __init__(self, model: nn.Module, save_dir: str, device: str,
                 lr: float = 1e-4, num_epochs: int = 50):
        self.model = model.to(device)
        self.device = device
        self.epochs = num_epochs

        # Loss & optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Save directory
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_loss_path = os.path.join(save_dir, 'best_val_loss.pth')
        self.best_acc_path = os.path.join(save_dir, 'best_val_acc.pth')

        # Logs
        self.train_log = os.path.join(save_dir, 'train_log.csv')
        self.test_log = os.path.join(save_dir, 'test_log.csv')
        with open(self.train_log, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_acc', 'val_acc', 'val_loss'])

    def evaluate(self, loader):
        """
        Compute accuracy and average loss over the given DataLoader.
        Returns (accuracy, avg_loss).
        """
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for rgb, _, _, _, wid in loader:
                rgb = rgb.to(self.device)
                wid = wid.to(self.device)
                outputs = self.model(rgb)
                loss = self.criterion(outputs, wid)
                preds = outputs.argmax(dim=1)
                correct += (preds == wid).sum().item()
                loss_sum += loss.item() * wid.size(0)
                total += wid.size(0)
        acc = correct / total if total > 0 else 0.0
        avg_loss = loss_sum / total if total > 0 else 0.0
        return acc, avg_loss

    def train(self, train_loader, val_loader):
        """
        Train for self.epochs epochs, save best models by val loss and val acc.
        Records metrics to CSV.
        """
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            correct = 0
            total = 0
            for rgb, _, _, _, wid in tqdm(train_loader, desc=f'[Train Epoch {epoch}]'):
                rgb = rgb.to(self.device)
                wid = wid.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(rgb)
                loss = self.criterion(outputs, wid)
                loss.backward()
                self.optimizer.step()

                preds = outputs.argmax(dim=1)
                correct += (preds == wid).sum().item()
                total += wid.size(0)

            train_acc = correct / total if total > 0 else 0.0
            val_acc, val_loss = self.evaluate(val_loader)

            print(f'Epoch {epoch}/{self.epochs} | '
                  f'Train Acc: {train_acc:.4f} | '
                  f'Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}')

            # Append to train log
            with open(self.train_log, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, train_acc, val_acc, val_loss])

            # Save best by loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_loss_path)
            # Save best by accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.best_acc_path)

    def test(self, test_loader):
        """
        Evaluate and log performance of both best-loss and best-acc models on the test set.
        """
        # Prepare test log
        with open(self.test_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'test_acc', 'test_loss'])

        for label, model_path in [('best_val_loss', self.best_loss_path),
                                  ('best_val_acc', self.best_acc_path)]:
            # Load saved weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            acc, loss = self.evaluate(test_loader)
            print(f'[Test | {label}] Acc: {acc:.4f} | Loss: {loss:.4f}')

            # Append to test log
            with open(self.test_log, 'a', newline='') as f:
                csv.writer(f).writerow([label, acc, loss])
