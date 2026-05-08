import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils import data_generator
from .model import PatchTreeTCN
import argparse
import time
import os
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=720)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--nhid', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--stride', type=int, default=2)
parser.add_argument('--levels', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--patience', type=int, default=30)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_dir = f"logs/weather_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)

file_path = os.path.join(os.path.dirname(__file__), 'data', 'weather.csv')
train_loader, val_loader, test_loader = data_generator(
    file_path, args.batch_size, args.seq_len, args.pred_len
)

input_channels = 21
output_channels = 1
channel_sizes = [args.nhid] * args.levels
model = PatchTreeTCN(
    input_channels, output_channels, channel_sizes,
    kernel_size=3, dropout=args.dropout, stride=args.stride,
    seq_len=args.seq_len, pred_len=args.pred_len, patch_size=args.patch_size
).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
criterion = nn.MSELoss()

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience)

best_val_loss = float('inf')
best_epoch = 0
early_stop_counter = 0

def calculate_metrics(output, target):
    mae = torch.mean(torch.abs(output - target)).item()
    mape = torch.mean(torch.abs((output - target) / (target + 1e-8))).item() * 100
    rmse = torch.sqrt(torch.mean((output - target) ** 2)).item()
    return mae, mape, rmse

def evaluate(loader):
    model.eval()
    total_loss = 0
    total_mae = 0
    total_mape = 0
    total_rmse = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if output.shape[1] == 1 and target.shape[1] > 1:
                target = target[:, -1:, :]

            loss = criterion(output, target).item()
            mae, mape, rmse = calculate_metrics(output, target)

            total_loss += loss
            total_mae += mae
            total_mape += mape
            total_rmse += rmse

    num_batches = len(loader)
    return (total_loss / num_batches, total_mae / num_batches,
            total_mape / num_batches, total_rmse / num_batches)

def train():
    global best_val_loss, best_epoch, early_stop_counter

    total_training_time = 0

    log_file = os.path.join(log_dir, "training_log.txt")

    with open(log_file, 'w') as f:
        f.write(f"TCN ETTh1 Model Training Log\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameters: {vars(args)}\n")
        f.write("=" * 100 + "\n")
        f.write("Epoch\tTrain_Loss\tVal_Loss\tMAE\t\tMAPE\t\tRMSE\t\tTime(s)\tLR\n")
        f.write("=" * 100 + "\n")

    print(f"Start training... Logs will be saved to {log_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            if model.channel_proj.out_channels == 1 and target.shape[1] > 1:
                target = target[:, -1:, :]

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

        val_loss, val_mae, val_mape, val_rmse = evaluate(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time

        print(f"Epoch {epoch:3d} | Train Loss: {total_loss/len(train_loader):.6f} | "
              f"Val Loss: {val_loss:.6f} | MAE: {val_mae:.4f} | "
              f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")

        with open(log_file, 'a') as f:
            f.write(f"{epoch}\t{total_loss/len(train_loader):.6f}\t{val_loss:.6f}\t"
                    f"{val_mae:.4f}\t{val_mape:.2f}\t{val_rmse:.4f}\t{epoch_time:.2f}\t{current_lr:.6f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            print(f"--> Best model saved (Val Loss: {val_loss:.6f})")
        else:
            early_stop_counter += 1
            print(f"--> No improvement. Counter: {early_stop_counter}/{args.patience}")
            if early_stop_counter >= args.patience:
                print(f"\n[Early Stopping] Training stopped at epoch {epoch}")
                with open(log_file, 'a') as f:
                    f.write(f"\nEarly Stopping triggered at epoch {epoch}\n")
                break

    with open(log_file, 'a') as f:
        f.write("=" * 100 + "\n")
        f.write(f"Total Training Time: {total_training_time:.2f}s\n")
        f.write(f"Best Epoch: {best_epoch} with Validation Loss: {best_val_loss:.6f}\n")

    print(f"\nTraining completed in {total_training_time:.2f}s")
    print(f"Best model was at epoch {best_epoch} with val_loss: {best_val_loss:.6f}")

    print("Evaluating best model on Test Set...")
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pth')))
    test_loss, test_mae, test_mape, test_rmse = evaluate(test_loader)
    print(f"Test Set Results -> Loss: {test_loss:.6f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}")
    with open(log_file, 'a') as f:
        f.write(f"Test Set Results -> Loss: {test_loss:.6f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}\n")

if __name__ == "__main__":
    train()