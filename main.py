import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
import wandb

# Input arguments before running the script
parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type = int, default = 2)
parser.add_argument("--series_len", type = int, default = 100)
parser.add_argument("--hidden_size", type = int, default = 128)
parser.add_argument("--num_layers", type = int, default = 1)
parser.add_argument("--epochs", type = int, default = 500)
parser.add_argument("--lr", type = float, default = 0.001)
parser.add_argument("--seed", type = int, default = 42)
parser.add_argument("--train_size", type = float, default = 0.8)
parser.add_argument("--device", 
                    type=torch.device, 
                    default=torch.cuda.is_available() and torch.device("cuda") or torch.device("cpu"))
                    
args = parser.parse_args()


class ModuleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out


def prepare_data(series_len, dataset, class_id):
    X = []
    y = []
    for i in range(len(dataset) - series_len - 1):
        X.append(dataset[["x", "y", "z"]].iloc[i:(i + series_len), 0])
        y.append(class_id)
    return np.array(X), np.array(y)


def accuracy(y_pred, y_true):
    return torch.sum(y_pred == y_true).item() / len(y_true)


def plot_data(train_losses, test_losses, train_accs, test_accs):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(train_losses, label="train")
    ax[0].plot(test_losses, label="test")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(train_accs, label="train")
    ax[1].plot(test_accs, label="test")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    return fig


if __name__ == "__main__":
    # Tracking setup (use wandb=offile on terminal to disable tracking)
    wandb.init(
        project="lstm-har",
        name=f"SL{args.series_len}_HS{args.hidden_size}_NL{args.num_layers}",
        config=args
    )
    wandb.config.update(args)

    # Datasets preparation
    walking_ds = pd.read_csv("data/walking/Accelerometer.csv", parse_dates=True)
    squatting_ds = pd.read_csv("data/squatting/Accelerometer.csv", parse_dates=True)
    X_walking, y_walking = prepare_data(args.series_len, walking_ds, 0)
    X_squatting, y_squatting = prepare_data(args.series_len, squatting_ds, 1)
    X = np.vstack((X_walking, X_squatting))
    y = np.hstack((y_walking, y_squatting))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=args.train_size, shuffle=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Model setup
    model = ModuleLSTM(
        input_size = args.series_len,
        hidden_size = args.hidden_size,
        num_layers = args.num_layers,
        num_classes = args.num_classes
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    X_train = torch.Tensor(X_train_scaled)
    y_train = torch.LongTensor(y_train)
    X_test = torch.Tensor(X_test_scaled)
    y_test = torch.LongTensor(y_test)

    # Model training
    for epoch in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_accs.append(accuracy(pred_class, y_train))
        with torch.no_grad():
            output = model(torch.Tensor(X_test))
            loss = criterion(output, torch.LongTensor(y_test))
            pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
            test_losses.append(loss.item())
            test_accs.append(accuracy(pred_class, y_test))
        wandb.log({
            "train/loss": train_losses[-1],
            "train/acc": train_accs[-1],
            "test/loss": test_losses[-1],
            "test/acc": test_accs[-1]
        })
        tqdm.write(
            f"Epoch: {epoch}, train/loss: {train_losses[-1]:.4f}, train/acc: {train_accs[-1]:.4f}, test/loss: {test_losses[-1]:.4f}, test/acc: {test_accs[-1]:.4f}"
        )
        
    # Plotting
    fig = plot_data(train_losses, test_losses, train_accs, test_accs)
    Path("./results").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"./results/SL{args.series_len}_HS{args.hidden_size}_NL{args.num_layers}_LR{args.lr}.png")