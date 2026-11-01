import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import prepare_datasets
import model  
from train import evaluate_epoch, train_epoch, training_statistics

train_ds, val_ds,_ = prepare_datasets()
mu, sigma = training_statistics(train_ds)
device = "cuda" if torch.cuda.is_available() else "cpu"


def objective(trial: optuna.Trial) -> float:
    
    model.HIDDEN1 = trial.suggest_categorical("hidden1", [256, 512])
    model.HIDDEN2 = trial.suggest_categorical("hidden2", [128, 256])
    model.DROPOUT1 = trial.suggest_float("dropout1", 0.05, 0.25, step=0.05)
    model.DROPOUT2 = trial.suggest_float("dropout2", 0.0, 0.25, step=0.05)
    
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("Batch_size",[32,64,128])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


    m = model.ConsumptionPredictor(mu=mu, sigma=sigma).to(device)

    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    for epoch in range(1, 30): #For param tuning we see fast what works best
        train_epoch(m, train_loader, optimizer, loss_fn, device)
        val_mae = evaluate_epoch(m, val_loader, loss_fn, device)
        if val_mae < best_val:
            best_val = val_mae

    return best_val


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters found :", study.best_params)