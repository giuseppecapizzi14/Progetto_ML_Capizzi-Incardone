import os

import torch.utils.data
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adadelta, Adagrad, Adam, Adamax, AdamW, ASGD, LBFGS, NAdam, Optimizer, RAdam, RMSprop, Rprop, SGD, SparseAdam
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import args
from data_classes.emovo_dataset import EmovoDataset, Sample
from metrics import Metrics, compute_metrics, evaluate
from model_classes.cnn_model import EmovoCNN


def train_one_epoch(
    model: Module,
    dataloader: DataLoader[Sample],
    loss_criterion: CrossEntropyLoss,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device
) -> Metrics:
    model.train()
    running_loss = 0.0
    predictions: list[int] = []
    references: list[int] = []

    for batch in tqdm(dataloader, desc="Training"):
        waveform = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(waveform)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        pred = torch.argmax(outputs, dim=1)
        predictions.extend(pred.cpu().numpy())
        references.extend(labels.cpu().numpy())

    return compute_metrics(predictions, references, running_loss, len(dataloader))

def manage_best_model_and_metrics(
    model: Module,
    evaluation_metric: str,
    val_metrics: Metrics,
    best_val_metric: float,
    best_model: Module,
    lower_is_better: bool
) -> tuple[float, Module]:
    metric = val_metrics[evaluation_metric] # type: ignore

    if lower_is_better:
        is_best = metric <= best_val_metric # type: ignore
    else:
        is_best = metric > best_val_metric # type: ignore

    if is_best:
        print(f"New best model found with val {evaluation_metric}: {metric:.4f}")
        best_val_metric = metric # type: ignore
        best_model = model

    return best_val_metric, best_model # type: ignore


if __name__ == "__main__":
    # Legge il file di configurazione
    config = args()

    # Carica il device da utilizzare tra CUDA, MPS e CPU
    device = config["training"]["device"]
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Caricamento Dataset
    data_dir = config["data"]["data_dir"]
    train_dataset = EmovoDataset(data_dir, train=True, resample=True)
    test_dataset = EmovoDataset(data_dir, train=False, resample=True)

    # Crea il modello
    dropout = config["model"]["dropout"]
    model = EmovoCNN(waveform_size = train_dataset.max_sample_len, dropout = dropout, device = device)
    model.to(device)

    # Calcola le dimensioni del Set di Train e del Set di Validation
    train_ratio = config["data"]["train_ratio"]
    train_size = int(train_ratio * len(train_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])

    # Crea i DataLoader
    batch_size = config["training"]["batch_size"]
    train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dl = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_dl = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    # Definisce una funzione di loss
    criterion = CrossEntropyLoss()

    # Definisce un optimizer
    OPTIMIZERS = {
        "adadelta": Adadelta,
        "adagrad": Adagrad,
        "adamax": Adamax,
        "adamw": AdamW,
        "asgd": ASGD,
        "lbfgs": LBFGS,
        "nadam": NAdam,
        "radam": RAdam,
        "rmsprop": RMSprop,
        "rprop": Rprop,
        "sgd": SGD,
        "sparse_adam": SparseAdam,
    }

    optimizer = config["training"]["optimizer"]
    optimizer = OPTIMIZERS.get(optimizer, Adam) # Adam come default

    lr = config["training"]["lr"]
    optimizer = optimizer(model.parameters(), lr = lr)

    # Definisce uno scheduler
    epochs = config["training"]["epochs"]
    total_steps = len(train_dl) * epochs

    warmup_ratio = config["training"]["warmup_ratio"]
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_warmup_linear_decay(step: int):
        return (step / warmup_steps) if step < warmup_steps else max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_warmup_linear_decay)

    # Modello migliore
    best_metric_lower_is_better = config["training"]["best_metric_lower_is_better"]
    best_val_metric = float("inf") if best_metric_lower_is_better else float("-inf")
    best_model = model

    # Stampa le informazioni sul processo di training
    print(f"Device: {device}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print()

    # Addestra il modello
    evaluation_metric = config["training"]["evaluation_metric"]
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        train_metrics = train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_dl, criterion, device)

        train_loss = train_metrics["loss"]
        train_accuracy = train_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        val_accuracy = val_metrics["accuracy"]

        print(f"Train loss: {train_loss:.4f} - Train accuracy: {train_accuracy:.4f}")
        print(f"Val loss: {val_loss:.4f} - Val accuracy: {val_accuracy:.4f}")

        best_val_metric, best_model = manage_best_model_and_metrics(
            model,
            evaluation_metric,
            val_metrics,
            best_val_metric,
            best_model,
            best_metric_lower_is_better
        )
        print()

    # Valuta le metriche del modello
    test_metrics = evaluate(best_model, test_dl, criterion, device)
    for key, value in test_metrics.items():
        print(f"Test {key}: {value:.4f}")

    # Salva il modello
    checkpoint_dir = config["training"]["checkpoint_dir"]
    model_name = config['training']['model_name']
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(best_model.state_dict(), f"{checkpoint_dir}/{model_name}.pt") # type: ignore

    print("Model saved")
