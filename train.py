import os

import torch.utils.data
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import OPTIMIZERS, Config
from data_classes.emovo_dataset import EmovoDataset, Sample
from metrics import Metrics, compute_metrics, evaluate, print_metrics
from model_classes.cnn_model import EmovoCNN


def train_one_epoch(
    model: Module,
    dataloader: DataLoader[Sample],
    loss_criterion: CrossEntropyLoss,
    scheduler: LRScheduler,
    device: torch.device
) -> Metrics:
    model.train()
    running_loss = 0.0
    predictions: list[int] = []
    references: list[int] = []

    for batch in tqdm(dataloader, desc = "Training"):
        waveforms: Tensor = batch["waveform"]
        waveforms = waveforms.to(device)

        labels: Tensor = batch["label"]
        labels = labels.to(device)

        scheduler.optimizer.zero_grad()

        outputs: Tensor = model(waveforms)
        loss: Tensor = loss_criterion(outputs, labels)
        loss.backward() # type: ignore
        scheduler.optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        pred = torch.argmax(outputs, dim = 1)
        predictions.extend(pred.cpu().numpy())
        references.extend(labels.cpu().numpy())

    return compute_metrics(predictions, references, running_loss, len(dataloader))

if __name__ == "__main__":
    # Legge il file di configurazione
    config = Config()

    device = config.training.device

    # Carica il dataset
    dataset = EmovoDataset(config.data.data_dir, resample = True)

    # Calcola le dimensioni dei dataset
    # |------- dataset -------|
    # |---train---|-val-|-test|
    dataset_size = len(dataset)

    train_size = int(config.data.train_ratio * dataset_size)

    test_val_size = dataset_size - train_size
    test_size = int(test_val_size * config.data.test_val_ratio)
    val_size = test_val_size - test_size

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    # Crea i DataLoader
    batch_size = config.training.batch_size
    train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dl = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    val_dl = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # Crea il modello
    model = EmovoCNN(waveform_size = dataset.max_sample_len, dropout = config.model.dropout, device = device)
    model.to(device)

    # Definisce una funzione di loss
    criterion = CrossEntropyLoss()

    # Definisce uno scheduler per il decay del learning rate
    epochs = config.training.epochs
    total_steps = len(train_dl) * epochs

    base_lr = config.training.base_lr
    min_lr = config.training.min_lr

    warmup_steps = int(total_steps * config.training.warmup_ratio)

    def lr_warmup_linear_decay(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps

        lr_factor = (total_steps - step) / (total_steps - warmup_steps)

        # Non ancora al minimo learning rate
        next_lr = base_lr * lr_factor
        if next_lr > min_lr:
            return lr_factor

        # Al minimo learning rate, quindi ritorniamo un fattore che moltiplicato al learning rate
        # corrente ci ritorna il learning rate minimo specificato da configurazione, quindi:
        # base_lr * next_min_lr_factor = min_lr -> next_min_rl_factor = min_lr / base_lr
        return min_lr / base_lr

    # Definisce un optimizer con il learning rate specificato
    optimizer = config.training.optimizer
    optimizer = OPTIMIZERS[optimizer]
    optimizer = optimizer(model.parameters(), lr = base_lr)

    scheduler = LambdaLR(optimizer, lr_lambda = lr_warmup_linear_decay)

    # Stampa le informazioni sul processo di training
    print(f"Device: {device}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print()

    # Teniamo traccia del modello e della metrica migliore
    best_metric_lower_is_better = config.training.best_metric_lower_is_better
    best_val_metric = float("inf") if best_metric_lower_is_better else float("-inf")
    best_model = model

    # Addestra il modello per il numero di epoche specificate
    evaluation_metric = config.training.evaluation_metric
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        last_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {last_lr:.4f}")

        train_metrics = train_one_epoch(model, train_dl, criterion, scheduler, device)
        val_metrics = evaluate(model, val_dl, criterion, device)

        print_metrics(("Train", train_metrics), ("Val", val_metrics))

        metric = val_metrics[evaluation_metric]
        is_best = metric <= best_val_metric if best_metric_lower_is_better else metric > best_val_metric
        if is_best:
            print(f"New best model found with val {evaluation_metric}: {metric:.4f}")
            best_val_metric = metric
            best_model = model

        print()

    # Valuta le metriche del modello mediante il dataset di test
    test_metrics = evaluate(best_model, test_dl, criterion, device)
    for key, value in test_metrics.items():
        print(f"Test {key}: {value:.4f}")

    # Salva il modello
    checkpoint_dir = config.training.checkpoint_dir
    model_name = config.training.model_name
    os.makedirs(checkpoint_dir, exist_ok = True)
    torch.save(best_model.state_dict(), f"{checkpoint_dir}/{model_name}.pt") # type: ignore

    print("Model saved")
