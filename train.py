import os
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from tqdm import tqdm
from metrics import compute_metrics, evaluate
from yaml_config_override import add_arguments
from addict import Dict
from data_classes.emovo_dataset import EmovoDataset
from model_classes.cnn_model import EmovoCNN


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    predictions = []
    references = []

    for _i, batch in enumerate(tqdm(dataloader, desc="Training")):
        waveform = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(waveform)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        pred = torch.argmax(outputs, dim=1)
        predictions.extend(pred.cpu().numpy())
        references.extend(labels.cpu().numpy())

    train_metrics = compute_metrics(predictions, references)
    train_metrics["loss"] = running_loss / len(dataloader)

    return train_metrics

def manage_best_model_and_metrics(model, evaluation_metric, val_metrics, best_val_metric, best_model, lower_is_better):
    if lower_is_better:
        is_best = val_metrics[evaluation_metric] <= best_val_metric
    else:
        is_best = val_metrics[evaluation_metric] > best_val_metric

    if is_best:
        print(f"New best model found with val {evaluation_metric}: {val_metrics[evaluation_metric]:.4f}")
        best_val_metric = val_metrics[evaluation_metric]
        best_model = model

    return best_val_metric, best_model

if __name__ == "__main__":
    # Legge il file di configurazione
    config = Dict(add_arguments())

    # Caricamento Dataset
    train_dataset = EmovoDataset(config.data.data_dir, train=True, resample=True)
    test_dataset = EmovoDataset(config.data.data_dir, train=False, resample=True)

    # Carica il device da utilizzare tra CUDA, MPS e CPU
    if config.training.devide == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.training.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Carica il modello
    model = EmovoCNN(waveform_size = train_dataset.max_sample_len, num_classes = len(EmovoDataset.LABEL_DICT), dropout = config.model.dropout, device = device)

    # Calcola le dimensioni del Set di Train e del Set di Validation
    train_size = int(config.data.train_ratio * len(train_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])

    # Crea il DataLoader di Train
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )

    # Crea il DataLoader di Validation
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )

    # Crea il DataLoader di Test
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )

    # Stampa le dimensioni dei vari set
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print()

    # Definisce una funzione di loss
    criterion = nn.CrossEntropyLoss()

    # Definisce un optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    # learning rate scheduler
    total_steps = len(train_dl) * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    # warmup + linear decay
    scheduler_lambda = lambda step: (step / warmup_steps) if step < warmup_steps else max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)

    if config.training.best_metric_lower_is_better:
        best_val_metric = float("inf")
    else:
        best_val_metric = float("-inf")

    best_model = model

    # Addestra il modello
    for epoch in range(config.training.epochs):
        print(f"Epoch {epoch+1}/{config.training.epochs}")

        train_metrics = train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_dl, criterion, device)

        print(f"Train loss: {train_metrics['loss']:.4f} - Train accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val loss: {val_metrics['loss']:.4f} - Val accuracy: {val_metrics['accuracy']:.4f}")

        best_val_metric, best_model = manage_best_model_and_metrics(
            model,
            config.training.evaluation_metric,
            val_metrics,
            best_val_metric,
            best_model,
            config.training.best_metric_lower_is_better
        )
        print()

    # Valuta le metriche del modello
    test_metrics = evaluate(best_model, test_dl, criterion, device)
    for key, value in test_metrics.items():
        print(f"Test {key}: {value:.4f}")

    # Non salviamo per il momento visto che non è implementata la funzione di caricamento
    # # Salva il modello
    # os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    # torch.save(best_model.state_dict(), f"{config.training.checkpoint_dir}/best_model.pt")

    # print("Model saved.")
