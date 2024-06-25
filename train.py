import os

import torch.utils.data
from matplotlib import pyplot
from matplotlib.axes import Axes
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import OPTIMIZERS, Config
from data_classes.emovo_dataset import EmovoDataset, Sample
from metrics import Metrics, MetricsHistory, compute_metrics, evaluate, print_metrics
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

    max_lr = config.training.max_lr
    min_lr = config.training.min_lr

    warmup_steps = int(total_steps * config.training.warmup_ratio)

    def lr_warmup_linear_decay(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps

        lr_factor = (total_steps - step) / (total_steps - warmup_steps)

        # Non ancora al minimo learning rate
        next_lr = max_lr * lr_factor
        if next_lr > min_lr:
            return lr_factor

        # Al minimo learning rate, quindi ritorniamo un fattore che moltiplicato al learning rate
        # corrente ci ritorna il learning rate minimo specificato da configurazione, quindi:
        # max_lr * next_min_lr_factor = min_lr -> next_min_rl_factor = min_lr / max_lr
        return min_lr / max_lr

    # Definisce un optimizer con il learning rate specificato
    optimizer = config.training.optimizer
    optimizer = OPTIMIZERS[optimizer]
    optimizer = optimizer(model.parameters(), lr = max_lr)

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

    # Teniamo traccia delle metriche di training e validation
    metrics: MetricsHistory | list[MetricsHistory] | None

    metrics_to_plot = config.plot.metrics
    match metrics_to_plot:
        case None:
            pass
        case str():
            metrics = { "metric": metrics_to_plot, "train": [0], "val": [0] }
        case list():
            metrics = []

            for metric_to_plot in metrics_to_plot:
                metric = { "metric": metric_to_plot, "train": [0], "val": [0] }
                metrics.append(metric)

    # Addestra il modello per il numero di epoche specificate
    evaluation_metric = config.training.evaluation_metric
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        last_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {last_lr:.4f}")

        train_metrics = train_one_epoch(model, train_dl, criterion, scheduler, device)
        val_metrics = evaluate(model, val_dl, criterion, device)

        print_metrics(("Train", train_metrics), ("Val", val_metrics))

        # Teniamo traccia delle metriche da plottare
        match metrics_to_plot:
            case None:
                pass
            case str():
                metric: MetricsHistory = metrics # type: ignore

                train_metric = train_metrics[metrics_to_plot]
                val_metric = val_metrics[metrics_to_plot]

                metric["train"].append(train_metric)
                metric["val"].append(val_metric)
            case list():
                multiple_metrics: list[MetricsHistory] = metrics # type: ignore
                for metric_history in multiple_metrics:
                    metric_to_plot = metric_history["metric"]
                    train_metric = train_metrics[metric_to_plot]
                    val_metric = val_metrics[metric_to_plot]

                    metric_history["train"].append(train_metric)
                    metric_history["val"].append(val_metric)

        # Teniamo traccia della metrica considerata migliore, e salva il modello che massimizza/minimizza
        val_metric = val_metrics[evaluation_metric]
        is_best = val_metric <= best_val_metric if best_metric_lower_is_better else val_metric > best_val_metric
        if is_best:
            print(f"New best model found with val {evaluation_metric}: {val_metric:.4f}")
            best_val_metric = val_metric
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


    match metrics_to_plot:
        case None:
            pass
        case str():
            # Plot delle metriche di training e validation
            epochs_steps = range(0, epochs + 1)

            metric: MetricsHistory = metrics # type: ignore
            metric_to_plot = metric["metric"]

            pyplot.figure(num = "Train and Validation metrics over epochs") # type: ignore
            pyplot.title(f"{metric_to_plot}") # type: ignore

            pyplot.plot(metric["train"], "b-", label = "Train") # type: ignore
            pyplot.plot(metric["val"], "y-", label = "Validation") # type: ignore

            pyplot.xlabel("epochs") # type: ignore
            pyplot.xlim(left = 0) # type: ignore
            pyplot.xticks(epochs_steps) # type: ignore

            pyplot.ylabel(f"{metric_to_plot}") # type: ignore

            pyplot.legend() # type: ignore
            pyplot.grid(True) # type: ignore
        case list():
            # Plot delle metriche di training e validation
            epochs_steps = range(0, epochs + 1)

            multiple_metrics: list[MetricsHistory] = metrics # type: ignore

            metrics_count = len(multiple_metrics)
            figure, plots = pyplot.subplots(metrics_count, num = "Train and Validation metrics over epochs", constrained_layout = True) # type: ignore
            plots: list[Axes]

            for metric, plot in zip(multiple_metrics, plots):
                metric_to_plot = metric["metric"]

                plot.set_title(f"{metric_to_plot}") # type: ignore

                plot.plot(metric["train"], "b-", label = "Train") # type: ignore
                plot.plot(metric["val"], "y-", label = "Validation") # type: ignore

                plot.set_xlim(left = 0)

                plot.set_ylabel(f"{metric_to_plot}") # type: ignore

                plot.legend() # type: ignore
                plot.grid(True) # type: ignore

            last_plot = plots[-1]
            last_plot.set_xticks(epochs_steps)
            last_plot.set_xlabel("epochs") # type: ignore

    pyplot.show() # type: ignore
