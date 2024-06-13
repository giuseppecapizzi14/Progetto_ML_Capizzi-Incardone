from typing import TypedDict

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # type: ignore
from torch.nn import Module
from torch.nn.modules.loss import _WeightedLoss  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_classes.emovo_dataset import Sample


class Metrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1: float
    loss: float

def compute_metrics(references: list[int], predictions: list[int], running_loss: float, dataset_len: int) -> Metrics:
    accuracy = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, average="macro") # type: ignore
    recall = recall_score(references, predictions, average="macro") # type: ignore
    f1 = f1_score(references, predictions, average="macro") # type: ignore

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": running_loss / dataset_len
    } # type: ignore

def evaluate(
    model: Module,
    dataloader: DataLoader[Sample],
    criterion: _WeightedLoss,
    device: torch.device
) -> Metrics:
    model.eval()
    running_loss = 0.0
    predictions: list[int] = []
    references: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate"):
            waveform = batch["waveform"].to(device)
            labels = batch["label"].to(device)

            outputs = model(waveform)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)
            predictions.extend(pred.cpu().numpy())
            references.extend(labels.cpu().numpy())

    return compute_metrics(references, predictions, running_loss, len(dataloader))
