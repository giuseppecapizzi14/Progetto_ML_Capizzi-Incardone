from typing import Literal, TypedDict

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # type: ignore
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_classes.emovo_dataset import Sample

EvaluationMetric = Literal[
    "accuracy",
    "precision",
    "recall",
    "f1",
    "loss",
]

class Metrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1: float
    loss: float

def compute_metrics(references: list[int], predictions: list[int], running_loss: float, dataset_len: int) -> Metrics:
    return {
        "accuracy": accuracy_score(references, predictions),
        "precision": precision_score(references, predictions, average = "macro"),
        "recall": recall_score(references, predictions, average = "macro"),
        "f1": f1_score(references, predictions, average = "macro"),
        "loss": running_loss / dataset_len
    } # type: ignore

def evaluate(
    model: Module,
    dataloader: DataLoader[Sample],
    loss_criterion: CrossEntropyLoss,
    device: torch.device
) -> Metrics:
    model.eval()
    running_loss = 0.0
    predictions: list[int] = []
    references: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc = "Evaluating"):
            waveform: Tensor = batch["waveform"]
            waveform = waveform.to(device)

            labels: Tensor = batch["label"]
            labels = labels.to(device)

            outputs: Tensor = model(waveform)
            loss: Tensor = loss_criterion(outputs, labels)

            running_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)
            predictions.extend(pred.cpu().numpy())
            references.extend(labels.cpu().numpy())

    return compute_metrics(references, predictions, running_loss, len(dataloader))
