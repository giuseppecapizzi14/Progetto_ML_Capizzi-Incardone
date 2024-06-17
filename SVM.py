from typing import Any

import numpy as np
import torch
import torch.utils.data
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import Tensor
from torch.types import Number
from torch.utils.data import DataLoader

from config.config import args
from data_classes.emovo_dataset import EmovoDataset, Sample
from extract_representetion.audio_embeddings import AudioEmbeddings


def extract_embeddings_and_labels(
    dataloader: DataLoader[Sample],
    embeddings_extractor: AudioEmbeddings
) -> tuple[NDArray[Any], NDArray[Any]]:
    embeddings_list: list[Tensor] = []
    labels_list: list[Number] = []

    for sample in dataloader:
        waveforms: Tensor = sample["waveform"]
        labels: Tensor = sample["label"]

        for waveform, label in zip(waveforms, labels):
            embeddings = embeddings_extractor.extract(waveform)
            # TODO(stefano): controllare se è necessario fare questo squeeze
            embeddings.squeeze_()

            embeddings_list.append(embeddings)
            labels_list.append(label.item())

    embeddings_array = np.vstack(embeddings_list) # type: ignore
    labels_array = np.array(labels_list)

    return embeddings_array, labels_array

if __name__ == "__main__":
    # Legge il file di configurazione
    config = args()

    # Carica il Dataset
    data_dir = config["data"]["data_dir"]
    train_dataset = EmovoDataset(data_dir, train = True, resample = True)
    test_dataset = EmovoDataset(data_dir, train = False, resample = True)

    # Carica il device da utilizzare tra CUDA, MPS e CPU
    device = config["training"]["device"]
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Calcola le dimensioni del Set di Train e del Set di Validation
    train_ratio = config["data"]["train_ratio"]
    train_size = int(train_ratio * len(train_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])

    # Crea i DataLoader
    batch_size = config["training"]["batch_size"]
    train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dl = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_dl = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    # Crea un'instanza della funzione di embeddings
    embeddings_extractor = AudioEmbeddings(device=device)

    # Estrai embeddings e etichette dai set di train e test
    train_embeddings, train_labels = extract_embeddings_and_labels(train_dl, embeddings_extractor)
    test_embeddings, test_labels = extract_embeddings_and_labels(test_dl, embeddings_extractor)

    print(train_embeddings.shape)
    print(train_labels.shape)
    print(test_embeddings.shape)
    print(test_labels.shape)

    # Normalizza le embeddings
    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings) # type: ignore
    test_embeddings = scaler.transform(test_embeddings) # type: ignore

    # Crea e addestra il classificatore SVM
    svm_classifier = SVC(kernel = "linear", C = 1, random_state = 42)
    svm_classifier.fit(train_embeddings, train_labels) # type: ignore

    # Predizioni sul set di training
    train_predictions = svm_classifier.predict(train_embeddings) # type: ignore

    # Calcola le metriche di valutazione sul set di training
    train_accuracy = accuracy_score(train_labels, train_predictions)
    train_class_report = classification_report(train_labels, train_predictions) # type: ignore
    train_conf_matrix = confusion_matrix(train_labels, train_predictions) # type: ignore

    print(f"Train Accuracy: {train_accuracy}")
    print(f"Train Classification Report:\n{train_class_report}")
    print(f"Train Confusion Matrix:\n{train_conf_matrix}")

    # Predizioni sul set di test
    test_predictions = svm_classifier.predict(test_embeddings) # type: ignore

    # Calcola le metriche di valutazione sul set di test
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_class_report = classification_report(test_labels, test_predictions) # type: ignore
    test_conf_matrix = confusion_matrix(test_labels, test_predictions) # type: ignore

    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Classification Report:\n{test_class_report}")
    print(f"Test Confusion Matrix:\n{test_conf_matrix}")
