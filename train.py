import os
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from tqdm import tqdm
from metrics import compute_metrics, evaluate
from yaml_config_override import add_arguments
from addict import Dict
from data_classes.emovo_dataset import EMOVODataset
from model_classes.cnn_model import CNN


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    predictions = []
    references = []
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        waveform = batch[0].to(device)
        labels = batch[2].to(device)
        
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
    train_metrics['loss'] = running_loss / len(dataloader)
    
    return train_metrics

def manage_best_model_and_metrics(model, evaluation_metric, val_metrics, best_val_metric, best_model, lower_is_better):
    if lower_is_better:
        is_best = val_metrics[evaluation_metric] < best_val_metric
    else:
        is_best = val_metrics[evaluation_metric] > best_val_metric
        
    if is_best:
        print(f"New best model found with val {evaluation_metric}: {val_metrics[evaluation_metric]:.4f}")
        best_val_metric = val_metrics[evaluation_metric]
        best_model = model
        
    return best_val_metric, best_model


if __name__ == '__main__':

    # Legge il file di configurazione
    config = Dict(add_arguments())

    # Caricamento Dataset
    train_dataset = EMOVODataset(config.data.data_dir, train=True, resample=True)
    test_dataset = EMOVODataset(config.data.data_dir, train=True, resample=True)

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


    # Carica il modello
    model = CNN(num_classes= len(EMOVODataset.LABEL_DICT, config.model.dropout))

    # Carica il device da utilizzare tra CUDA, MPS e CPU
    if config.training.devide == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")

    elif config.training.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        print("Device not available. Using CPU!")
        device = torch.device("cpu")

    model.to(device)


    # Definisce una funzione di loss
    criterion = nn.CrossEntropyLoss()

    # Definisce un optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    



