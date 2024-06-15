import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from config.config import args

from data_classes.emovo_dataset import EmovoDataset
from metrics import evaluate
from model_classes.cnn_model import EmovoCNN

if __name__ == "__main__":
    # Legge in file di configurazione
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
    test_dataset = EmovoDataset(data_dir, train=False, resample=True)

    # Crea il DataLoader
    batch_size = config["training"]["batch_size"]
    test_dl = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    # Crea il modello
    dropout = config["model"]["dropout"]
    model = EmovoCNN(waveform_size = test_dataset.max_sample_len, dropout = dropout, device = device)
    model.to(device)

    # Carica il modello precedentemente salvato in fase di train
    checkpoint_dir = config["training"]["checkpoint_dir"]
    model_name = config['training']['model_name']
    model_state = torch.load(f"{checkpoint_dir}/{model_name}.pt") # type: ignore
    model.load_state_dict(model_state)

    print("Model loaded")

    # Definisce una funzione di loss
    criterion = CrossEntropyLoss()

    print(f"Device: {device}")

    # Valuta le metriche del modello mediante il dataset di test
    test_metrics = evaluate(model, test_dl, criterion, device)
    for key, value in test_metrics.items():
        print(f"Test {key}: {value:.4f}")
