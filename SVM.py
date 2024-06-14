from data_classes.emovo_dataset import EmovoDataset
from extract_representetion.audio_embeddings import AudioEmbeddings
import torch
from torch.utils.data import DataLoader
from yaml_config_override import add_arguments # type: ignore
import numpy as np


if __name__ == "__main__":
    # Legge il file di configurazione
    config = add_arguments()

    # Carica il dataset
    dataset = EmovoDataset(config["data"]["data_dir"], train=True, resample=True)

    # Carica il device da utilizzare tra CUDA, MPS e CPU
    if config["training"]["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config["training"]["device"] == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Crea il DataLoader
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    # Crea un'instanza della funzione di embeddings
    embeddings_extractor = AudioEmbeddings(model_name='ALM/hubert-base-audioset', device=device)

    # Iterazione sul dataset per estrarre le embeddings
    embeddings_list = []
    labels_list = []

    for sample in dataloader:
        waveform = sample['waveform'].squeeze(0)
        # Convert to mono if the waveform is not mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ensure waveform is 2D: (batch, length)
        waveform = waveform.squeeze(0).numpy()
        embeddings = embeddings_extractor.extract(waveform)
        embeddings_list.append(embeddings)
        labels_list.extend(sample['label'].squeeze().tolist())

    # Converti in array numpy
    embeddings_array = np.vstack(embeddings_list)
    labels_array = np.array(labels_list)

    print(embeddings_array.shape)
    print(labels_array.shape)
