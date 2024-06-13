import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from metrics import evaluate
from yaml_config_override import add_arguments
from data_classes.emovo_dataset import EmovoDataset
from model_classes.cnn_model import EmovoCNN

if __name__ == '__main__':
    # Load configuration
    config = add_arguments()

    # Carica il device da utilizzare tra CUDA, MPS e CPU
    if config["training"]["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config["training"]["device"] == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Load data
    test_dataset = EmovoDataset(config["data"]["data_dir"], train=False, resample=True)

    # Crea il DataLoader di Test
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    # Carica il modello
    model = EmovoCNN(waveform_size = test_dataset.max_sample_len, dropout = config["model"]["dropout"], device = device)
    model.to(device)

    # Load model weights
    model_state = torch.load(f"{config['training']['checkpoint_dir']}/{config['training']['model_name']}.pt")
    model.load_state_dict(model_state)
    print("Model loaded.")

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    test_metrics = evaluate(model, test_dl, criterion, device)
    for key, value in test_metrics.items():
        print(f"Test {key}: {value:.4f}")
