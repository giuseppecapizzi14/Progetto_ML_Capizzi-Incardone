from data_classes.emovo_dataset import EmovoDataset
from extract_representetion.audio_embeddings import AudioEmbeddings
import torch
from torch.utils.data import DataLoader
from yaml_config_override import add_arguments # type: ignore
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def process_waveform(waveform):
    # Converte in mono tutti gli audio che non lo
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)
    return waveform.numpy()

def extract_embeddings_and_labels(dataloader, embeddings_extractor):
    embeddings_list = []
    labels_list = []

    for sample in dataloader:
        waveform = sample["waveform"]
        label = sample["label"]
        
        # Itera in ogni sample del batch
        for i in range(waveform.size(0)): 
            waveform_i = waveform[i]
            label_i = label[i].item()
            
            waveform_i = process_waveform(waveform_i)
            embeddings = embeddings_extractor.extract(waveform_i)
            
            embeddings_list.append(embeddings)
            labels_list.append(label_i)

    embeddings_array = np.vstack(embeddings_list)
    labels_array = np.array(labels_list)

    return embeddings_array, labels_array

if __name__ == "__main__":
    # Legge il file di configurazione
    config = add_arguments()

    # Carica il Dataset
    data_dir = config["data"]["data_dir"]
    train_dataset = EmovoDataset(data_dir, train=True, resample=True)
    test_dataset = EmovoDataset(data_dir, train=False, resample=True)

    # Carica il device da utilizzare tra CUDA, MPS e CPU
    if config["training"]["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config["training"]["device"] == "mps" and torch.backends.mps.is_available():
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
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Crea un'instanza della funzione di embeddings
    embeddings_extractor = AudioEmbeddings(model_name="ALM/hubert-base-audioset", device=device)

    # Estrai embeddings e etichette dai set di train e test
    train_embeddings_array, train_labels_array = extract_embeddings_and_labels(train_dl, embeddings_extractor)
    test_embeddings_array, test_labels_array = extract_embeddings_and_labels(test_dl, embeddings_extractor)

    print(train_embeddings_array.shape)
    print(train_labels_array.shape)
    print(test_embeddings_array.shape)
    print(test_labels_array.shape)

    # Normalizza le embeddings
    scaler = StandardScaler()
    train_embeddings_array = scaler.fit_transform(train_embeddings_array)
    test_embeddings_array = scaler.transform(test_embeddings_array)

    # Crea e addestra il classificatore SVM
    svm_classifier = SVC(kernel='linear', C=1, random_state=42)
    svm_classifier.fit(train_embeddings_array, train_labels_array)

    # Predizioni sul set di test
    test_predictions = svm_classifier.predict(test_embeddings_array)

    # Predizioni sul set di training
    train_predictions = svm_classifier.predict(train_embeddings_array)

    # Calcola le metriche di valutazione sul set di test
    test_accuracy = accuracy_score(test_labels_array, test_predictions)
    test_class_report = classification_report(test_labels_array, test_predictions)
    test_conf_matrix = confusion_matrix(test_labels_array, test_predictions)

    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Classification Report:\n{test_class_report}")
    print(f"Test Confusion Matrix:\n{test_conf_matrix}")

    # Calcola le metriche di valutazione sul set di training
    train_accuracy = accuracy_score(train_labels_array, train_predictions)
    train_class_report = classification_report(train_labels_array, train_predictions)
    train_conf_matrix = confusion_matrix(train_labels_array, train_predictions)

    print(f"Train Accuracy: {train_accuracy}")
    print(f"Train Classification Report:\n{train_class_report}")
    print(f"Train Confusion Matrix:\n{train_conf_matrix}")
