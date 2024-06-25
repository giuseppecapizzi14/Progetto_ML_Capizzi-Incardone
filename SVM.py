import torch
import torch.utils.data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from config.config import Config
from data_classes.emovo_dataset import EmovoDataset
from extract_representetion.audio_embeddings import AudioEmbeddings

if __name__ == "__main__":
    # Legge il file di configurazione
    config = Config()

    device = config.training.device

    # Carica il dataset
    dataset = EmovoDataset(config.data.data_dir, resample = True)

    # Calcola le dimensioni dei dataset
    # |------- dataset -------|
    # |---train---|---test----|
    dataset_size = len(dataset)
    train_size = int(config.data.train_ratio * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Crea i DataLoader
    batch_size = config.training.batch_size
    train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dl = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    # Stampa le informazioni sul processo di training
    print(f"Device: {device}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print()

    # Crea un'instanza della funzione di embeddings
    embeddings_extractor = AudioEmbeddings(device = device)

    # Estrai embeddings e etichette dai set di train e test
    train_embeddings, train_labels = embeddings_extractor.extract_embeddings_and_labels(train_dl)
    test_embeddings, test_labels = embeddings_extractor.extract_embeddings_and_labels(test_dl)

    # Normalizza le embeddings
    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings) # type: ignore
    test_embeddings = scaler.transform(test_embeddings) # type: ignore

    # Crea e addestra il classificatore SVM
    svm_classifier = SVC(kernel = "linear", C = 1, random_state = 42)
    svm_classifier.fit(train_embeddings, train_labels) # type: ignore

    # Predizioni sul set di test
    test_predictions = svm_classifier.predict(test_embeddings) # type: ignore

    # Calcola le metriche di valutazione sul set di test
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_class_report = classification_report(test_labels, test_predictions) # type: ignore
    test_conf_matrix = confusion_matrix(test_labels, test_predictions) # type: ignore

    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Classification Report:\n{test_class_report}")
    print(f"Test Confusion Matrix:\n{test_conf_matrix}")
