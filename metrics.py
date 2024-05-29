import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(predictions, references):

    acc = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, average="macro")
    recall = recall_score(references, predictions, average="macro")
    f1 = f1_score(references, predictions, average="macro")
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in dataloader:
            waveform = batch["waveform"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(waveform)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            pred = torch.argmax(outputs, dim=1)
            predictions.extend(pred.cpu().numpy())
            references.extend(labels.cpu().numpy())
            
    val_metrics = compute_metrics(predictions, references)
    val_metrics["loss"] = running_loss / len(dataloader)
    
    return val_metrics
