import torch
import torch.nn as nn
import torch.functional as F

class CNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float
    ):
        
        super(CNN, self).__init__()

        # Primo strato convoluzionale
        self.conv1 = nn.Conv1d(in_channels = 2, out_channels = 16, kernel_size = 10, stride = 5)

        # Secondo strato convoluzionale
        self.conv2 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2)
    
        # Terzo strato convoluzionale
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 2, stride = 2)

        # Max pooling
        self.pool = nn.MaxPool1d(kernel_size = 2, stride = 2)

        # Primo strato completamente connesso
        self.fc1 = nn.Linear(in_features = 64 * 5600, out_features = 128)
        # Numero di unità in input: 64 * 5600 (64 canali * lunghezza del segnale dopo il pooling)

        # Secondo strato completamente connesso (output)
        self.fc2 = nn.Linear(in_features = 128, out_features = num_classes)

        # Dropout per ridurre l'overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Passaggio attraverso il primo strato convoluzionale seguito da attivazione ReLU e max pooling
        x = self.pool(nn.ReLU(self.conv1(x)))
        # Passaggio attraverso il secondo strato convoluzionale seguito da attivazione ReLU e max pooling
        x = self.pool(nn.ReLU(self.conv2(x)))
        # Passaggio attraverso il terzo strato convoluzionale seguito da attivazione ReLU e max pooling
        x = self.pool(nn.ReLU(self.conv3(x)))
        # Riformatta l'output per il passaggio attraverso i layer completamente connessi
        x = x.view(-1, 64 * 4000)
        # Passaggio attraverso il primo strato completamente connesso seguito da attivazione ReLU e dropout
        x = nn.ReLU(self.fc1(x))
        x = self.dropout(x)
        # Passaggio attraverso il secondo strato completamente connesso (output)
        x = self.fc2(x)
        return x


