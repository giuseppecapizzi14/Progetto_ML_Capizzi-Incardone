import torch
import torch.nn as nn
import torch.functional as F

class CNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layers: list,
        num_classes: int,
        dropout: float = 0.2
    ):
        
        super(CNN, self).__init__()

        # self.conv1=nn.Sequential(
        #     nn.Conv1d(1, 16,kernel_size=3,stride=1,padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2)
        # )
        # self.conv2=nn.Sequential(
        #     nn.Conv1d(16, 32,kernel_size=3,stride=1,padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2)
        # )
        # self.conv3=nn.Sequential(
        #     nn.Conv1d(32, 64,kernel_size=3,stride=1,padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2)
        # )

        
        # Primo strato convoluzionale
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        # Numero di canali in input: 1 (un canale per le onde sonore)
        # Numero di filtri (feature maps) in uscita: 16
        # Dimensione del kernel: 3 (larghezza del filtro)
        # Stride: 1 (il filtro si sposta di una posizione alla volta)
        # Padding: 1 (aggiunge un padding di 1 ai bordi per mantenere la stessa dimensione dell'input)

        # Secondo strato convoluzionale
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        # Numero di canali in input: 16 (output del primo strato)
        # Numero di filtri (feature maps) in uscita: 32
        # Dimensione del kernel: 3
        # Stride: 1
        # Padding: 1

        # Terzo strato convoluzionale
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        # Numero di canali in input: 32 (output del secondo strato)
        # Numero di filtri (feature maps) in uscita: 64
        # Dimensione del kernel: 3
        # Stride: 1
        # Padding: 1

        # Max pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # Dimensione della finestra di pooling: 2 (riduce la lunghezza dell'output di un fattore di 2)
        # Stride: 2 (il pooling si sposta di 2 posizioni alla volta)
        # Padding: 0 (nessun padding)

        # Primo strato completamente connesso
        self.fc1 = nn.Linear(64 * 4000, 128)
        # Numero di unità in input: 64 * 4000 (64 canali * lunghezza del segnale dopo il pooling)
        # Numero di unità in output: 128

        # Secondo strato completamente connesso (output)
        self.fc2 = nn.Linear(128, num_classes)
        # Numero di unità in input: 128
        # Numero di unità in output: num_classes (numero di classi di output)

        # Dropout per ridurre l'overfitting
        self.dropout = nn.Dropout(0.5)
        # Probabilità di dropout: 0.5 (50% di dropout durante l'addestramento)

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

