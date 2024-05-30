import torch
import torch.nn as nn

class EmovoCNN(nn.Module):
    def __init__(self, waveform_size: int, num_classes: int, dropout: float, device: torch.device):
        def output_size(input_size: int, padding: int, kernel_size: int, stride: int) -> int:
            return (input_size + 2 * padding - kernel_size) // stride + 1

        super(EmovoCNN, self).__init__() # type: ignore

        # Primo strato convoluzionale
        conv1_kernel_size = 10
        conv1_stride = 5
        self.sample_len = output_size(waveform_size, 0, conv1_kernel_size, conv1_stride)

        # Primo strato pooling
        pool1_kernel_size = 3
        pool1_stride = 2
        self.sample_len = output_size(self.sample_len, 0, pool1_kernel_size, pool1_stride)

        # Primo strato convoluzionale sequenziale
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 2, out_channels = 16, kernel_size = conv1_kernel_size, stride = conv1_stride, device = device),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 3, stride = 2)
        )

        # Secondo strato convoluzionale
        conv2_kernel_size = 3
        conv2_stride = 2
        self.sample_len = output_size(self.sample_len, 0, conv2_kernel_size, conv2_stride)

        # Secondo strato pooling
        pool2_kernel_size = 3
        pool2_stride = 2
        self.sample_len = output_size(self.sample_len, 0, pool2_kernel_size, pool2_stride)

        # Secondo strato convoluzionale sequenziale
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = conv2_kernel_size, stride = conv2_stride, device = device),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = pool2_kernel_size, stride = pool2_stride)
        )

        # Terzo strato convoluzionale
        conv3_kernel_size = 3
        conv3_stride = 2
        self.sample_len = output_size(self.sample_len, 0, conv3_kernel_size, conv3_stride)

        # Terzo strato pooling
        pool3_kernel_size = 3
        pool3_stride = 2
        self.sample_len = output_size(self.sample_len, 0, pool3_kernel_size, pool3_stride)

        # Terzo strato convoluzionale sequenziale
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = conv3_kernel_size, stride = conv3_stride, device = device),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = pool3_kernel_size, stride = pool3_stride)
        )

        # Primo strato completamente connesso
        self.fc1 = nn.Sequential(
            # Numero di unità in input: 64 canali * lunghezza del segnale dopo il pooling
            nn.Linear(in_features = 64 * self.sample_len, out_features = 128, device = device),
            nn.ReLU(),
            # Dropout per ridurre l'overfitting
            nn.Dropout(dropout)
        )

        # Secondo strato completamente connesso (output)
        self.fc2 = nn.Linear(in_features = 128, out_features = num_classes, device = device)

    def forward(self, x: torch.Tensor):
        # Passaggio attraverso gli strati convoluzionali
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Riformatta l'output per il passaggio attraverso i layer completamente connessi
        x = x.view(-1, 64 * self.sample_len)

        # Passaggio attraverso gli strati completamente connessi
        x = self.fc1(x)
        x = self.fc2(x)

        return x
