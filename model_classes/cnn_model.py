import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Conv1d, Dropout, Linear, MaxPool1d, Module, ReLU, Sequential

from data_classes.emovo_dataset import EmovoDataset


class EmovoCNN(Module):
    def __init__(self, waveform_size: int, dropout: float, device: torch.device):
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
        self.conv1 = Sequential(
            Conv1d(in_channels = 2, out_channels = 8, kernel_size = conv1_kernel_size, stride = conv1_stride, device = device),
            BatchNorm1d(num_features = 8),
            ReLU(),
            MaxPool1d(kernel_size = pool1_kernel_size, stride = pool1_stride),
            Dropout(dropout)
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
        self.conv2 = Sequential(
            Conv1d(in_channels = 8, out_channels = 16, kernel_size = conv2_kernel_size, stride = conv2_stride, device = device),
            BatchNorm1d(num_features = 16),
            ReLU(),
            MaxPool1d(kernel_size = pool2_kernel_size, stride = pool2_stride),
            Dropout(dropout)
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
        self.conv3 = Sequential(
            Conv1d(in_channels = 16, out_channels = 32, kernel_size = conv3_kernel_size, stride = conv3_stride, device = device),
            BatchNorm1d(num_features = 32),
            ReLU(),
            MaxPool1d(kernel_size = pool3_kernel_size, stride = pool3_stride),
            Dropout(dropout)
        )

        # Quarto strato convoluzionale
        conv4_kernel_size = 3
        conv4_stride = 2
        self.sample_len = output_size(self.sample_len, 0, conv4_kernel_size, conv4_stride)

        # Quarto strato pooling
        pool4_kernel_size = 3
        pool4_stride = 2
        self.sample_len = output_size(self.sample_len, 0, pool4_kernel_size, pool4_stride)

        # Quarto strato convoluzionale sequenziale
        self.conv4 = Sequential(
            Conv1d(in_channels = 32, out_channels = 64, kernel_size = conv4_kernel_size, stride = conv4_stride, device = device),
            BatchNorm1d(num_features = 64),
            ReLU(),
            MaxPool1d(kernel_size = pool4_kernel_size, stride = pool4_stride),
            Dropout(dropout)
        )

        # Quinto strato convoluzionale
        conv5_kernel_size = 3
        conv5_stride = 2
        self.sample_len = output_size(self.sample_len, 0, conv5_kernel_size, conv5_stride)

        # Quinto strato pooling
        pool5_kernel_size = 3
        pool5_stride = 2
        self.sample_len = output_size(self.sample_len, 0, pool5_kernel_size, pool5_stride)

        # Quinto strato convoluzionale sequenziale
        self.conv5 = Sequential(
            Conv1d(in_channels = 64, out_channels = 128, kernel_size = conv5_kernel_size, stride = conv5_stride, device = device),
            BatchNorm1d(num_features = 128),
            ReLU(),
            MaxPool1d(kernel_size = pool5_kernel_size, stride = pool5_stride),
            Dropout(dropout)
        )

        # Primo strato completamente connesso
        self.fc1 = Sequential(
            # Numero di unità in input: 64 canali * lunghezza del segnale dopo il pooling
            Linear(in_features = 128 * self.sample_len, out_features = 128, device = device),
            ReLU(),
            # Dropout per ridurre l'overfitting
            Dropout(dropout)
        )

        # Secondo strato completamente connesso (output)
        self.fc2 = Linear(in_features = 128, out_features = len(EmovoDataset.LABEL_DICT), device = device)

    def forward(self, x: Tensor):
        # Passaggio attraverso gli strati convoluzionali
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Riformatta l'output per il passaggio attraverso i layer completamente connessi
        x = x.view(-1, 128 * self.sample_len)

        # Passaggio attraverso gli strati completamente connessi
        x = self.fc1(x)
        x = self.fc2(x)

        return x
