import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Conv1d, Dropout, Linear, MaxPool1d, Module, ReLU, Sequential, Softmax

from data_classes.emovo_dataset import EmovoDataset


class EmovoCNN(Module):
    def __init__(self, waveform_size: int, dropout: float, device: torch.device):
        def output_size(input_size: int, padding: int, kernel_size: int, stride: int) -> int:
            return (input_size + 2 * padding - kernel_size) // stride + 1

        super(EmovoCNN, self).__init__() # type: ignore

        self.feature_extraction = Sequential(
            # Primo strato convoluzionale
            Conv1d(in_channels = 2, out_channels = 16, kernel_size = 11, stride = 5, padding = 0, device = device),
            BatchNorm1d(num_features = 16, device = device),
            ReLU(inplace = True),
            MaxPool1d(kernel_size = 3, stride = 2, padding = 0),
            Dropout(dropout),

            # Secondo strato convoluzionale
            Conv1d(in_channels = 16, out_channels = 32, kernel_size = 7, stride = 3, padding = 0, device = device),
            BatchNorm1d(num_features = 32, device = device),
            ReLU(inplace = True),
            MaxPool1d(kernel_size = 3, stride = 2, padding = 0),
            Dropout(dropout),

            # Terzo strato convoluzionale
            Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 0, device = device),
            BatchNorm1d(num_features = 64, device = device),
            ReLU(inplace = True),
            MaxPool1d(kernel_size = 3, stride = 2, padding = 0),
            Dropout(dropout),

            # Quarto strato convoluzionale
            Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 0, device = device),
            BatchNorm1d(num_features = 128, device = device),
            ReLU(inplace = True),
            MaxPool1d(kernel_size = 3, stride = 2, padding = 0),
            Dropout(dropout),

            # Quinto strato convoluzionale
            Conv1d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 0, device = device),
            BatchNorm1d(num_features = 256, device = device),
            ReLU(inplace = True),
            MaxPool1d(kernel_size = 3, stride = 2, padding = 0),
            Dropout(dropout)
        )

        # Calcoliamo la dimensione dell'output di tutti gli strati convoluzionali
        self.sample_len = waveform_size
        for layer in self.feature_extraction.modules():
            match layer:
                case Conv1d():
                    padding: int = layer.padding[0] # type: ignore
                    kernel_size = layer.kernel_size[0]
                    stride = layer.stride[0]

                    self.sample_len = output_size(self.sample_len, padding, kernel_size, stride)
                case MaxPool1d():
                    padding: int = layer.padding # type: ignore
                    kernel_size: int = layer.kernel_size # type: ignore
                    stride: int = layer.stride # type: ignore

                    self.sample_len = output_size(self.sample_len, padding, kernel_size, stride)
                case _:
                    pass

        self.classification = Sequential(
            # Primo strato completamente connesso
            Linear(in_features = 256 * self.sample_len, out_features = 128, device = device),
            ReLU(inplace = True),

            # Secondo strato completamente connesso (output)
            Linear(in_features = 128, out_features = len(EmovoDataset.LABEL_DICT), device = device),
            Softmax(1)
        )

    def forward(self, x: Tensor):
        # Passaggio attraverso gli strati convoluzionali
        x = self.feature_extraction(x)

        # Riformatta l'output per il passaggio attraverso i layer completamente connessi
        x = x.flatten(1)

        # Passaggio attraverso gli strati completamente connessi
        x = self.classification(x)

        return x
