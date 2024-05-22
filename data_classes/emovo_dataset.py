import torch
import torchaudio
import os
from torch.utils.data import Dataset
import torchaudio.transforms as transforms

class EMOVODataset(Dataset[tuple[torch.Tensor, float, int]]):
    def __init__(self, data_path: str, label_dict: dict[str, int], train: bool = True):
        self.data_path = data_path
        self.train = train
        self.download = True
        self.label_dict = label_dict
        self.audio_files: list[str] = []
        self.labels: list[int] = []
        self.spectrogram_transform = transforms.MelSpectrogram()


        # Scansione della directory degli attori
        for actor_dir in os.listdir(data_path):
            actor_path = os.path.join(data_path, actor_dir)
            if os.path.isdir(actor_path):
                for file_name in os.listdir(actor_path):
                    # Controlla se il file ha l'estenzione .wav
                    if file_name.endswith('.wav'):
                        # Estrae l'etichetta dal nome del file usando la funzione extract_label
                        label = self.extract_label(file_name)
                        if label in label_dict:
                            # Aggiunge il percorso completo del file audio alla lista self.audio_files
                            self.audio_files.append(os.path.join(actor_path, file_name))
                            # Aggiunge l'etichetta numerica corrispondente alla lista self.labels
                            self.labels.append(label_dict[label])

    def extract_label(self, file_name: str) -> str:
        ''' Estrae la parte del nome del file che contiene la label poichè i nomi dei file sono in questo formato 'dis-f1-b1.wav'
        con la label key messa al primo posto '''
        label = file_name.split('-')[0]
        return label

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Carica il file audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Genera lo spettrogramma
        spectrogram = self.spectrogram_transform(waveform)

        return spectrogram, sample_rate, label
