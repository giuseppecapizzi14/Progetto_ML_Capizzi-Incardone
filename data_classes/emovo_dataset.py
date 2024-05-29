import torch
import torchaudio
import os
from torch.utils.data import Dataset
import torchaudio.transforms as transforms
import torch.nn.functional as F

class EMOVODataset(Dataset[dict[str, torch.Tensor | int]]):
    LABEL_DICT = {
        'dis': 0,
        'gio': 1,
        'pau': 2,
        'rab': 3,
        'sor': 4,
        'tri': 5,
        'neu': 6
    }

    TARGET_SAMPLE_RATE = 16_000

    def __init__(self, data_path: str, train: bool = True, resample: bool = True):
        self.data_path = data_path
        self.train = train
        self.download = True
        self.audio_files: list[str] = []
        self.labels: list[int] = []
        self.resample = resample
        self.max_sample_len = 0

        # Scansione della directory degli attori
        for actor_dir in os.listdir(data_path):
            actor_path = os.path.join(data_path, actor_dir)
            if os.path.isdir(actor_path):
                for file_name in os.listdir(actor_path):
                    # Controlla se il file ha l'estenzione .wav
                    if file_name.endswith('.wav'):
                        # Estrae l'etichetta dal nome del file usando la funzione extract_label
                        label = self.extract_label(file_name)
                        if label in EMOVODataset.LABEL_DICT:
                            audio_path = os.path.join(actor_path, file_name)

                            # Trova la traccia audio con la dimensione massima
                            waveform, _sample_rate = torchaudio.load(audio_path) # type: ignore
                            waveform_sample_len = waveform.shape[1]
                            if waveform_sample_len > self.max_sample_len:
                                self.max_sample_len = waveform_sample_len

                            # Aggiunge il percorso completo del file audio alla lista self.audio_files
                            self.audio_files.append(audio_path)
                            # Aggiunge l'etichetta numerica corrispondente alla lista self.labels
                            self.labels.append(EMOVODataset.LABEL_DICT[label])

        SAMPLE_RATE = 48000
        misalignment = self.max_sample_len % SAMPLE_RATE
        if misalignment != 0:
            padding = SAMPLE_RATE - misalignment
            self.max_sample_len += padding

        self.max_sample_len -= 1

    def extract_label(self, file_name: str) -> str:
        """ Estrae la parte del nome del file che contiene la label poichè i nomi dei file sono in questo formato 'dis-f1-b1.wav'
        con la label key messa al primo posto """

        label = file_name.split('-')[0]
        return label

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        """
        Returns:
            dict:
                "waveform" (torch.Tensor): un sample del dataset
                "sample_rate" (int): il target sample rate utilizzato nel modello
                "label" (int): una label del dataset
        """

        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Carica il file audio
        waveform, sample_rate = torchaudio.load(audio_path) # type: ignore

        # Resampling
        if self.resample and sample_rate != EMOVODataset.TARGET_SAMPLE_RATE:
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=EMOVODataset.TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = EMOVODataset.TARGET_SAMPLE_RATE

        # Uniforma la lunghezza di tutti gli audio alla lunghezza massima, aggiungendo padding (silenzio)
        waveform_sample_len = waveform.shape[1]
        if waveform_sample_len < self.max_sample_len:
            padding = self.max_sample_len - waveform_sample_len
            waveform = F.pad(waveform, (1, padding))

        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "label": label
        }
