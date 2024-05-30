import torch
import torchaudio
import os
from torch.utils.data import Dataset
import torchaudio.transforms as transforms
import torch.nn.functional as F

class EMOVODataset(Dataset[dict[str, torch.Tensor | int]]):
    LABEL_DICT = {
        "dis": 0,
        "gio": 1,
        "pau": 2,
        "rab": 3,
        "sor": 4,
        "tri": 5,
        "neu": 6,
    }

    TARGET_SAMPLE_RATE = 16_000

    def __init__(self, data_path: str, train: bool = True, resample: bool = True):
        self.data_path = data_path
        self.train = train
        self.audio_files: list[str] = []
        self.labels: list[int] = []
        self.resample = resample
        self.max_sample_len = 0

        # Scansione della directory degli attori
        for dir_path, _dir_names, file_names in os.walk(data_path):
            for file_name in file_names:
                # Controlla se il file ha l'estenzione .wav
                if not file_name.endswith(".wav"):
                    continue

                # Estrae la parte del nome del file che contiene la label poichè i nomi dei file
                # sono nel formato 'dis-f1-b1.wav' con la label key messa al primo posto
                label = file_name.split('-')[0]
                if label not in EMOVODataset.LABEL_DICT:
                    continue

                audio_path = os.path.join(dir_path, file_name)

                # Trova la traccia audio con la dimensione massima
                waveform, _sample_rate = torchaudio.load(audio_path) # type: ignore
                waveform_sample_len = waveform.shape[1]
                if waveform_sample_len > self.max_sample_len:
                    self.max_sample_len = waveform_sample_len

                # Registriamo il percorso del file audio e la sua etichetta corrispondente
                self.audio_files.append(audio_path)
                self.labels.append(EMOVODataset.LABEL_DICT[label])
        # Arrotondo i sample per eccesso a una durate di n secondi
        SAMPLE_RATE = 48000
        misalignment = self.max_sample_len % SAMPLE_RATE
        if misalignment != 0:
            padding = SAMPLE_RATE - misalignment
            self.max_sample_len += padding

        # Porto il max_sample_len ad utilizzare il nuovo sample rate
        if resample:
            SAMPLE_RATE_RATIO = int(SAMPLE_RATE // EMOVODataset.TARGET_SAMPLE_RATE)
            self.max_sample_len //= SAMPLE_RATE_RATIO

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

        waveform, sample_rate = torchaudio.load(audio_path) # type: ignore

        # Uniforma la lunghezza di tutti gli audio alla lunghezza massima, aggiungendo padding (silenzio)
        waveform_sample_len = waveform.shape[1]
        if waveform_sample_len < self.max_sample_len:
            padding = self.max_sample_len - waveform_sample_len - 1
            waveform = F.pad(input = waveform, pad = (1, padding), mode = "constant")

        # Resampling
        if self.resample and sample_rate != EMOVODataset.TARGET_SAMPLE_RATE:
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=EMOVODataset.TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = EMOVODataset.TARGET_SAMPLE_RATE

        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "label": label
        }
