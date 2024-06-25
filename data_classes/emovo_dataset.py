import os
from typing import TypedDict

import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.transforms import Resample


class Sample(TypedDict):
    waveform: Tensor
    label: int

class EmovoDataset(Dataset[Sample]):
    LABEL_DICT = {
        "dis": 0,
        "gio": 1,
        "pau": 2,
        "rab": 3,
        "sor": 4,
        "tri": 5,
        "neu": 6,
    }

    EXPECTED_SAMPLE_RATE = 48_000
    TARGET_SAMPLE_RATE = 16_000

    # Ci aspettiamo che gli audio abbiano due canali, cioè siano in audio stereo
    EXPECTED_CHANNELS = 2

    audio_files: list[str]
    labels: list[int]
    resample: bool
    max_sample_len: int

    def __init__(self, data_path: str, resample: bool = True):
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
                if label not in EmovoDataset.LABEL_DICT:
                    continue

                audio_path = os.path.join(dir_path, file_name)
                waveform, _sample_rate = torchaudio.load(audio_path) # type: ignore

                # Registriamo la traccia audio con la dimensione massima
                waveform_sample_len = waveform.shape[1]
                if waveform_sample_len > self.max_sample_len:
                    self.max_sample_len = waveform_sample_len

                # Registriamo il percorso del file audio e la sua etichetta corrispondente
                self.audio_files.append(audio_path)
                self.labels.append(EmovoDataset.LABEL_DICT[label])

        # Arrotondo i sample per eccesso a una durate di n secondi
        misalignment = self.max_sample_len % EmovoDataset.EXPECTED_SAMPLE_RATE
        if misalignment != 0:
            padding = EmovoDataset.EXPECTED_SAMPLE_RATE - misalignment
            self.max_sample_len += padding

        # Porto il max_sample_len ad utilizzare il nuovo sample rate
        if resample:
            SAMPLE_RATE_RATIO = int(EmovoDataset.EXPECTED_SAMPLE_RATE // EmovoDataset.TARGET_SAMPLE_RATE)
            self.max_sample_len //= SAMPLE_RATE_RATIO

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Sample:
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(audio_path) # type: ignore

        # Resampling
        if self.resample and sample_rate != EmovoDataset.TARGET_SAMPLE_RATE:
            resampler = Resample(orig_freq = sample_rate, new_freq = EmovoDataset.TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = EmovoDataset.TARGET_SAMPLE_RATE

        # Controlliamo che le traccie siano tutte in modalità stereo (canale destro e sinistro)
        current_waveform_channels = waveform.shape[0]
        if current_waveform_channels != EmovoDataset.EXPECTED_CHANNELS:
            # Se le traccie non sono in modalità stereo duplichiamo il singolo canale esistente
            waveform = waveform.repeat(2, 1)

        # Uniforma la lunghezza di tutti gli audio alla lunghezza massima, aggiungendo padding (silenzio)
        waveform_sample_len = waveform.shape[1]
        if waveform_sample_len < self.max_sample_len:
            padding = self.max_sample_len - waveform_sample_len - 1
            waveform = F.pad(input = waveform, pad = (1, padding), mode = "constant")

        return {
            "waveform": waveform,
            "label": label
        }
