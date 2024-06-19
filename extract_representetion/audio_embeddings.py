from typing import Any

import numpy
import torch
from numpy import int64
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, PreTrainedModel, Wav2Vec2FeatureExtractor
from transformers.modeling_outputs import BaseModelOutput

from data_classes.emovo_dataset import EmovoDataset, Sample


class AudioEmbeddings:
    """
    Quessta classe ha il compito di estrarre gli embeddings dai modelli audio.
    Usa il modello Wav2Vec2 come default.
    """

    def __init__(self, device: torch.device, model_name: str = "ALM/hubert-base-audioset"):
        self.processor: Wav2Vec2FeatureExtractor = AutoFeatureExtractor.from_pretrained(model_name) # type: ignore
        self.model: PreTrainedModel = AutoModel.from_pretrained(model_name) # type: ignore

        self.device = device
        self.model.to(self.device) # type: ignore

        self.model_name = model_name

        # eval mode
        self.model.eval() # type: ignore

    def extract_embeddings_and_labels(self, dataloader: DataLoader[Sample]) -> tuple[NDArray[Any], NDArray[int64]]:
        batch_count = len(dataloader)
        embeddings_list: list[Tensor] = []
        labels_list: list[int] = []

        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}/{batch_count}")

            labels: Tensor = batch["label"]
            labels_list.extend(labels.tolist()) # type: ignore

            waveforms: Tensor = batch["waveform"]
            for waveform in tqdm(waveforms, desc = "Extracting"):
                # Converte in mono l'audio se non lo è
                if waveform.ndim > 1 and waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim = 0)

                features = self.processor(waveform.numpy(), return_tensors = "pt", sampling_rate = EmovoDataset.TARGET_SAMPLE_RATE)
                input_values: Tensor = features.data["input_values"] # type: ignore
                input_values = input_values.to(self.device) # type: ignore

                with torch.no_grad():
                    outputs: BaseModelOutput = self.model(input_values)

                embeddings = outputs.last_hidden_state.mean(dim = 1).cpu().squeeze_()
                embeddings_list.append(embeddings)

            print()

        embeddings_array = numpy.vstack(embeddings_list) # type: ignore
        labels_array = numpy.array(labels_list)

        return embeddings_array, labels_array
