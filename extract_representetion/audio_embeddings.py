import torch
from torch import Tensor
from transformers import AutoFeatureExtractor, AutoModel, PreTrainedModel, Wav2Vec2FeatureExtractor
from transformers.modeling_outputs import BaseModelOutput

from data_classes.emovo_dataset import EmovoDataset


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

    def extract(self, speech: Tensor) -> Tensor:
        # Converte in mono l'audio se non lo è
        if speech.ndim > 1 and speech.shape[0] > 1:
            speech = torch.mean(speech, dim = 0)

        features = self.processor(speech.numpy(), return_tensors = "pt", sampling_rate = EmovoDataset.TARGET_SAMPLE_RATE)
        input_values: Tensor = features.data["input_values"] # type: ignore
        input_values = input_values.to(self.device) # type: ignore

        with torch.no_grad():
            outputs: BaseModelOutput = self.model(input_values)

        return outputs.last_hidden_state.mean(dim = 1).cpu()
