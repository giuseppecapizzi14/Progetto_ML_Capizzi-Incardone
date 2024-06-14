from transformers import AutoFeatureExtractor, AutoModel
import torch
from data_classes.emovo_dataset import EmovoDataset

class AudioEmbeddings:
    '''
    This class is intended to extract embeddings from audio models.
    It uses Wav2Vec2 as a default model.
    '''
    
    def __init__(self, model_name='ALM/hubert-base-audioset', device='cuda'):
        
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.device = device
        self.model.to(self.device)
        
        self.model_name = model_name
        
        # eval mode
        self.model.eval()
        
    def extract(self, speech): #passare waveform
        '''
        Extract embeddings from a speech.
        
        Args:
            speech (str): Speech to extract embeddings from.
        
        Returns:
            torch.Tensor: Embeddings of the speech.
        '''
        
        inputs = self.processor(speech, return_tensors="pt", padding="longest", sampling_rate=EmovoDataset.TARGET_SAMPLE_RATE)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()