import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class SentenceTransformer(nn.Module):
    def __init__(self,tokenizer_name='sentence-transformers/all-mpnet-base-v2',model_name='sentence-transformers/all-mpnet-base-v2',device='cpu'):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)


    def forward(self,sentences):
        encoded_input = self.tokenize(sentences).to(self.device)
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def tokenize(self,sentences):
        tokens=self.tokenizer(list(sentences), padding=True, truncation=True, return_tensors='pt')
        return tokens