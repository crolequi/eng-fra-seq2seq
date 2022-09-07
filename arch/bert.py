import torch
import torch.nn as nn
from arch.transformer import TransformerEncoder, TransformerEncoderLayer
import transformers


class BERTEncoder(nn.Module):

    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, dropout, num_layers, max_len=1000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Parameter(torch.empty(max_len, 1, d_model))  # (L, N, d)
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers)
        


        self._reset_parameters()

    def forward(self,):
        pass
        
        


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

