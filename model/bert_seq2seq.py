import torch
import torch.nn as nn

from arch.transformer import TransformerEncoderLayer, TransformerEncoder




class BertEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len=1000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.empty(max_len, 1, d_model))
        self.segment_embedding = nn.Embedding(2, d_model)


        self._reset_parameters()
    
    def forward(self, ):
        pass




class BertEncoder(nn.Module):

    def __init__(self, vocab_size, d_model=768, max_len=1000, dropout=0.1, num_encoder_layers=12) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Parameter(torch.empty(max_len, 1, d_model))  # (L, N ,d)
        self.segment_embedding = nn.Embedding(2, d_model)


        self.encoder_layer = TransformerEncoderLayer(d_model, dropout, dim_feedforward=4*d_model, dropout=dropout)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)


        self._reset_parameters()



    def forward(self, tokens, segments, src_mask=None, src_key_padding_mask=None):

        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.positional_embedding[:X.size(0)]
        return self.encoder(X, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
