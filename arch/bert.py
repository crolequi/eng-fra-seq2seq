import torch
import torch.nn as nn
from arch.transformer import TransformerEncoder, TransformerEncoderLayer


class BertEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        # Three kinds of embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.pos_embedding = nn.Parameter(torch.empty(max_len, 1, d_model))  # (L, N, d)

    def forward(self, src, segments):
        """
        Args:
            src: (L, N, d_model)
            segments: _description_
        """
        token_embedding = self.token_embedding(src)
        if segments is not None:
            segment_embedding = self.segment_embedding(src)
        pos_embedding = self.pos_embedding[:src.size(0)]



class BertEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=12, dropout=0.1, max_len=1000):
        super().__init__()
        # Three kinds of embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.pos_embedding = nn.Parameter(torch.empty(max_len, 1, d_model))  # (L, N, d)

        # Encoder
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                     nhead=nhead,
                                                     dim_feedforward=4 * d_model,
                                                     dropout=dropout)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Initialize
        self._reset_parameters()

    def forward(self, src, segments=None, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: (L, N)
            segments: (L, N)
            src_mask: (L, L)
            src_key_padding_mask: (N, L)
        """
        token_embedding = self.token_embedding(src)
        if segments is not None:
            segment_embedding = self.segment_embedding(src)
        pos_embedding = self.pos_embedding[:src.size(0)]
        bert_embedding = 

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)