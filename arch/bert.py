import torch
import torch.nn as nn

from arch.transformer import TransformerEncoderLayer, TransformerEncoder


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.pos_embedding = nn.Parameter(torch.empty(max_len, 1, d_model))

        self._reset_parameters()

    def forward(self, x, segments):
        """
        Args:
            x: (N, L), where N is batch size, L is sequence length, and the elements range from 0 to vocab_size - 1.
            segments: (N, L), contains only 0s and 1s, where 0 represents the first sentence and 1 represents the second sentence
        """
        x, segments = x.transpose(0, 1), segments.transpose(0, 1)  # (L, N)
        token_embedding = self.token_embedding(x)  # (L, N, d_model)
        segment_embedding = self.segment_embedding(segments)  # (L, N, d_model)
        pos_embedding = self.pos_embedding[:x.size(0)]  # (L, 1, d_model)
        bert_embedding = token_embedding + segment_embedding + pos_embedding  # broadcast
        return bert_embedding  # (L, N, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class BertEncoder(nn.Module):
    def __init__(self, d_model=768, nhead=12, dropout=0.1, num_encoder_layers=12):
        super().__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                     nhead=nhead,
                                                     dim_feedforward=4 * d_model,
                                                     dropout=dropout)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self._reset_parameters()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: (L, N, d_model), where src is bert embedding.
            src_mask: (L, L).
            src_key_padding_mask: (N, L).
        """
        return self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
