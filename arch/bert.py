import torch
import torch.nn as nn
from arch.transformer import TransformerEncoder, TransformerEncoderLayer
import transformers


class BERTEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model=768,
                 nhead=12,
                 dim_feedforward=3072,
                 dropout=0.1,
                 num_encoder_layers=12,
                 max_len=1000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Parameter(torch.empty(max_len, 1, d_model))  # (L, N, d)
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers)

        self._reset_parameters()

    def forward(self, tokens, segments, src_mask=None, src_key_padding_mask=None):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.position_embedding[:X.size(0)]
        return self.encoder(X)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MaskedLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vocab_size, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.size(1)
        bsz = torch.arange(X.size(0))
        batch_idx = torch.repeat_interleave(bsz, num_pred_positions)
        num_pred_positions = num_pred_positions.reshape(-1)
        return self.mlp(X[batch_idx, num_pred_positions])


class NextSentencePrediction(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, ):
        pass