import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: (n, N, embed_dim)
            key: (m, N, embed_dim)
            value: (m, N, embed_dim)
            attn_mask (bool Tensor or float Tensor): (n, m) or (N * num_heads, n, m)
            key_padding_mask (bool Tensor): (N, m)

        Returns:
            attn_output: (n, N, embed_dim)
            attn_output_weights: (N, num_heads, n, m)
        """
        return self._multi_head_forward_attention(query,
                                                  key,
                                                  value,
                                                  dropout_p=self.dropout,
                                                  attn_mask=attn_mask,
                                                  key_padding_mask=key_padding_mask,
                                                  training=self.training)

    def _multi_head_forward_attention(self,
                                      query,
                                      key,
                                      value,
                                      dropout_p,
                                      attn_mask=None,
                                      key_padding_mask=None,
                                      training=True):
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        n, N, embed_dim = q.size()
        m = key.size(0)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                assert attn_mask.shape == (n, m)
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                assert attn_mask.shape == (N * self.num_heads, n, m)
            else:
                raise RuntimeError

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (N, m)
            key_padding_mask = key_padding_mask.view(N, 1, 1, m).repeat(1, self.num_heads, 1,
                                                                        1).reshape(N * self.num_heads, 1, m)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, -1e9)

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, -1e9)
            attn_mask = new_attn_mask

        q = q.reshape(n, N * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(m, N * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(m, N * self.num_heads, self.head_dim).transpose(0, 1)

        if not training:
            dropout_p = 0.0

        attn_output, attn_output_weights = self._scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        attn_output = attn_output.transpose(0, 1).reshape(n, N, embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output_weights = attn_output_weights.reshape(N, self.num_heads, n, m)
        return attn_output, attn_output_weights

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p=0.0):
        """
        Args:
            q: (N, n, E), where E is embedding dimension.
            k: (N, m, E)
            v: (N, m, E)
            attn_mask: (n, m) or (N, n, m)
        
        Returns:
            attn_output: (N, n, E)
            attn_weights: (N, n, m)
        """
        q = q / math.sqrt(q.size(2))
        if attn_mask is not None:
            scores = q @ k.transpose(-2, -1) + attn_mask
        else:
            scores = q @ k.transpose(-2, -1)

        attn_weights = F.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        attn_output = attn_weights @ v
        return attn_output, attn_weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout=dropout, bias=bias)

    def forward(self, X, attn_mask=None, key_padding_mask=None):
        """
        Args:
            X (input sequence): (L, N, embed_dim), where L is sequence length.
        """
        return self.mha(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
