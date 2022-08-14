import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=1)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self, encoder_inputs):
        encoder_inputs = self.embedding(encoder_inputs).permute(1, 0, 2)
        output, (h_n, c_n) = self.rnn(encoder_inputs)
        # output shape: (seq_len, batch_size, 2 * hidden_size)
        # h_n shape: (2 * num_layers, batch_size, hidden_size)
        # c_n shape: same as h_n
        return output, h_n, c_n


class AttentionMechanism(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, decoder_state, encoder_output):
        # 解码器的隐藏层大小是编码器的两倍，否则无法进行接下来的内积操作
        # decoder_state shape: (batch_size, 2 * hidden_size)
        # encoder_output shape: (seq_len, batch_size, 2 * hidden_size)
        # -----before-----
        decoder_state = decoder_state.unsqueeze(1)  # (batch_size, 1, 2 * hidden_size)
        encoder_output = encoder_output.transpose(0, 1)  # (batch_size, seq_len, 2 * hidden_size)
        # scores shape: (batch_size, seq_len)
        scores = torch.sum(decoder_state * encoder_output, dim=-1) / math.sqrt(decoder_state.shape[2])  # 广播机制
        attn_weights = F.softmax(scores, dim=-1)
        # context shape: (batch_size, 2 * hidden_size)
        context = torch.sum(attn_weights.unsqueeze(-1) * encoder_output, dim=1)  # 广播机制
        return context


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=1)
        self.attn = AttentionMechanism()
        self.rnn = nn.LSTM(emb_size + 2 * hidden_size, 2 * hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, decoder_inputs, encoder_output, h_n, c_n):
        decoder_inputs = self.embedding(decoder_inputs).permute(1, 0, 2)  # (seq_len, batch_size, emb_size)
        h_n = torch.cat((h_n[::2], h_n[1::2]), dim=2)  # (num_layers, batch_size, 2 * hidden_size)
        c_n = torch.cat((c_n[::2], c_n[1::2]), dim=2)  # (num_layers, batch_size, 2 * hidden_size)
        decoder_outputs = torch.zeros()
        for i in range(len(decoder_inputs)):
            context = self.attn(h_n[-1], encoder_output)  # (batch_size, 2 * hidden_size)
            output, (h_n, c_n) = self.rnn(torch.cat((decoder_inputs[i], context), -1).unsqueeze(0), (h_n, c_n))
            # output shape: (1, batch_size, 2 * hidden_size)


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_inputs, decoder_inputs):
        return self.decoder(decoder_inputs, self.encoder(encoder_inputs))


def get_model():
    src_vocab, tgt_vocab, _, _ = get_vocab_loader()
    encoder = Seq2SeqEncoder(len(src_vocab), len(src_vocab), 256, num_layers=2, dropout=0.1)
    decoder = Seq2SeqDecoder(len(tgt_vocab), len(tgt_vocab), 256, num_layers=2, dropout=0.1)
    net = Seq2SeqModel(encoder, decoder)
    return net
