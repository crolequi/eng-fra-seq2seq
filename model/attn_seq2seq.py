import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_process import *
from utils.setup import set_seed
from utils.metrics import bleu

import numpy as np
import matplotlib.pyplot as plt


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=1)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self, encoder_inputs):
        encoder_inputs = self.embedding(encoder_inputs).permute(1, 0, 2)
        output, (h_n, c_n) = self.rnn(encoder_inputs)  # output shape: (seq_len, batch_size, 2 * hidden_size)
        h_n = torch.cat((h_n[::2], h_n[1::2]), dim=2)  # (num_layers, batch_size, 2 * hidden_size)
        c_n = torch.cat((c_n[::2], c_n[1::2]), dim=2)  # (num_layers, batch_size, 2 * hidden_size)
        return output, h_n, c_n


class AttentionMechanism(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, decoder_state, encoder_output):
        # 解码器的隐藏层大小必须是编码器的两倍，否则无法进行接下来的内积操作
        # decoder_state shape: (batch_size, 2 * hidden_size)
        # encoder_output shape: (seq_len, batch_size, 2 * hidden_size)
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
        # 注意将其移动到GPU上
        decoder_output = torch.zeros(decoder_inputs.shape[0], *h_n.shape[1:]).to(device)  # (seq_len, batch_size, 2 * hidden_size)
        for i in range(len(decoder_inputs)):
            context = self.attn(h_n[-1], encoder_output)  # (batch_size, 2 * hidden_size)
            # single_step_output shape: (1, batch_size, 2 * hidden_size)
            single_step_output, (h_n, c_n) = self.rnn(torch.cat((decoder_inputs[i], context), -1).unsqueeze(0), (h_n, c_n))
            decoder_output[i] = single_step_output.squeeze()
        logits = self.fc(decoder_output)  # (seq_len, batch_size, vocab_size)
        return logits, h_n, c_n


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_inputs, decoder_inputs):
        return self.decoder(decoder_inputs, *self.encoder(encoder_inputs))


def train(train_loader, model, criterion, optimizer, num_epochs):
    train_loss = []
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (encoder_inputs, decoder_targets) in enumerate(train_loader):
            encoder_inputs, decoder_targets = encoder_inputs.to(device), decoder_targets.to(device)
            bos_column = torch.tensor([tgt_vocab['<bos>']] * decoder_targets.shape[0]).reshape(-1, 1).to(device)
            decoder_inputs = torch.cat((bos_column, decoder_targets[:, :-1]), dim=1)
            pred, _, _ = model(encoder_inputs, decoder_inputs)
            loss = criterion(pred.permute(1, 2, 0), decoder_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if (batch_idx + 1) % 50 == 0:
                print(
                    f'[Epoch {epoch + 1}] [{(batch_idx + 1) * len(encoder_inputs)}/{len(train_loader.dataset)}] loss: {loss:.4f}')
        print()
    return train_loss


def evaluate(test_loader, model, bleu_k):
    bleu_scores = []
    translation_results = []
    model.eval()
    for src_seq, tgt_seq in test_loader:
        encoder_inputs = src_seq.to(device)
        encoder_output, h_n, c_n = model.encoder(encoder_inputs)
        pred_seq = [tgt_vocab['<bos>']]
        for _ in range(SEQ_LEN):
            decoder_inputs = torch.tensor(pred_seq[-1]).reshape(1, 1).to(device)
            pred, h_n, c_n = model.decoder(decoder_inputs, encoder_output, h_n, c_n)
            next_token_idx = pred.squeeze().argmax().item()
            if next_token_idx == tgt_vocab['<eos>']:
                break
            pred_seq.append(next_token_idx)
        pred_seq = tgt_vocab[pred_seq[1:]]
        tgt_seq = tgt_seq.squeeze().tolist()
        tgt_seq = tgt_vocab[tgt_seq[:tgt_seq.index(tgt_vocab['<eos>'])]] if tgt_vocab['<eos>'] in tgt_seq else tgt_vocab[tgt_seq]
        translation_results.append((' '.join(tgt_seq), ' '.join(pred_seq)))
        if len(pred_seq) >= bleu_k:
            bleu_scores.append(bleu(tgt_seq, pred_seq, k=bleu_k))

    return bleu_scores, translation_results


# Parameter settings
set_seed()
BATCH_SIZE = 512
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Dataloader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

# Model building
device = 'cuda'
encoder = Seq2SeqEncoder(len(src_vocab), len(src_vocab), 256)
decoder = Seq2SeqDecoder(len(tgt_vocab), len(tgt_vocab), 256)
net = Seq2SeqModel(encoder, decoder).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Training phase
train_loss = train(train_loader, net, criterion, optimizer, NUM_EPOCHS)
torch.save(net.state_dict(), './params/attn_seq2seq.pt')
plt.plot(train_loss)
plt.ylabel('train loss')
plt.savefig('./output/loss.png')

# Evaluation
bleu_2_scores, _ = evaluate(test_loader, net, bleu_k=2)
bleu_3_scores, _ = evaluate(test_loader, net, bleu_k=3)
bleu_4_scores, _ = evaluate(test_loader, net, bleu_k=4)
print(f"BLEU-2: {np.mean(bleu_2_scores)} | BLEU-3: {np.mean(bleu_3_scores)} | BLEU-4: {np.mean(bleu_4_scores)}")
