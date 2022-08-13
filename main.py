import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import math
import matplotlib.pyplot as plt
from tkinter import _flatten
from collections import Counter


class Vocab:

    def __init__(self, tokens, min_freq=0):
        self.tokens = tokens
        self.min_freq = min_freq
        self.token2idx = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
        self.token2idx.update({
            token: idx + 4
            for idx, (token, freq) in enumerate(
                sorted(Counter(_flatten(self.tokens)).items(), key=lambda x: x[1], reverse=True))
            if freq >= self.min_freq
        })
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def __getitem__(self, tokens_or_indices):
        if isinstance(tokens_or_indices, (str, int)):
            return self.token2idx.get(tokens_or_indices, 0) if isinstance(
                tokens_or_indices, str) else self.idx2token.get(tokens_or_indices, '<unk>')
        elif isinstance(tokens_or_indices, (list, tuple)):
            return [self.__getitem__(item) for item in tokens_or_indices]
        else:
            raise TypeError

    def __len__(self):
        return len(self.idx2token)


def data_cleaning(content):
    for i in range(len(content)):
        special_chars = ['\u200b', '\xad', '\u2009', '\u202f', '\xa0']
        for j, char in enumerate(special_chars):
            content[i] = content[i].replace(char, ' ' if j > 1 else '')
        content[i] = content[i].lower()
        content[i] = ''.join([
            ' ' + char if j > 0 and char in ',.!?' and content[i][j - 1] != ' ' else char
            for j, char in enumerate(content[i])
        ])
    return content


def tokenize(cleaned_content):
    src_tokens, tgt_tokens = [], []
    for line in cleaned_content:
        pair = line.split('\t')
        src_tokens.append(pair[0].split(' '))
        tgt_tokens.append(pair[1].split(' '))
    return src_tokens, tgt_tokens


def truncate_pad(line, seq_len):
    return line[:seq_len] if len(line) > seq_len else line + ['<pad>'] * (seq_len - len(line))


def build_data(tokens, vocab, seq_len):
    return torch.tensor([vocab[truncate_pad(line + ['<eos>'], seq_len)] for line in tokens])


class Seq2SeqEncoder(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=1)
        self.rnn = nn.GRU(emb_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, encoder_inputs):
        encoder_inputs = self.embedding(encoder_inputs).permute(1, 0, 2)
        output, h_n = self.rnn(encoder_inputs)
        return h_n


class Seq2SeqDecoder(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=1)
        self.rnn = nn.GRU(emb_size + hidden_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, decoder_inputs, encoder_states):
        decoder_inputs = self.embedding(decoder_inputs).permute(1, 0, 2)
        context = encoder_states[-1]
        context = context.repeat(decoder_inputs.shape[0], 1, 1)
        output, h_n = self.rnn(torch.cat((decoder_inputs, context), -1), encoder_states)
        logits = self.fc(output)
        return logits, h_n


class Seq2SeqModel(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_inputs, decoder_inputs):
        return self.decoder(decoder_inputs, self.encoder(encoder_inputs))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bleu(label, pred, k=4):
    score = math.exp(min(0, 1 - len(label) / len(pred)))
    for n in range(1, k + 1):
        hashtable = Counter([' '.join(label[i:i + n]) for i in range(len(label) - n + 1)])
        num_matches = 0
        for i in range(len(pred) - n + 1):
            ngram = ' '.join(pred[i:i + n])
            if ngram in hashtable and hashtable[ngram] > 0:
                num_matches += 1
                hashtable[ngram] -= 1
        score *= math.pow(num_matches / (len(pred) - n + 1), math.pow(0.5, n))
    return score


def train(train_loader, model, criterion, optimizer, num_epochs):
    train_loss = []
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (encoder_inputs, decoder_targets) in enumerate(train_loader):
            encoder_inputs, decoder_targets = encoder_inputs.to(device), decoder_targets.to(device)
            bos_column = torch.tensor([tgt_vocab['<bos>']] * decoder_targets.shape[0]).reshape(-1, 1).to(device)
            decoder_inputs = torch.cat((bos_column, decoder_targets[:, :-1]), dim=1)
            pred, _ = model(encoder_inputs, decoder_inputs)
            loss = criterion(pred.permute(1, 2, 0), decoder_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if (batch_idx + 1) % 50 == 0:
                print(
                    f'[Epoch {epoch + 1}] [{(batch_idx + 1) * len(encoder_inputs)}/{len(train_loader.dataset)}] loss: {loss:.4f}'
                )
        print()
    return train_loss


def evaluate(test_loader, model):
    bleu_scores = []
    translation_results = []
    model.eval()
    for src_seq, tgt_seq in test_loader:
        encoder_inputs = src_seq.to(device)
        h_n = model.encoder(encoder_inputs)
        pred_seq = [tgt_vocab['<bos>']]
        for _ in range(SEQ_LEN):
            decoder_inputs = torch.tensor(pred_seq[-1]).reshape(1, 1).to(device)
            pred, h_n = model.decoder(decoder_inputs, h_n)
            next_token_idx = pred.squeeze().argmax().item()
            if next_token_idx == tgt_vocab['<eos>']:
                break
            pred_seq.append(next_token_idx)
        pred_seq = tgt_vocab[pred_seq[1:]]
        tgt_seq = tgt_seq.squeeze().tolist()
        tgt_seq = tgt_vocab[
            tgt_seq[:tgt_seq.index(tgt_vocab['<eos>'])]] if tgt_vocab['<eos>'] in tgt_seq else tgt_vocab[tgt_seq]
        translation_results.append((' '.join(tgt_seq), ' '.join(pred_seq)))
        bleu_scores.append(bleu(tgt_seq, pred_seq, k=2))

    return bleu_scores, translation_results


# Seed settings (for reproducibility)
setup_seed(42)

# Parameter settings
TRAIN_SIZE = 190000
TEST_SIZE = 4000
BATCH_SIZE = 512
SEQ_LEN = 45
LR = 0.001
EPOCHS = 50

# Read file
with open('fra.txt', encoding='utf-8') as f:
    content = ['\t'.join(line.strip().split('\t')[:-1]) for line in f.readlines()]

# Data preprocessing
src_tokens, tgt_tokens = tokenize(data_cleaning(content))
src_vocab, tgt_vocab = Vocab(src_tokens, min_freq=2), Vocab(tgt_tokens, min_freq=2)
src_data, tgt_data = build_data(src_tokens, src_vocab, SEQ_LEN), build_data(tgt_tokens, tgt_vocab, SEQ_LEN)
indices = torch.randperm(len(src_data))
src_data, tgt_data = src_data[indices], tgt_data[indices]
src_train_data, src_test_data = src_data[:TRAIN_SIZE], src_data[-TEST_SIZE:]
tgt_train_data, tgt_test_data = tgt_data[:TRAIN_SIZE], tgt_data[-TEST_SIZE:]
train_data = TensorDataset(src_train_data, tgt_train_data)
test_data = TensorDataset(src_test_data, tgt_test_data)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

# Model building
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = Seq2SeqEncoder(len(src_vocab), len(src_vocab), 256, num_layers=2, dropout=0.1)
decoder = Seq2SeqDecoder(len(tgt_vocab), len(tgt_vocab), 256, num_layers=2, dropout=0.1)
net = Seq2SeqModel(encoder, decoder).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

# Training phase
# When training, please comment out the code in the test phase
train_loss = train(train_loader, net, criterion, optimizer, EPOCHS)
torch.save(net.state_dict(), 'seq2seq_params.pt')
plt.plot(train_loss)
plt.ylabel('train loss')
plt.show()

# Test phase
# When training, please comment out the code in the training phase
net.load_state_dict(torch.load('seq2seq_params.pt'))
bleu_scores, translation_results = evaluate(test_loader, net)
plt.bar(range(len(bleu_scores)), bleu_scores)
plt.show()
