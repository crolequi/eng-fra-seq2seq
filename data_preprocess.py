import torch
from torch.utils.data import TensorDataset, DataLoader
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


def get_vocab_loader(train_size=190000, test_size=4000, batch_size=512, seq_len=45):
    with open('data/eng-fra.txt', encoding='utf-8') as f:
        content = ['\t'.join(line.strip().split('\t')[:-1]) for line in f.readlines()]

    src_tokens, tgt_tokens = tokenize(data_cleaning(content))
    src_vocab, tgt_vocab = Vocab(src_tokens, min_freq=2), Vocab(tgt_tokens, min_freq=2)
    src_data, tgt_data = build_data(src_tokens, src_vocab, seq_len), build_data(tgt_tokens, tgt_vocab, seq_len)
    indices = torch.randperm(len(src_data))
    src_data, tgt_data = src_data[indices], tgt_data[indices]
    src_train_data, src_test_data = src_data[:train_size], src_data[-test_size:]
    tgt_train_data, tgt_test_data = tgt_data[:train_size], tgt_data[-test_size:]
    train_data = TensorDataset(src_train_data, tgt_train_data)
    test_data = TensorDataset(src_test_data, tgt_test_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1)

    return src_vocab, tgt_vocab, train_loader, test_loader
