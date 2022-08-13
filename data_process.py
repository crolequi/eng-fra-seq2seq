import torch
from torch.utils.data import TensorDataset
from utils.vocab import Vocab
from utils.setup import set_seed


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


# Parameter settings
set_seed(10086)
TRAIN_SIZE = 190000
TEST_SIZE = 4000
SEQ_LEN = 45

# Read file
with open('./data/eng-fra.txt', encoding='utf-8') as f:
    content = ['\t'.join(line.strip().split('\t')[:-1]) for line in f.readlines()]

# Get vocabulary
src_tokens, tgt_tokens = tokenize(data_cleaning(content))
src_vocab, tgt_vocab = Vocab(src_tokens), Vocab(tgt_tokens)

# Shuffle data
src_data, tgt_data = build_data(src_tokens, src_vocab, SEQ_LEN), build_data(tgt_tokens, tgt_vocab, SEQ_LEN)
random_indices = torch.randperm(len(src_data))
src_data, tgt_data = src_data[random_indices], tgt_data[random_indices]

# Divide training set and test set
src_train_data, src_test_data = src_data[:TRAIN_SIZE], src_data[-TEST_SIZE:]
tgt_train_data, tgt_test_data = tgt_data[:TRAIN_SIZE], tgt_data[-TEST_SIZE:]

# Build the dataset
train_data = TensorDataset(src_train_data, tgt_train_data)
test_data = TensorDataset(src_test_data, tgt_test_data)
