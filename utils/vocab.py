from collections import Counter
from tkinter import _flatten


class Vocab:
    def __init__(self, tokens, min_count=2, special_tokens=['<unk>', '<pad>', '<bos>', '<eos>']):
        assert min_count >= 2 and '<unk>' in special_tokens
        self.tokens = tokens
        self.min_count = min_count
        self.token2idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.token2idx.update({
            token: idx + len(special_tokens)
            for idx, (token, count) in enumerate(
                sorted(Counter(_flatten(self.tokens)).items(), key=lambda x: x[1], reverse=True))
            if count >= self.min_count
        })
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def __getitem__(self, tokens_or_indices):
        if isinstance(tokens_or_indices, (str, int)):
            return self.token2idx.get(tokens_or_indices, self.token2idx['<unk>']) if isinstance(
                tokens_or_indices, str) else self.idx2token.get(tokens_or_indices, '<unk>')
        elif isinstance(tokens_or_indices, (list, tuple)):
            return [self.__getitem__(item) for item in tokens_or_indices]
        else:
            raise TypeError

    def __len__(self):
        return len(self.idx2token)
