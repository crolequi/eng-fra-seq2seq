import math
from collections import Counter


def bleu(label, pred, k=4):
    assert len(pred) >= k, "The length of the prediction sequence must not be less than k"
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
