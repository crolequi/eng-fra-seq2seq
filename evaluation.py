import math
import torch

from collections import Counter
from data_preprocess import get_vocab_loader
# Import different models for evaluation
from model.vanilla_seq2seq import get_model


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


def evaluate(test_loader, model, bleu_k):
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
        bleu_scores.append(bleu(tgt_seq, pred_seq, k=bleu_k))

    return bleu_scores, translation_results


# Basic settings
SEQ_LEN = 45
src_vocab, tgt_vocab, _, test_loader = get_vocab_loader()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = get_model().to(device)
net.load_state_dict(torch.load('params/vanilla_seq2seq.pt'))

# Evaluation
bleu_1_scores, _ = evaluate(test_loader, net, bleu_k=1)
bleu_2_scores, _ = evaluate(test_loader, net, bleu_k=2)
bleu_3_scores, _ = evaluate(test_loader, net, bleu_k=3)
print(
    f"BLEU-1: {math.mean(bleu_1_scores)} | BLEU-2: {math.mean(bleu_2_scores)} | BLEU-3: {math.mean(bleu_3_scores)}"
)