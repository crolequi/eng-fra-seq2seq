# eng-fra-seq2seq
Some seq2seq implementations(vanilla, attn, transformer, bert).

## Comparison of different models

|model | Encoder|Decoder|
|:-:|:-:|:-:|
|vanilla seq2seq|GRU|GRU|
|attention-based seq2seq|Bi-LSTM|LSTM+Attention|


| model| Avg BLEU-2|Avg BLEU-3|Avg BLEU-4|
|:-:|:-:|:-:|:-:|
|vanilla seq2seq|0.6341|0.5121|0.4103|