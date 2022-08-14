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


## Train and evaluate model

Take `vanilla_seq2seq` as an example, simply run the following command in the terminal

```bash
mkdir params output && nohup python -um model.vanilla_seq2seq > ./output/train.log 2>&1 &
```

View training progress in real time

```bash
tail -f ./output/train.log
```