# eng-fra-seq2seq
Some seq2seq implementations(vanilla, attn, transformer, bert).

## Comparison of different models

Architecture:

|filename|model | Encoder|Decoder|
|:-:|:-:|:-:|:-:|
|`model/vanilla_seq2seq.py`|Vanilla seq2seq|GRU|GRU|
|`model/attn_seq2seq.py`|Attention-based seq2seq|Bi-LSTM|LSTM+Attention|

Performance:



| model| Avg BLEU-2|Avg BLEU-3|Avg BLEU-4|
|:-:|:-:|:-:|:-:|
|Vanilla seq2seq|0.4799|0.3229|0.2144|
|Attention-based seq2seq|0.5711|0.4195|0.3036|


## Train and evaluate model

Take `vanilla_seq2seq` as an example, simply run the following commands in the terminal

```bash
git clone git@github.com:sonvier/eng-fra-seq2seq.git
cd eng-fra-seq2seq/
mkdir params output && nohup python -um model.vanilla_seq2seq > ./output/train.log 2>&1 &
```

View training progress in real time

```bash
tail -f ./output/train.log
```
