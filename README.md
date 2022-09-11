# eng-fra-seq2seq
Some seq2seq implementations based on eng-fra dataset(NMT task).

## Comparison of different models

Architecture:

<div align="center">

|filename|model | Encoder|Decoder|
|:-:|:-:|:-:|:-:|
|`model/vanilla_seq2seq.py`|Vanilla seq2seq|GRU|GRU|
|`model/lstm_seq2seq.py`|LSTM|LSTM|LSTM|
|`model/attn_seq2seq.py`|Attention-based seq2seq|Bi-LSTM|LSTM+Attention|
|`model/trans_seq2seq.py`|Transformer|Transformer|Transformer|




</div>

Performance:

<div align="center">

| model| Avg BLEU-2|Avg BLEU-3|Avg BLEU-4|
|:-:|:-:|:-:|:-:|
|Vanilla seq2seq|0.4799|0.3229|0.2144|
|LSTM| 0.5021 |  0.3587 |  0.2496 |
|Attention-based seq2seq|0.5711|0.4195|0.3036|
|Transformer| 0.7992| 0.7579| 0.7337 |

</div>





## Usage

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
