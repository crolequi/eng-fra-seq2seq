import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ..data_preprocess import get_loader_vocab
from ..model.vanilla_seq2seq import Seq2SeqDecoder, Seq2SeqEncoder, Seq2SeqModel


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


LR = 0.001
EPOCHS = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'
src_vocab, tgt_vocab, train_loader, test_loader = get_loader_vocab()
encoder = Seq2SeqEncoder(len(src_vocab), len(src_vocab), 256, num_layers=2, dropout=0.1)
decoder = Seq2SeqDecoder(len(tgt_vocab), len(tgt_vocab), 256, num_layers=2, dropout=0.1)
net = Seq2SeqModel(encoder, decoder).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

train_loss = train(train_loader, net, criterion, optimizer, EPOCHS)
torch.save(net.state_dict(), 'seq2seq_params.pt')
plt.plot(train_loss)
plt.ylabel('train loss')
plt.show()
