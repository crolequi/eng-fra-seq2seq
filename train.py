import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_preprocess import get_vocab_loader
# Import different models for training
from model.vanilla_seq2seq import get_model


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


# Basic settings
setup_seed(42)
LEARNING_RATE = 0.001
NUM_EPOCH = 50
BATCH_SIZE = 512
SEQ_LEN = 45

device = 'cuda' if torch.cuda.is_available() else 'cpu'
src_vocab, tgt_vocab, train_loader, _ = get_vocab_loader(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
net = get_model().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Training
train_loss = train(train_loader, net, criterion, optimizer, NUM_EPOCH)
torch.save(net.state_dict(), 'vanilla_seq2seq.pt')
plt.plot(train_loss)
plt.ylabel('train loss')
plt.savefig('./loss.png')
