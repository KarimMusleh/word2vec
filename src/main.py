from preprocess import simple_filter_tokenize
from dataloader import get_sg_dataloader
from models import Skipgram
import torch
from torch import optim
from torch import nn

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')
    data = open('../data.txt', 'r').read()
    tokens = simple_filter_tokenize(data)
    print('tokens loaded')

    words = set(tokens)
    # When working with sents I can use this itertools.chain.from_iterable(sents) to trasnform to words 
    word2id = {word: i for i, word in enumerate(words)}
    id2word = {i: word for i, word in enumerate(words)} # for future use

    window_size = 2
    ids = [word2id[token] for token in tokens]
    dataloader = get_sg_dataloader(ids, window_size=window_size)
    print('dataloader loaded')

    vocab_size = len(words)
    model = Skipgram(vocab_size, 10)
    model.to(device)
    lr = 1e-2
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 300
    loss_values = []
    for epoch in range(epochs):
        running_loss = 0.0
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1} / {epochs}, Loss: {epoch_loss}')
        loss_values.append(epoch_loss)
    

    
if __name__ == '__main__':
    main()
