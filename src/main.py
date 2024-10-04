from preprocess import simple_filter_tokenize
from dataloader import sm_sg_dataloader, ns_sg_dataloader
from models import Skipgram_SM, Skipgram_NS
import torch
from torch.nn.functional import one_hot
from torch import optim
from torch import nn
from collections import Counter
from constants import (
    MODELS_DIR,
    EMBEDDING_SIZE
)

def main():
    softmax_skipgram()
def ns_skipgram():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')
    data = open('../data/shakespeare.txt', 'r').read()
    tokens = simple_filter_tokenize(data)[:10000]
    print('tokens loaded')

    words = set(tokens)
    # When working with sents I can use this itertools.chain.from_iterable(sents) to trasnform to words 
    word2id = {word: i for i, word in enumerate(words)}
    torch.save(word2id, MODELS_DIR + 'shakespeare-10-30-word2id.pl')
    id2word = {i: word for i, word in enumerate(words)} # useful for debugging

    window_size = 2
    vocab_size = len(words)
    
    ids = [word2id[token] for token in tokens]
    # idx_freq = Counter(ids)
    # min_freq = 10
    # ids = [idx for idx in ids if idx_freq[idx] >= min_freq]
    dataloader = ns_sg_dataloader(ids, window_size=window_size)
    print('dataloader loaded')

    model = Skipgram_NS(vocab_size, EMBEDDING_SIZE)
    model.to(device)
    lr = 0.025 
    loss_fn = nn.BCEWithLogitsLoss()
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
            loss = loss_fn(preds, y) # we don't have to onehot encode y nn.CrossEntropyLoss does it for us
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1} / {epochs}, Loss: {epoch_loss}')
        loss_values.append(epoch_loss)
    torch.save(model.state_dict(), MODELS_DIR + 'shakespeare-10-30.pl')
    
def softmax_skipgram():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')
    data = open('../data/shakespeare.txt', 'r').read()
    tokens = simple_filter_tokenize(data)[:10000]
    print('tokens loaded')

    words = set(tokens)
    # When working with sents I can use this itertools.chain.from_iterable(sents) to trasnform to words 
    word2id = {word: i for i, word in enumerate(words)}
    torch.save(word2id, MODELS_DIR + 'shakespeare-10-30-word2id.pl')
    id2word = {i: word for i, word in enumerate(words)} # for future use

    window_size = 2
    vocab_size = len(words)
    
    ids = [word2id[token] for token in tokens]
    # idx_freq = Counter(ids)
    # min_freq = 10
    # ids = [idx for idx in ids if idx_freq[idx] >= min_freq]
    dataloader = _sg_dataloader(ids, id2word, window_size=window_size)
    print('dataloader loaded')

    model = Skipgram_SM(vocab_size, EMBEDDING_SIZE)
    model.to(device)
    lr = 0.025 
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
            loss = loss_fn(preds, y) # we don't have to onehot encode y nn.CrossEntropyLoss does it for us
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1} / {epochs}, Loss: {epoch_loss}')
        loss_values.append(epoch_loss)
    torch.save(model.state_dict(), MODELS_DIR + 'shakespeare-10-30.pl')

    
if __name__ == '__main__':
    main()
