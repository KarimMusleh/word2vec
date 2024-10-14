from dataloader import sm_sg_dataloader, ns_sg_dataloader
from models import Skipgram_SM, Skipgram_NS
import torch
from torch.nn.functional import one_hot, cross_entropy
from torch import optim
from torch import nn
from collections import Counter
from constants import (
    MODELS_DIR,
    EMBEDDING_SIZE
)
from itertools import chain

gen = torch.manual_seed(42)

class Trainer:
    def __init__(
        self,
        model,
        epochs,
        dataloader,
        lr,
        ):
        pass

def sm_skipgram_cross_entropy(preds, target):
    batch_size, context_len = y.shape
    loss = 0
    for i in range(context_len):
        loss += cross_entropy(preds, y[:,i], reduction='sum') # we don't have to onehot encode y, cross_entropy does it for us
    # We have to find the loss ourselves because the batches don't all have the same shape
    return loss / (batch_size * context_size) 
     

def main():
    ns_skipgram()

def sm_skipgram():
    """Train a Skipgram model like was explained in the first paper by Mikolov et.al."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')
    data = open('../data/shakespeare_normalized.txt', 'r').read()

    words = set(data.split())
    vocab_size = len(words)
    # When working with sents I can use this itertools.chain.from_iterable(sents) to trasnform to words 
    word2id = {word: i for i, word in enumerate(words)}
    torch.save(word2id, MODELS_DIR + 'shakespeare-10-30-word2id.pl')
    id2word = {i: word for i, word in enumerate(words)} # for future use
    print('tokens loaded')

    window_size = 2
    
    freq = Counter(data.split())
    # I implemented this for the future but right now I don't need it because my data is pretty small
    min_freq = 1 
    # min_sent_len = 2 is the smallest reasonable number as it removes all the sents that only consist of one word
    min_sent_len = 2
    
    # Split the input data into sentences and make sure that the length of a sentence isn't less than min_sent_len
    # Split the dataset into words and ensure that all the words have a frequency above min_freq
    id_sents = [[word2id[word] for word in sent.split(' ') if freq[word] >= min_freq] for sent in data.split('\n') if len(sent) >= min_sent_len]
    ids_len = len(list(chain().from_iterable(id_sents)))
    dataloader = sm_sg_dataloader(id_sents, id2word, window_size=window_size, gen=gen)
    print('dataloader loaded')
    len_ids = len(list(chain().from_iterable(id_sents)))

    model = Skipgram_SM(vocab_size, EMBEDDING_SIZE)
    model.to(device)
    lr = 0.025 
    loss_fn = sm_skipgram_cross_entropy
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
        epoch_loss = running_loss / len_ids 
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1} / {epochs}, Loss: {epoch_loss}')
        loss_values.append(epoch_loss)
    torch.save(model.state_dict(), MODELS_DIR + 'shakespeare-10-30.pl')

def ns_skipgram():
    """Train a Skipgram model with Negative sampling like was explained in the second paper by Mikolov et.al."""
    def find_neg_samples(center_idx):
        for _ in range(NEGATIVE_SAMPLING_COUNT):
            token_num = randint(0, num_tokens)
            # We want to find the first instance of token_prefix_sum that is greater than or equal to token_num

            l, h = 0, vocab_size - 1
            while l <= h:
                m = (l + h) // 2
                if token_num <= token_prefix_sum[m]:
                    h = m - 1
                else:
                    l = m + 1
            center.append(center_idx)
            context.append(-l)
    
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
    dataloader = ns_sg_dataloader(ids, token_prefix_sum, id2word, window_size=window_size, gen=gen)
    ids_tensor = torch.tensor(ids) # We use this to access random elements for negative sampling
    print('dataloader loaded')
    return
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
        epoch_loss = running_loss / len_ids 
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1} / {epochs}, Loss: {epoch_loss}')
        loss_values.append(epoch_loss)
    torch.save(model.state_dict(), MODELS_DIR + 'shakespeare-10-30.pl')


if __name__ == '__main__':
    main()

