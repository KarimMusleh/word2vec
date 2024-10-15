from dataloader import sm_sg_dataloader, ns_sg_dataloader
from models import Skipgram_SM, Skipgram_NS
import torch
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
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

def main():
    ns_skipgram()

def sm_skipgram():
    """Train a Skipgram model like was explained in the first paper by Mikolov et.al."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')
    data = open('../data/shakespeare_normalized.txt', 'r').read()
    assert all([word.isalpha() and word.islower() for word in data.split()]), 'The data should contain sentences that are newline-seperated. The sentences should consist of space-seperated lowercase words'

    words = set(data.split())
    # When working with sents I can use this itertools.chain.from_iterable(sents) to trasnform to words 
    torch.save(word2id, MODELS_DIR + 'shakespeare-10-30-word2id.pl')
    print('tokens loaded')

    window_size = 2
    
    freq = Counter(data.split())
    # I implemented this for the future but right now I don't need it because my data is pretty small
    min_freq = 1 
    # min_sent_len = 2 is the smallest reasonable number as it removes all the sents that only consist of one word
    min_sent_len = 2
    
    word2id = {word: i for i, word in enumerate(words) if freq[word] >= min_freq}
    id2word = {i: word for word, i in word2id} # for future use

    # Split the input data into sentences and make sure that the length of a sentence isn't less than min_sent_len
    # Split the dataset into words and ensure that all the words have a frequency above min_freq
    id_sents = [[word2id[word] for word in sent.split(' ') if freq[word] >= min_freq] for sent in data.split('\n') if len(sent) >= min_sent_len]
    vocab_size = len(word2id)
    ids_len = len(list(chain().from_iterable(id_sents)))
    loader = sm_sg_dataloader(id_sents, id2word, window_size=window_size, gen=gen)
    print('dataloader loaded')

    # len_ids = len(list(chain().from_iterable(id_sents)))

    model = Skipgram_SM(vocab_size, EMBEDDING_SIZE)
    model.to(device)
    lr = 0.025 
    loss_fn = cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 300
    loss_values = []
    for epoch in range(epochs):
        running_loss = 0.0
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            context_len = y.shape()[1]
            loss = loss_fn(preds.unsqueeze(2).repeat(1, 1, context_len), y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(loader) 
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1} / {epochs}, Loss: {epoch_loss}')
        loss_values.append(epoch_loss)
    torch.save(model.state_dict(), MODELS_DIR + 'shakespeare-10-30.pl')

def ns_skipgram(window_size=2, negative_samples_count=4, embedding_size=EMBEDDING_SIZE):
    """Train a Skipgram model with Negative sampling like was explained in the second paper by Mikolov et.al."""
    def find_negative(token_num):
        """Find the first instance of token_prefix_sum that is greater than or equal to token_num"""

        l, h = 0, vocab_size - 1
        while l <= h:
            m = (l + h) // 2
            if token_num <= token_prefix_sum[m]:
                h = m - 1
            else:
                l = m + 1
        if l == vocab_size:
            l -= 1
        return l

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')
    data = open('../data/shakespeare_normalized.txt', 'r').read()

    words = set(data.split())
    # When working with sents I can use this itertools.chain.from_iterable(sents) to trasnform to words 
    
    freq = Counter(data.split())

    # I implemented this for the future but right now I don't need it because my data is pretty small
    min_freq = 1 
    # min_sent_len = 2 is the smallest reasonable number as it removes all the sents that only consist of one word
    min_sent_len = 2

    word2id = {word: i for i, word in enumerate(words) if freq[word] >= min_freq}
    torch.save(word2id, MODELS_DIR + 'shakespeare-10-30-word2id.pl')
    id2word = {i: word for word, i in word2id.items()} # for future use
    vocab_size = len(word2id)
    
    # Split the input data into sentences and make sure that the length of a sentence isn't less than min_sent_len
    # Split the dataset into words and ensure that all the words have a frequency above min_freq
    id_sents = [[word2id[word] for word in sent.split(' ') if freq[word] >= min_freq] for sent in data.split('\n') if len(sent) >= min_sent_len]

    print('tokens loaded')
    
    token_prefix_sum = [0] * vocab_size
    token_prefix_sum[0] = freq[0] ** (3/4)
    for i in range(1, vocab_size):
        token_prefix_sum[i] = token_prefix_sum[i - 1] + freq[i]

    token_prefix_sum_max = int(token_prefix_sum[-1]) + 1 # The idea here is that it's most likely gonna be a float so we floor it and add one to find the ciel
    # ids_len = len(list(chain().from_iterable(id_sents)))
    loader = sm_sg_dataloader(id_sents, id2word, window_size=window_size, gen=gen)
    print('dataloader loaded')

    # len_ids = len(list(chain().from_iterable(id_sents)))

    model = Skipgram_NS(vocab_size, embedding_size)
    model.to(device)
    lr = 0.025
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 300
    loss_values = []
    for epoch in range(epochs):
        running_loss = 0.0
        for centers, positives in loader:
            # X.shape: batch_size, context_len
            batch_size, context_len = positives.shape
            negatives = torch.randint(0, token_prefix_sum_max, (batch_size, context_len * negative_samples_count)).apply_(find_negative)
            
            positives = positives.to(device)
            negatives = negatives.to(device)
            centers = centers.to(device)

            optimizer.zero_grad()
            positive_score, negative_score = model(centers, positives, negatives)
            loss = (-positive_score.sigmoid().log().sum() / (batch_size * context_len) - (-negative_score).sigmoid().log().sum() / (batch_size * negative_samples_count * context_len))
            running_loss += loss.item()
            print(loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(loader) 
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1} / {epochs}, Loss: {epoch_loss}')
        loss_values.append(epoch_loss)
    torch.save(model.state_dict(), MODELS_DIR + 'shakespeare-10-30.pl')

if __name__ == '__main__':
    main()

