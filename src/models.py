import torch
from torch import nn
from nltk.tokenize import word_tokenize

class Skipgram(nn.Module):
    """The original context skip-gram model with softmax"""
    def __init__(self, vocab_size, embedding_dim):
        super(Skipgram, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            # max_norm=max_norm
        )
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size, bias=False) # I am not sure about the bias being false
    def forward(self, inputs):
        return self.linear(self.embed(inputs))
