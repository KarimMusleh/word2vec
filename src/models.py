import torch
from torch import nn
from nltk.tokenize import word_tokenize

class Skipgram_SM(nn.Module):
    """The original context skip-gram model with softmax"""
    def __init__(self, vocab_size, embedding_dim):
        super(Skipgram_SM, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            # max_norm=max_norm
        )
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False) # I am not sure about the bias being false
    def forward(self, center):
        return self.linear(self.embed(center))
class Skipgram_NS(nn.Module):
    """Skipgram model with negative sampling"""
    def __init__(self, vocab_size, embedding_dim):
        super(Skipgram_NS, self).__init__()
        self.target_embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.context_embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, center, context):
        center_embedding = self.center_embed(center)
        context_embedding = self.context_embed(context)
        
