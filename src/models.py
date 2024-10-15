import torch
from torch import nn
from nltk.tokenize import word_tokenize

class Skipgram_SM(nn.Module):
    """The original context skip-gram model with softmax"""
    def __init__(self, vocab_size, embedding_dim):
        super(Skipgram_SM, self).__init__()
        self.target_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            # max_norm=max_norm
        )
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, center):
        center_embedding = self.target_embeddings(center) # batch_size, embedding_dim
        context_preds = self.linear(center_embedding) # batch_size, vocab_size
        return context_preds
class Skipgram_NS(nn.Module):
    """Skipgram model with negative sampling"""
    def __init__(self, vocab_size, embedding_dim):
        super(Skipgram_NS, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        nn.init.xavier_uniform_(self.target_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)

    def forward(self, center, positive, negative):
        center_embedding = self.target_embeddings(center) # shape: batch_size, embedding_dim
        positive_embedding = self.context_embeddings(positive) # shape: batch_size, context_len, embedding_dim
        negative_embedding = self.context_embeddings(negative) # shape: batch_size, context_len * negative_samples_count, embedding_dim
        
        positive_score = torch.bmm(positive_embedding, center_embedding.unsqueeze(2)).squeeze(2) # shape: batch_size, context_len
        negative_score = torch.bmm(negative_embedding, center_embedding.unsqueeze(2)).squeeze(2) # shape: batch_size, context_len * negative_samples_count
        
        return positive_score, negative_score