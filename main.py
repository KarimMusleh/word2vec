import torch
from torch import nn
from nltk.tokenize import word_tokenize

class word2vec(nn.Module):
    def __init__(self, mode='skipgram'):
        
        nn.Embedding(num_embeddings=vocab_size)


def main():
    train = open('train.txt', 'r').read()
    tkn = word_tokenize(train, language='english', preserve_line=True)
    tkn[:10]
     
if __name__ == '__main__':
    main()
