from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np # To my understanding this is recommended for preprocessing data rather than pytorch
from constants import (
    EMBEDDING_SIZE,
    WINDOW_SIZE
)

class SG_Dataset(Dataset):
    """
    A dataset for training the skipgram model.
    This can be used for either one of the skipgram models
    """
    def __init__(self, center, context):
        self.X = center
        self.y = context
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        center = self.X[idx]
        context = self.y[idx]
        return center,  context
class CBOW_Dataset(Dataset):
    def __init__(self, df):
        self.X = df.drop('center',axis=1) 
        self.y = df['center']
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        center = X[idx]
        context = y[idx]
        return context, center 

def ns_sg_dataloader(ids, words, window_size=WINDOW_SIZE, batch_size=64):
    vocab_size = len(words)
    center, context = [], []
    len_ids = len(ids)
    for i in range(1, len_ids):
        begin = max(0, i - window_size)
        for j in range(begin, i):
            center.append(ids[i])
            context.append(ids[j])

            center.append(ids[j])
            context.append(ids[i])
    center = np.stack(center)
    context = np.stack(context)

    print(f'length of input {len(ids)}, length of training set {len(center)}')

    sg_dataset = SG_Dataset(center, context)
    loader = DataLoader(sg_dataset, batch_size=batch_size, shuffle=True)
    return loader

def sm_sg_dataloader(ids, id2word, window_size=WINDOW_SIZE, batch_size=64):
    center, context = [], []
    len_ids = len(ids)
    for i in range(1, len_ids):
        begin = max(0, i - window_size)
        for j in range(begin, i):
            center.append(ids[i])
            context.append(ids[j])

            center.append(ids[j])
            context.append(ids[i])
    center = np.stack(center)
    context = np.stack(context)

    print(f'length of input {len(ids)}, length of training set {len(center)}')

    sg_dataset = SG_Dataset(center, context)
    loader = DataLoader(sg_dataset, batch_size=batch_size, shuffle=True)
    return loader

def main():
    pass
if __name__ == '__main__':
    main()
