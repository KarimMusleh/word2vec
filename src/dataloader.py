from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np # To my understanding this is recommended for preprocessing data rather than pytorch

class SG_Dataset(Dataset):
    """A dataset for training the skipgram model. Given a center word it has to give a probability distribution for the context words"""
    def __init__(self, df):
        self.X = df['center']
        self.y = df['context']
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

def get_sg_dataloader(ids, window_size=2, batch_size=64):
    center, context = [], []
    len_ids = len(ids)
    for i in range(1, len_ids):
        begin = max(0, i - window_size)
        for j in range(begin, i):
            center.append(ids[i])
            context.append(ids[j])

            center.append(ids[j])
            context.append(ids[i])
    center = np.array(center)
    context = np.array(context)

    df = pd.DataFrame({'center': center, 'context': context})
    sg_dataset = SG_Dataset(df)
    loader = DataLoader(sg_dataset, batch_size=batch_size, shuffle=True)
    return loader

def main():
    pass
if __name__ == '__main__':
    main()
