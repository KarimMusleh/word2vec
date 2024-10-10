from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
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

class SG_Softmax_Batch_Sampler(Sampler):
    """A Sequential BatchSampler for Skipgram_SM CumulativeDataset
    For more information: https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#BatchSampler
    """
    def calculate_len(self):
        # Below is code to count the length of the DataLoader
        # I put it here because I don't want to recalculate it constantly
        cum_len = 0
        prev_size = 0
        for cum_size, batch_size in zip(self.cumulative_sizes, self.batch_sizes):
            curr_size = cum_size - prev_size
            cum_len = (curr_size + batch_size - 1) // batch_size
            prev_size = cum_size
        self.cum_len = cum_len
    def __init__(self, cumulative_sizes, batch_size):
        self.cumulative_sizes = cumulative_sizes
        print(self.cumulative_sizes)
        self.batch_sizes = [0] * len(cumulative_sizes)
        for i in range(len(cumulative_sizes)):
            self.batch_sizes[i] = max(1, round(batch_size/(i + 1)))
        self.calculate_len()
    def __len__(self):
        return self.cum_len
    def __iter__(self):
        prev_size = 0
        for size, batch_size in zip(self.cumulative_sizes, self.batch_sizes):
            batch = [0] * batch_size
            idx_in_batch = 0
            # Begin sequential sampling
            for i in range(prev_size, size):
                batch[idx_in_batch] = i
                idx_in_batch += 1
                if idx_in_batch == batch_size:
                    yield batch
                    idx_in_batch = 0
                    # The line below I found in the pytorch implementation of BatchSampler. It seems to me that it's not necessary
                    # batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
            prev_size = size


def sm_sg_dataloader(id_sents, id2word, window_size=WINDOW_SIZE, batch_size=60, gen=None):
    """
    A DataLoader for Skipgram with Softmax at the end.
    This implementation is very hacky, it splits the training data into two parts:
    1) Tokens that have window_size * 2 neighboring tokens.
    2) Tokens that don't.
    During training one has to train on both of them seperately.
    The biggest benefit of this implementation is that it saves memory by storing the center token only once.
    """
    centers, contexts = [], []
    for ids in id_sents:
        len_ids = len(ids)
        center, context = ids, [[] for _ in range(len_ids)]
        for i in range(len_ids):
            begin = max(0, i - window_size)
            for j in range(begin, i):
                context[i].append(ids[j])
                context[j].append(ids[i])
        centers.extend(center)
        contexts.extend(context)

    centers_contexts = list(zip(centers, contexts))
    d = [[[], []] for _ in range(window_size * 2)]

    for center, context in centers_contexts:
        if context:
            d[len(context) - 1][0].append(center)
            d[len(context) - 1][1].append(context)
    print([len(d[i][0]) for i in range(window_size * 2)])
    datasets = [SG_Dataset(centers, np.stack(contexts)) for centers, contexts in d if centers]

    merged_dataset = ConcatDataset(datasets)

    # We device the batch_size by the length of the context to ensure that the batch size is similar between iterations

    loader = DataLoader(merged_dataset, batch_sampler=SG_Softmax_Batch_Sampler(merged_dataset.cumulative_sizes, batch_size))
    return loader 

def ns_sg_dataloader(ids, id2word, window_size=WINDOW_SIZE, batch_size=64, gen=None):
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
    loader = DataLoader(sg_dataset, batch_size=batch_size, shuffle=True, generator=gen)
    return loader

def main():
    pass
if __name__ == '__main__':
    main()
