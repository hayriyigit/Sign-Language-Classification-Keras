import numpy as np
from tensorflow.keras.utils import Sequence

class Dataset:
    def __init__(self, data):
        self.images, self.labels = data

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.images)

class DataLoader(Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.indices = np.arange(len(dataset))
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size

        data = []
        for idx in range(start, stop):
            data.append(self.dataset[idx])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
                #images, #labels
        return batch[0], batch[1]

    def __len__(self):
        return len(self.indices) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indices = np.random.permutation(self.indices)