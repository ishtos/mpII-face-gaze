import os
import sys
import h5py
import numpy as np
import pandas as pd

import torch
import torch.utils.data

class MPIIFaceGazeDataset(torch.utils.data.Dataset):
    def __init__(self, subject_id, dataset_dir):
        path = os.path.join(dataset_dir, '{}.h5'.format(subject_id))
        with h5py.File(path) as f:
            self.images = f['data'].value
            self.gazes = f['label'].value
        self.length = len(self.images)

        self.images = torch.unsqueeze(torch.from_numpy(self.images), 1)
        self.gazes = torch.from_numpy(self.gazes)

    def __getitem__(self, index):
        return self.images[index][[2, 1, 0],:,:].transpose((1, 2, 0)), self.gazes[index][0:2]

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__


def get_loader(dataset_dir, test_subject_id, batch_size, num_workers, use_gpu):
    assert os.path.exists(dataset_dir)
    assert test_subject_id in range(15)

    subject_ids = ['p{:02}'.format(i) for i in range(15)]
    test_subject_id = subject_ids[test_subject_id]

    train_dataset = torch.utils.data.ConcatDataset([
        MPIIFaceGazeDataset(subject_id, dataset_dir) for subject_id in subject_ids
        if subject_id != test_subject_id
    ])
    test_dataset = MPIIFaceGazeDataset(test_subject_id, dataset_dir)

    assert len(train_dataset) == 42000
    assert len(test_dataset) == 3000

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader
