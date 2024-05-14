
import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor: Tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index: int) -> Tensor:
        return self.data_tensor[index]

    def __len__(self) -> int:
        return self.data_tensor.size(0)


def return_data(args, return_testset: bool = False, ):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size if not return_testset else 1
    num_workers = args.num_workers
    image_size = args.image_size
    datasets = ['train', 'valid']
    if return_testset:
        datasets = ['test', ]

    dataloaders = []
    for dataset in datasets:

        root = os.path.join('.', dset_dir, name, f'{dataset}.npz') 
        data_orig = np.load(root, encoding='bytes', allow_pickle = True)
        data = torch.from_numpy(data_orig['images']).float()
        if return_testset: 
            mask = data_orig['masks']
            detailed_mask = None
            try: detailed_mask = data_orig['detailed_masks'] 
            except:print("No detailed mask is available.")

        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset

        loader = DataLoader(dset(**train_kwargs),
                            batch_size=batch_size,
                            shuffle=not return_testset,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True)

        dataloaders.append(loader)

    if not return_testset:   return dataloaders
    else: return dataloaders, mask, detailed_mask


