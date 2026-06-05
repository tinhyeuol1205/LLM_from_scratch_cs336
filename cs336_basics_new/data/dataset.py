import torch
from torch.utils.data import Dataset
import numpy as np

class LMDataset(Dataset):
    def __init__(self, data_path: str, context_length: int):
        self.dataset = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.context_length = context_length
    def __len__(self):
        return len(self.dataset) - self.context_length
    def __getitem__(self, idx):
        x = torch.from_numpy(self.dataset[idx:idx+self.context_length].astype(np.int64))
        y = torch.from_numpy(self.dataset[idx+1:idx+self.context_length+1].astype(np.int64))
        return (x, y)

