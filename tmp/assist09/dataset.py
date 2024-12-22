"""
数据集加载策略
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class UserDataset(Dataset):
    def __init__(self):
        self.user_seq = torch.tensor(np.load('../../hy-tmp/data2009/user_seq.npy'), dtype=torch.int64)
        self.user_res = torch.tensor(np.load('../../hy-tmp/data2009/user_res.npy'), dtype=torch.int64)
        self.user_mask = torch.tensor(np.load('../../hy-tmp/data2009/user_mask.npy'), dtype=torch.bool)
        self.user_u = torch.tensor(np.load('../../hy-tmp/data2009/user_u.npy'), dtype=torch.int64)#

    def __getitem__(self, index):
        return torch.stack([self.user_seq[index], self.user_res[index], self.user_mask[index],self.user_u[index]], dim=-1)
    def __len__(self):
        return self.user_seq.shape[0]