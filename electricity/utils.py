import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ElectricityDataset(Dataset):
    def __init__(self, file_path, seq_len=96, pred_len=24, mode='train'):
        self.seq_len = seq_len
        self.pred_len = pred_len

        df_raw = pd.read_csv(file_path)
        data = df_raw.values[:, 1:].astype(np.float32)

        num_train = int(len(data) * 0.7)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]

        if mode == 'train':
            curr_border = (border1s[0], border2s[0])
        elif mode == 'val':
            curr_border = (border1s[1], border2s[1])
        else:
            curr_border = (border1s[2], border2s[2])

        self.scaler = StandardScaler()
        train_data = data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(data)[curr_border[0]:curr_border[1]]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return torch.tensor(seq_x).permute(1, 0), torch.tensor(seq_y)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

def data_generator(file_path, batch_size, seq_len, pred_len):
    train_set = ElectricityDataset(file_path, seq_len, pred_len, mode='train')
    val_set = ElectricityDataset(file_path, seq_len, pred_len, mode='val')
    test_set = ElectricityDataset(file_path, seq_len, pred_len, mode='test')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader