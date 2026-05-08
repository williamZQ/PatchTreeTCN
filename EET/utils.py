import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ETTh1Dataset(Dataset):
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

        self.scaler_x = StandardScaler()
        train_data_x = data[border1s[0]:border2s[0]]
        self.scaler_x.fit(train_data_x)
        self.data_x = self.scaler_x.transform(data)[curr_border[0]:curr_border[1]]

        self.scaler_y = StandardScaler()
        train_data_y = data[border1s[0]:border2s[0], -1].reshape(-1, 1)
        self.scaler_y.fit(train_data_y)
        self.data_y = self.scaler_y.transform(data[:, -1].reshape(-1, 1))[curr_border[0]:curr_border[1]]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.tensor(seq_x).permute(1, 0), torch.tensor(seq_y).permute(1, 0)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

def data_generator(file_path, batch_size, seq_len, pred_len):
    train_set = ETTh1Dataset(file_path, seq_len, pred_len, mode='train')
    val_set = ETTh1Dataset(file_path, seq_len, pred_len, mode='val')
    test_set = ETTh1Dataset(file_path, seq_len, pred_len, mode='test')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader