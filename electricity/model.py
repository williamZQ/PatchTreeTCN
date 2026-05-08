import torch
from torch import nn
from tree_tcn import ForestTCN

class PatchTreeTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, stride=2, seq_len=96, pred_len=96, patch_size=4):
        super(PatchTreeTCN, self).__init__()

        self.patch_size = patch_size
        self.pred_len = pred_len
        self.input_size = input_size

        self.patch_dim = input_size * patch_size
        self.patch_len = seq_len // patch_size

        if self.patch_len < 1:
            raise ValueError("Patch size is too large for the sequence length!")

        self.backbone_stride = 1

        self.forest_tcn = ForestTCN(
            input_size=self.patch_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            stride_per_level=self.backbone_stride
        )

        self.reduced_len = self.patch_len

        if self.backbone_stride > 1:
            for _ in range(len(num_channels)):
                self.reduced_len = (self.reduced_len + self.backbone_stride - 1) // self.backbone_stride

        print(f"DEBUG Info: SeqLen={seq_len} -> PatchLen={self.patch_len} -> TCN_Stride={self.backbone_stride} -> Final_Reduced_Len={self.reduced_len}")

        self.dropout = nn.Dropout(p=dropout)

        self.channel_proj = nn.Conv1d(num_channels[-1], output_size, kernel_size=1)

        self.temporal_proj = nn.Linear(self.reduced_len, pred_len)

    def forward(self, x):
        seq_mean = x.mean(dim=2, keepdim=True)
        seq_std = x.std(dim=2, keepdim=True) + 1e-5
        x = (x - seq_mean) / seq_std

        B, C, L = x.shape
        x = x.view(B, C, self.patch_len, self.patch_size)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * self.patch_size, self.patch_len)

        x = self.forest_tcn(x)
        x = self.dropout(x)

        x = self.channel_proj(x)

        x = self.temporal_proj(x)

        x = x * seq_std + seq_mean

        return x.permute(0, 2, 1)