import torch
import torch.nn as nn
import torch.nn.functional as F

class TreeNodeBlock(nn.Module):
    """
    深度优化版树节点模块：
    1. 引入 BatchNorm1d：加速收敛，稳定训练，对抗梯度消失/爆炸。
    2. 使用 GELU 激活：比 ReLU 更平滑，保留负值信息，防止神经元坏死。
    3. 强化残差连接：采用 Pre-activation 或 Post-activation 结构。
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TreeNodeBlock, self).__init__()

        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.pad_len = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=0, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=1, padding=0, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = None
        if n_inputs != n_outputs or stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(n_inputs, n_outputs, 1, stride=stride, bias=False),
                nn.BatchNorm1d(n_outputs)
            )

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = F.pad(x, (self.pad_len, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = F.pad(out, (self.pad_len, 0))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout2(out)

        if out.size(2) != residual.size(2):
            min_len = min(out.size(2), residual.size(2))
            out = out[:, :, :min_len]
            residual = residual[:, :, :min_len]

        return out + residual

class ForestTCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, stride_per_level=2):
        super(ForestTCN, self).__init__()
        self.layers = nn.ModuleList()
        self.num_levels = len(num_channels)

        for i in range(self.num_levels):
            dilation_size = 1
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            self.layers.append(
                TreeNodeBlock(
                    in_channels, out_channels, kernel_size,
                    stride=stride_per_level,
                    dilation=dilation_size,
                    dropout=dropout
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x