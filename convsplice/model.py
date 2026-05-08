import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.SiLU()
        self._initialize()

    def _initialize(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, dilation_rate, stride=1):
        super().__init__()
        padding = dilation_rate * (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_feat, out_feat, kernel_size, stride, padding=padding, dilation=dilation_rate),
            nn.BatchNorm1d(out_feat),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_feat, out_feat, kernel_size, padding=padding, dilation=dilation_rate),
            nn.BatchNorm1d(out_feat),
        )
        self._initialize()

    def _initialize(self):
        nn.init.kaiming_normal_(self.conv1[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1[0].bias, 0)
        nn.init.kaiming_normal_(self.conv2[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv2[0].bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class ConvEncoder(nn.Module):
    def __init__(self, feat_dim, CL, kernel_sizes, dilation_rates):
        super().__init__()
        self.conv1 = BasicConv(4, feat_dim, kernel_size=1)
        self.skip = nn.Conv1d(feat_dim, feat_dim, kernel_size=1)

        self.conv_blocks = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            self.conv_blocks.append(ConvBlock(feat_dim, feat_dim, kernel_sizes[i], dilation_rates[i]))

        self.conv_block_ends = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            if ((i + 1) % 4 == 0) or ((i + 1) == kernel_sizes):
                self.conv_block_ends.append(nn.Conv1d(feat_dim, feat_dim, kernel_size=1))

        self.CL = CL

    def forward(self, x):
        x = x.transpose(1, 2).float()
        x = self.conv1(x)
        skip = self.skip(x)
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.conv_blocks)):
                dense = self.conv_block_ends[i // 4](x)
                skip = skip + dense
        skip = skip[:, :, self.CL // 2:-self.CL // 2]
        return skip


class ConvSplice(nn.Module):
    """ConvSplice splice-site classifier.

    Input: one-hot encoded sequence of length CL + center, channels-last (B, L, 4).
    Output: per-base softmax over (neither, acceptor, donor) for the central window,
    shape (B, center, 3).
    """

    def __init__(self, feat_dim, CL, kernel_sizes, dilation_rates):
        super().__init__()
        self.conv_encoder = ConvEncoder(feat_dim, CL, kernel_sizes, dilation_rates)
        self.conv1 = nn.Conv1d(feat_dim, 128, kernel_size=1)
        self.relu = nn.SiLU()
        self.conv2 = nn.Conv1d(128, 3, kernel_size=1)

    def forward(self, x):
        x = self.conv_encoder(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = F.softmax(x, dim=1)
        return out.transpose(1, 2).float()


# Default architecture used by the released ConvSplice (20kb context) weights.
DEFAULT_KERNEL_SIZES = [11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21,
                        41, 41, 41, 41,
                        51, 51, 51, 51]
DEFAULT_DILATION_RATES = [1, 1, 1, 1, 4, 4, 4, 4,
                          10, 10, 10, 10,
                          25, 25, 25, 25,
                          25, 25, 25, 25]
DEFAULT_CL = 20000
DEFAULT_FEAT_DIM = 32


def build_default_model():
    return ConvSplice(DEFAULT_FEAT_DIM, DEFAULT_CL, DEFAULT_KERNEL_SIZES, DEFAULT_DILATION_RATES)
