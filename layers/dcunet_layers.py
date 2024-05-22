from torch import nn
from layers import complex_nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, isComplex=True,
                 padding_mode="zeros"):
        super().__init__()

        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding

        if isComplex:
            conv = complex_nn.ComplexConv2d
            bn = complex_nn.ComplexBatchNorm2d
            act = complex_nn.ComplexActivation
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
            act = nn.LeakyReLU

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), output_padding=(0, 0),
                 isComplex=True, isLast=False):
        super().__init__()

        if isComplex:
            tconv = complex_nn.ComplexConvTranspose2d
            bn = complex_nn.ComplexBatchNorm2d
            act = complex_nn.ComplexActivation
            self.act = act(isLast=isLast)

        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
            act = nn.LeakyReLU
            self.act = act()

        self.transconv = tconv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, output_padding=output_padding)
        self.bn = bn(out_channels)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ComplexAttention(nn.Module):
    def __init__(self, channel, feature_shape):
        super().__init__()
        # DCUNET_28k-ATT
        self.att = complex_nn.SkipAttention(channel, feature_shape[0], feature_shape[1])

        # SkipConv
        # self.att = complex_nn.SkipConv(channel, feature_shape[0], feature_shape[1])

        # DCUNET_28k TFSA DE
        # self.att = complex_nn.SelfAttention(channel, feature_shape[0], feature_shape[1])

    def forward(self, Q, K):
        res = self.att(Q, K)  # [batch, 2, channel, frequency, time_frame]
        return res