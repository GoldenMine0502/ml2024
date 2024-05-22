import torch
import torch.nn as nn


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, isGAP=False,**kwargs):
        super().__init__()
        ## Model components
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,2,channel, Frequency,Frame]
        part_r = x[:, 0, ...]
        part_i = x[:, 1, ...]

        real = self.conv_re(part_r) - self.conv_im(part_i) # 실수
        imaginary = self.conv_re(part_i) + self.conv_im(part_r) # 허수

        # Concatenates a sequence of tensors along a new dimension.
        output = torch.stack((real, imaginary), dim=1)
        return output


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, isGAP=False,**kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,2,channel, Frequency,Frame]
        part_r = x[:, 0, ...]
        part_i = x[:, 1, ...]

        real = self.conv_re(part_r) - self.conv_im(part_i)
        imaginary = self.conv_re(part_i) + self.conv_im(part_r)
        output = torch.stack((real, imaginary), dim=1)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):  # shpae of x : [batch,2,channel, frequency, frame]
        real = self.tconv_re(x[:, 0, ...]) - self.tconv_im(x[:, 1, ...])
        imaginary = self.tconv_re(x[:, 1, ...]) + self.tconv_im(x[:, 0, ...])
        output = torch.stack((real, imaginary), dim=1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[:, 0, ...])
        imag = self.bn_im(x[:, 1, ...])
        output = torch.stack((real, imag), dim=1)
        return output


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm1d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm1d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[:, 0, ...])
        imag = self.bn_im(x[:, 1, ...])
        output = torch.stack((real, imag), dim=1)
        return output


class ComplexActivation(nn.Module):
    def __init__(self, isLast=False):
        super().__init__()
        if isLast:
            self.act_real = nn.Tanh()
            self.act_imag = nn.Tanh()

        else:
            self.act_real = nn.LeakyReLU()
            self.act_imag = nn.LeakyReLU()

    def forward(self, x):
        real = self.act_real(x[:, 0, ...])
        imag = self.act_imag(x[:, 1, ...])
        output = torch.stack((real, imag), dim=1)

        return output


# DCUNET-ATT
class SkipAttention(nn.Module):

    # self.feature_shape = [(5, 55), (9, 112), (17, 114), (33, 230), (65, 232), (129, 467), (257, 471)]
    # frequency는 feature_shape의 앞자리
    # frame는 feature_shape의 뒷자리
    def __init__(self, channel, frequency, frame):
        super().__init__()

        self.p_conv = ComplexConv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))
        self.q_conv = ComplexConv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))

        self.conv = ComplexConv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))

        self.p_bn = ComplexBatchNorm2d(channel)
        self.q_bn = ComplexBatchNorm2d(channel)

        self.bn = ComplexBatchNorm2d(channel)

        self.relu_real = nn.ReLU()
        self.relu_imag = nn.ReLU()

        self.sigmoid_real = nn.Sigmoid()
        self.sigmoid_imag = nn.Sigmoid()

    # att = self.attentions[i](xs[self.model_len - i - 1], p)
    # P는 이전 인코더에서의 출력값
    # Q는 이전 디코더에서의 출력값
    def forward(self, P, Q):
        real_p = torch.abs(P[:, 0, ...])
        imag_p = torch.abs(P[:, 1, ...])
        p = torch.stack([real_p, imag_p], dim=1)

        real_q = torch.abs(Q[:, 0, ...])
        imag_q = torch.abs(Q[:, 1, ...])
        q = torch.stack([real_q, imag_q], dim=1)

        p = self.p_conv(p)
        p = self.p_bn(p)
        q = self.q_conv(q)
        q = self.q_bn(q)

        att = p + q

        real_r = self.relu_real(att[:, 0, ...])
        imag_r = self.relu_imag(att[:, 1, ...])
        att = torch.stack((real_r, imag_r), dim=1)

        att = self.conv(att)
        att = self.bn(att)

        real = self.sigmoid_real(att[:, 0, ...])
        imag = self.sigmoid_imag(att[:, 1, ...])
        output = torch.stack((real, imag), dim=1)
        return output


# SkipConvNet
class SkipConv(nn.Module):
    def __init__(self, channel, frequency, frame):
        super().__init__()
        self.TimeAxis_R = SkipConvModule(channel, frequency, frame)
        self.TimeAxis_I = SkipConvModule(channel, frequency, frame)

    def forward(self, P, Q):  # P: Encoder, Q: Decoder
        # shape: [batch, 2, channel, Frequency, Frame]
        SC_R = self.TimeAxis_R(P[:, 0, ...])
        SC_I = self.TimeAxis_I(P[:, 1, ...])
        SC = torch.stack((SC_R, SC_I), dim=1)

        return SC


class SkipConvModule(nn.Module):
    def __init__(self, channel, frequency, frame):
        super().__init__()
        self.relu = nn.ReLU()
        self.freq = frequency

        self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))

        '''
        if frequency < 5:
            self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(frequency, frequency), padding=(frequency//2, frequency//2))
        else:
            self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(5, 5), padding=(2, 2))
        '''

        self.bn = nn.BatchNorm2d(channel)

    def forward(self, P):
        previous = self.relu(P)

        att = self.conv(previous)

        att = att[:, :, :previous.shape[2], :previous.shape[3]]

        att_output = previous + att

        att_output = self.bn(att_output)

        return att_output


# Skip TFSA DE
class SelfAttention(nn.Module):
    def __init__(self, channel, frequency, frame, isComplex=False):
        super().__init__()
        self.isComplex = isComplex
        if isComplex:
            self.TimeAxisSA = SelfAttentionModule_Time(channel, frequency, frame, isComplex=isComplex)
            self.FreqAxisSA = SelfAttentionModule_Freq(channel, frequency, frame, isComplex=isComplex)

        else:
            self.TimeAxisSA_R = SelfAttentionModule_Time(channel, frequency, frame, isComplex=isComplex)
            self.TimeAxisSA_I = SelfAttentionModule_Time(channel, frequency, frame, isComplex=isComplex)

            self.FreqAxisSA_R = SelfAttentionModule_Freq(channel, frequency, frame, isComplex=isComplex)
            self.FreqAxisSA_I = SelfAttentionModule_Freq(channel, frequency, frame, isComplex=isComplex)

    def forward(self, P, Q):  # P: Encoder, Q: Decoder
        # shape: [batch, 2, channel, Frequency, Frame]
        if self.isComplex:
            SA_T = self.TimeAxisSA(P, Q)
            SA_F = self.FreqAxisSA(P, Q)

            SA_output = SA_T + SA_F + P

        else:
            SA_T_R = self.TimeAxisSA_R(P[:, 0, ...], Q[:, 0, ...])
            SA_T_I = self.TimeAxisSA_I(P[:, 1, ...], Q[:, 1, ...])
            SA_T = torch.stack((SA_T_R, SA_T_I), dim=1)

            SA_F_R = self.FreqAxisSA_R(P[:, 0, ...], Q[:, 0, ...])
            SA_F_I = self.FreqAxisSA_I(P[:, 1, ...], Q[:, 1, ...])
            SA_F = torch.stack((SA_F_R, SA_F_I), dim=1)

            SA_output = SA_T + SA_F + P

        return SA_output


class SelfAttentionModule_Time(nn.Module):
    def __init__(self, channel, frequency, frame, isComplex=False):
        super().__init__()
        self.isComplex = isComplex
        if isComplex:
            self.query = ComplexConv2d(in_channels=channel, out_channels=channel, kernel_size=1)
            self.key = ComplexConv2d(in_channels=channel, out_channels=channel, kernel_size=1)
            self.value = ComplexConv2d(in_channels=channel, out_channels=channel, kernel_size=1)

            self.q_bn = ComplexBatchNorm2d(channel)
            self.k_bn = ComplexBatchNorm2d(channel)
            self.v_bn = ComplexBatchNorm2d(channel)

            self.q_act = ComplexActivation()
            self.k_act = ComplexActivation()
            self.v_act = ComplexActivation()

        else:
            self.query = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))
            self.key = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))
            self.value = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))

            self.q_bn = nn.BatchNorm2d(channel)
            self.k_bn = nn.BatchNorm2d(channel)
            self.v_bn = nn.BatchNorm2d(channel)

            self.q_act = nn.LeakyReLU()
            self.k_act = nn.LeakyReLU()
            self.v_act = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=2)

    def forward(self, P, Q):
        query = self.q_act(self.q_bn(self.query(Q)))
        key = self.k_act(self.k_bn(self.key(P)))
        value = self.v_act(self.v_bn(self.value(P)))

        # Time axis self attention
        if self.isComplex:
            # [ B, 2, C, F, T ]
            query_T = query.view([query.shape[0], query.shape[1], -1, query.shape[-1]])
            key_T = key.view([key.shape[0], key.shape[1], -1, key.shape[-1]])
            value_T = value.view([value.shape[0], value.shape[1], -1, value.shape[-1]])

            # [ B, 2, CxF, T ]
            query_T = query_T.transpose(2, 3)
            key_T = key_T.transpose(2, 3)
            value_T = value_T.transpose(2, 3)

            # [ B, 2, T, CxF ]
            att_weight_r = torch.matmul(query_T[:, 0, ...], key_T.transpose(2, 3)[:, 0, ...]) + torch.matmul(
                query_T[:, 1, ...], key_T.transpose(2, 3)[:, 1, ...])
            att_weight_i = torch.matmul(query_T[:, 1, ...], key_T.transpose(2, 3)[:, 0, ...]) - torch.matmul(
                query_T[:, 0, ...], key_T.transpose(2, 3)[:, 1, ...])

            att_weight_abs = torch.sqrt(torch.pow(att_weight_r, 2) + torch.pow(att_weight_i, 2))
            att_weight = self.softmax(att_weight_abs)

            att_output_r = torch.matmul(att_weight, value_T[:, 0, ...])
            att_output_i = torch.matmul(att_weight, value_T[:, 1, ...])

            att_output = torch.stack((att_output_r, att_output_i), dim=1)

            # [ B, 2, T, CxF ]
            att_output = att_output.transpose(2, 3)    # [ B, 2, CxF, T ]
            att_output = att_output.view(P.shape)  # [B, 2, C, F, T]

        else:
            # [B, C, F, T]
            query_T = query.view([query.shape[0], -1, query.shape[-1]])
            key_T = key.view([key.shape[0], -1, key.shape[-1]])
            value_T = value.view([value.shape[0], -1, value.shape[-1]])

            # [B, CxF, T]
            query_T = query_T.transpose(1, 2)
            key_T = key_T.transpose(1, 2)
            value_T = value_T.transpose(1, 2)

            # [B, T, CxF]
            att_weight = torch.matmul(query_T, key_T.transpose(1, 2))
            att_weight = self.softmax(att_weight)
            att_output = torch.matmul(att_weight, value_T)

            # [B, T, CxF]
            att_output = att_output.transpose(1, 2)    # [B, CxF, T]
            att_output = att_output.view(P.shape)      # [B, C, F, T]

        return att_output


class SelfAttentionModule_Freq(nn.Module):
    def __init__(self, channel, frequency, frame, isComplex=False):
        super().__init__()
        self.isComplex = isComplex

        if isComplex:
            self.query = ComplexConv2d(in_channels=channel, out_channels=channel, kernel_size=1)
            self.key = ComplexConv2d(in_channels=channel, out_channels=channel, kernel_size=1)
            self.value = ComplexConv2d(in_channels=channel, out_channels=channel, kernel_size=1)

            self.q_bn = ComplexBatchNorm2d(channel)
            self.k_bn = ComplexBatchNorm2d(channel)
            self.v_bn = ComplexBatchNorm2d(channel)

            self.q_act = ComplexActivation()
            self.k_act = ComplexActivation()
            self.v_act = ComplexActivation()

        else:
            self.query = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))
            self.key = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))
            self.value = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))

            self.q_bn = nn.BatchNorm2d(channel)
            self.k_bn = nn.BatchNorm2d(channel)
            self.v_bn = nn.BatchNorm2d(channel)

            self.q_act = nn.LeakyReLU()
            self.k_act = nn.LeakyReLU()
            self.v_act = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=2)

    def forward(self, P, Q):

        query = self.q_act(self.q_bn(self.query(Q)))
        key = self.k_act(self.k_bn(self.key(P)))
        value = self.v_act(self.v_bn(self.value(P)))

        # Freq axis self attention
        if self.isComplex:
            # [B, 2, C, F, T]
            query_F = query.transpose(2, 3)
            key_F = key.transpose(2, 3)
            value_F = value.transpose(2, 3)

            # [B, 2, F, C, T]
            query_F = query_F.contiguous().view(
                [query.shape[0], query.shape[1], query.shape[3], query.shape[2] * query.shape[4]])
            key_F = key_F.contiguous().view(
                [key.shape[0], key.shape[1], key.shape[3], key.shape[2] * key.shape[4]])
            value_F = value_F.contiguous().view(
                [value.shape[0], value.shape[1], value.shape[3], value.shape[2] * value.shape[4]])

            # [B, 2, F, CxT]
            att_weight_r = torch.matmul(query_F[:, 0, ...], key_F.transpose(2, 3)[:, 0, ...]) \
                           + torch.matmul(query_F[:, 1, ...], key_F.transpose(2, 3)[:, 1, ...])
            att_weight_i = torch.matmul(query_F[:, 1, ...], key_F.transpose(2, 3)[:, 0, ...]) \
                           - torch.matmul(query_F[:, 0, ...], key_F.transpose(2, 3)[:, 1, ...])

            att_weight_abs = torch.sqrt(torch.pow(att_weight_r, 2) + torch.pow(att_weight_i, 2))

            att_weight = self.softmax(att_weight_abs)

            att_output_r = torch.matmul(att_weight, value_F[:, 0, ...])
            att_output_i = torch.matmul(att_weight, value_F[:, 1, ...])
            att_output = torch.stack((att_output_r, att_output_i), dim=1)

            # [B, 2, F, CxT]
            att_output = att_output.view([P.shape[0], P.shape[1], P.shape[3], P.shape[2], P.shape[4]])
            # [B, 2, F, C, T]
            att_output = att_output.transpose(2, 3)  # [B, 2, C, F, T]

        else:
            # [B, C, F, T]
            query_F = query.transpose(1, 2)
            key_F = key.transpose(1, 2)
            value_F = value.transpose(1, 2)

            # [B, F, C, T]
            query_F = query_F.contiguous().view(
                [query.shape[0], query.shape[2], query.shape[1] * query.shape[3]])
            key_F = key_F.contiguous().view([key.shape[0], key.shape[2], key.shape[1] * key.shape[3]])
            value_F = value_F.contiguous().view(
                [value.shape[0], value.shape[2], value.shape[1] * value.shape[3]])

            # [B, F, CxT]
            att_weight = torch.matmul(query_F, key_F.transpose(1, 2))
            att_weight = self.softmax(att_weight)
            att_output = torch.matmul(att_weight, value_F)

            att_output = att_output.view([P.shape[0], P.shape[2], P.shape[1], P.shape[3]])
            # [B, F, C, T]
            att_output = att_output.transpose(1, 2)
            # [B, C, F, T]

        return att_output


# SDAB
class SDAB(nn.Module):


    def __init__(self, channel, frequency, frame):
        super().__init__()
        self.TimeAxis_R = SDAB_module(channel, frequency, frame, isTime=True)
        self.TimeAxis_I = SDAB_module(channel, frequency, frame, isTime=True)

        self.FreqAxis_R = SDAB_module(channel, frequency, frame)
        self.FreqAxis_I = SDAB_module(channel, frequency, frame)

        self.conv_r = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))
        self.conv_i = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1))

    def forward(self, P, Q):  # P: Encoder, Q: Decoder
        # shape: [batch, 2, channel, Frequency, Frame]
        T_R = self.TimeAxis_R(P[:, 0, ...], P[:, 0, ...])
        T_I = self.TimeAxis_I(P[:, 1, ...], P[:, 1, ...])
        T = torch.stack((T_R, T_I), dim=1)

        F_R = self.FreqAxis_R(P[:, 0, ...], P[:, 0, ...])
        F_I = self.FreqAxis_I(P[:, 1, ...], P[:, 1, ...])
        F = torch.stack((F_R, F_I), dim=1)

        output = T + F + P

        output_R = self.conv_r(output[:, 0, ...])
        output_I = self.conv_i(output[:, 1, ...])
        output_conv = torch.stack((output_R, output_I), dim=1)

        return output_conv


class SDAB_module(nn.Module):
    def __init__(self, channel, frequency, frame, isTime=False):
        super().__init__()
        self.isTime = isTime
        if isTime:
            self.fc = nn.Linear(frame, frame)
            self.bn = nn.BatchNorm1d(frame)

        else:
            self.fc = nn.Linear(frequency, frequency)
            self.bn = nn.BatchNorm1d(frequency)

        self.relu = nn.ReLU()

    def forward(self, P, Q):
        if self.isTime:
            # [B, C, F, T]
            P_T = P.view([P.shape[0], -1, P.shape[-1]])
            # [B, CxF, T]
            att = self.fc(P_T).transpose(1, 2)

            att = self.bn(att).transpose(1, 2)

            att = self.relu(att)
            att = att.view(P.shape)

        else:
            # [B, C, F, T]
            P_F = P.transpose(1, 2)
            # [B, F, C, T]
            P_F = P_F.contiguous().view([P.shape[0], P.shape[2], -1])
            # [B, F, CxT]
            P_F = P_F.transpose(1, 2)
            # [B, CxT, F]
            att = self.fc(P_F).transpose(1, 2)
            att = self.bn(att)
            att = self.relu(att)

            att = att.view([P.shape[0], P.shape[2], P.shape[1], P.shape[3]])
            att = att.transpose(1, 2)

        return att