from layers.dcunet_layers import *
import torch


class Dcunet(nn.Module):
    def __init__(self, audio, is_complex=True, is_attention=True):
        super().__init__()
        self.set_param()
        self.audio = audio
        self.is_attention = is_attention

        self.encoders = []
        self.attentions = []
        self.model_len = len(self.enc_channels)-1

        for i in range(self.model_len):
            module = Encoder(in_channels=self.enc_channels[i], out_channels=self.enc_channels[i+1], kernel_size=self.enc_kernel[i],
                             isComplex=is_complex, stride=self.enc_stride[i], padding=self.enc_padding[i])
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

            if i == self.model_len-1:
                break

            if is_attention:
                att = ComplexAttention(self.enc_channels[self.model_len-i-1], self.feature_shape[i])
                self.add_module("attention{}".format(i), att)
                self.attentions.append(att)

        self.decoders = []
        for i in range(self.model_len-1):
            in_channel = self.dec_channels[i] + self.enc_channels[self.model_len - i]
            module = Decoder(in_channels=in_channel, out_channels=self.dec_channels[i+1], isComplex=is_complex,
                             kernel_size=self.dec_kernel[i], stride=self.dec_stride[i], padding=self.dec_padding[i],
                             output_padding=self.dec_output_padding[i])
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        i = self.model_len-1
        module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_len - i],
                         out_channels=self.dec_channels[i + 1],
                         kernel_size=self.dec_kernel[i], stride=self.dec_stride[i], padding=self.dec_padding[i],
                         output_padding=self.dec_output_padding[i], isComplex=is_complex, isLast=True)
        self.add_module("decoder{}".format(i), module)
        self.decoders.append(module)

    def forward(self, spec):
        """
        Forward pass of generator.
        Args:
            x: input batch (signal)
        """
        # spec : [B, 2, frequency, time_frame]
        x = spec.unsqueeze(2)  # add channel : [B, 2, 1, frequency, time_frame]
        xs = []

        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            x = encoder(x)

        p = x.clone()

        for i, decoder in enumerate(self.decoders):
            p = decoder(p)

            ## batch zero padding ##

            if p.shape[-1] < xs[self.model_len - i - 1].shape[-1]:
                # print("dec : ", p.shape)
                # print("enc : ", xs[self.model_len - i - 1].shape)
                gap = xs[self.model_len - i - 1].shape[-1] - p.shape[-1]
                p = torch.cat([p, torch.zeros(p.shape[0], p.shape[1], p.shape[2], p.shape[3], gap).cuda()], dim=4)

            elif p.shape[-1] > xs[self.model_len - i - 1].shape[-1]:
                # print("dec : ", p.shape)
                # print("enc : ", xs[self.model_len - i - 1].shape)
                p = p[:, :, :, :, :xs[self.model_len - i - 1].shape[-1]]


            if i == self.model_len -1:   # Skip-connection is not applied to the last layer
                break

            if self.is_attention:  # Skip-connection is applied attention
                att = self.attentions[i](xs[self.model_len - i - 1], p)

                ## DCUNET-ATT
                p = torch.cat([p, xs[self.model_len - i - 1] * att], dim=2)

                ## DCUNET TFSA DE / SkipConv / SDAB
                #p = torch.cat([p, att], dim=2)

            else:  # Skip-connection (DCUNET)
                p = torch.cat([p, xs[self.model_len - i - 1]], dim=2)

        mask = p.squeeze(2)

        return mask

    def set_param(self, input_channels=1):
        self.enc_channels = [input_channels, 45, 45, 90, 90, 90, 90, 90, 90]
        self.enc_kernel = [(7, 5), (7, 5), (7, 5), (5, 3), (5, 3), (5, 3), (5, 3), (5, 1)]
        self.enc_stride = [(2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1)]
        self.enc_padding = [(3, 0), (3, 0), (3, 0), (2, 0), (2, 0), (2, 0), (2, 0), (2, 0)]

        self.dec_channels = [0, 90, 90, 90, 90, 90, 45, 45, 1]
        self.dec_kernel = [(5, 1), (5, 3), (5, 3), (5, 3), (5, 3), (7, 5), (7, 5), (7, 5)]
        self.dec_stride = [(2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2)]
        self.dec_padding = [(2, 0), (2, 0), (2, 0), (2, 0), (2, 0), (3, 0), (3, 0), (3, 0)]
        self.dec_output_padding = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]

        self.feature_shape = [(5, 55), (9, 112), (17, 114), (33, 230), (65, 232), (129, 467), (257, 471)]

