import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class SpeechDataset(Dataset):
    """
    A dataset class with audio that cuts them/paddes them to a specified length, applies a Short-tome Fourier transform,
    normalizes and leads to a tensor.
    """

    def __init__(self, noisy_files, clean_files, n_fft=64, hop_length=16):
        super().__init__()
        # list of files
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)

        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.len_ = len(self.noisy_files)

        # fixed len
        self.max_len = 167700

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform

    def __getitem__(self, index):
        # print(self.clean_files[index], self.noisy_files[index])
        # load to tensors and normalization
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])

        # padding/cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)

        # conv시 마지막 0=mag, 1=phase
        #         x_real = x[..., 0]
        #         x_im = x[..., 1]
        # Short-time Fourier transform
        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft,
                                  hop_length=self.hop_length, normalized=True, return_complex=False)
        x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft,
                                  hop_length=self.hop_length, normalized=True, return_complex=False)

        # print(x_noisy.shape, torch.abs(x_noisy_stft).shape)
        #
        # def to_mag_phase(complex):
        #
        #     return torch.cat([
        #         torch.abs(complex).unsqueeze(2),
        #         torch.angle(complex).unsqueeze(2)
        #     ], dim=2)
        #
        # x_noisy_stft = to_mag_phase(x_noisy_stft)
        # x_clean_stft = to_mag_phase(x_clean_stft)

        return x_noisy_stft, x_clean_stft

    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]

        output = np.zeros((1, self.max_len), dtype='float32')
        output[0, -current_len:] = waveform[0, :self.max_len]
        output = torch.from_numpy(output)

        return output


class SpeechInferenceDataset(Dataset):
    """
    A dataset class with audio that cuts them/paddes them to a specified length, applies a Short-tome Fourier transform,
    normalizes and leads to a tensor.
    """

    def __init__(self, noisy_files, n_fft=64, hop_length=16):
        super().__init__()
        # list of files
        self.noisy_files = sorted(noisy_files)

        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.len_ = len(self.noisy_files)

        # fixed len
        self.max_len = 167700

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform

    def __getitem__(self, index):
        x_noisy = self.load_sample(self.noisy_files[index])
        x_noisy = self._prepare_sample(x_noisy)

        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft,
                                  hop_length=self.hop_length, normalized=True, return_complex=False)

        return x_noisy_stft, self.noisy_files[index]

    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]

        output = np.zeros((1, self.max_len), dtype='float32')
        output[0, -current_len:] = waveform[0, :self.max_len]
        output = torch.from_numpy(output)

        return output