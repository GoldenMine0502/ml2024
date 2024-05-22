import librosa
import numpy as np
import torch
import torch.nn.functional as F

class Audio():
    def __init__(self, hp):
        self.hp = hp

    def wav2spec(self, y):  # wav to complex-valued spectrum : [spec_real, spec_imag]
        D = self.stft(y)
        r, i = D.real, D.imag
        return np.array([r, i])

    def magNorm_torch(self, mag):
        mag_clip = self.amp_to_db(mag) - self.hp.audio.ref_level_db
        mag_norm = self.normalize(mag_clip)
        return mag_norm

    def amp_to_db(self, x):
        return 20.0 * torch.log10(torch.maximum(torch.Tensor([0.00001].cuda()), x))

    def db_to_amp(self, x):
        return torch.pow(10.0, x * 0.05)

    def normalize(self, S):
            return torch.clip(S / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (torch.clip(S, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.hp.audio.n_fft,
                            hop_length=self.hp.audio.hop_length,
                            win_length=self.hp.audio.win_length)

    # def spec2wav(self, stft_matrix):  # mag, phase : [batch, frequency, time_frame]
    #     # est_wav = torch.istft(stft_matrix, hop_length=self.hp.audio.hop_length, win_length=self.hp.audio.win_length, n_fft=self.hp.audio.n_fft)
    #     est_wav = self.istft(stft_matrix, hop_length=self.hp.audio.hop_length, win_length=self.hp.audio.win_length)
    #     return est_wav


    # def spec2wav(self, mag, phase):  # mag, phase : [batch, frequency, time_frame]
    #     spec_real = mag * torch.cos(phase)
    #     spec_imag = mag * torch.sin(phase)
    #     stft_matrix = torch.stack([spec_real, spec_imag], dim=-1)
    #     est_wav = torch.istft(stft_matrix, hop_length=self.hp.audio.hop_length, win_length=self.hp.audio.win_length, n_fft=self.hp.audio.n_fft)
    #     # est_wav = self.istft(stft_matrix, hop_length=self.hp.audio.hop_length, win_length=self.hp.audio.win_length)
    #     return est_wav

    def spec2wav(self, mag, phase):  # mag, phase : [batch, frequency, time_frame]
        spec_real = mag * torch.cos(phase)
        spec_imag = mag * torch.sin(phase)
        # input (Tensor) –
        #
        # The input tensor.
        # Expected to be in the format of stft(), output.
        # That is a complex tensor of shape (channel, fft_size, n_frame),
        # where the channel dimension is optional.
        # Changed in version 2.0: Real datatype inputs are no longer supported.
        # Input must now have a complex datatype, as returned by stft(..., return_complex=True).
        # stft_matrix = torch.stack([spec_real, spec_imag], dim=-1)
        # print(mag.shape, phase.shape, stft_matrix.shape)
        stft_matrix = torch.complex(spec_real, spec_imag)
        est_wav = torch.istft(stft_matrix,
                              hop_length=self.hp.audio.hop_length,
                              win_length=self.hp.audio.win_length,
                              n_fft=self.hp.audio.n_fft,
                              # onesided=False,
                              # return_complex=True
                              )
        # est_wav = self.istft(stft_matrix, hop_length=self.hp.audio.hop_length, win_length=self.hp.audio.win_length)
        return est_wav


    def spec2magphase_torch(self, spec, power=1. ):  #
        mag = spec.pow(2).sum(1).pow(power/2)
        phase = torch.atan2(spec[:, 1, ...], spec[:, 0, ...])
        return mag, phase

    '''
    def spec2magphase_torch_batch(self, spec, power=1. ):  #
        mag = spec.pow(2).sum(1).pow(power/2)
        phase = torch.atan2(spec[:, 1, ...], spec[:, 0, ...])
        return mag, phase
    '''

    def istft(self, stft_matrix, hop_length=None, win_length=None, window='hann',
              center=True, normalized=False, onesided=True, length=None):
        # keunwoochoi's implementation
        # https://gist.github.com/keunwoochoi/2f349e72cc941f6f10d4adf9b0d3f37e

        """stft_matrix = (batch, freq, time, complex)
        All based on librosa
            - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
        What's missing?
            - normalize by sum of squared window --> do we need it here?
            Actually the result is ok by simply dividing y by 2.
        """
        assert normalized == False
        assert onesided == True
        assert window == "hann"
        assert center == True

        device = stft_matrix.device
        n_fft = 2 * (stft_matrix.shape[-3] - 1)

        batch = stft_matrix.shape[0]

        # By default, use the entire frame
        if win_length is None:
            win_length = n_fft

        if hop_length is None:
            hop_length = int(win_length // 4)

        istft_window = torch.hann_window(n_fft).to(device).view(1, -1)  # (batch, freq)

        n_frames = stft_matrix.shape[-2]
        expected_signal_len = n_fft + hop_length * (n_frames - 1)

        y = torch.zeros(batch, expected_signal_len, device=device)
        for i in range(n_frames):
            sample = i * hop_length
            spec = stft_matrix[:, :, i]
            iffted = torch.irfft(spec, signal_ndim=1, signal_sizes=(win_length,))

            ytmp = istft_window * iffted
            y[:, sample:(sample + n_fft)] += ytmp

        y = y[:, n_fft // 2:]

        if length is not None:
            if y.shape[1] > length:
                y = y[:, :length]
            elif y.shape[1] < length:
                y = F.pad(y, (0, length - y.shape[1]))
                # y = torch.cat((y[:, :length], torch.zeros(y.shape[0], length - y.shape[1])))

        coeff = n_fft / float(
            hop_length) / 2.0  # -> this might go wrong if curretnly asserted values (especially, `normalized`) changes.
        return y / coeff
