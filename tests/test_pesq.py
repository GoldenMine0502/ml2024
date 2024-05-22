import os

import librosa
import torch
from pesq import pesq
# from torchmetrics.audio import SignalDistortionRatio, PerceptualEvaluationSpeechQuality

from tests.abstract_test import AbstractTest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelTestPESQ(AbstractTest):
    def __init__(self, hp):
        self.hp = hp
        # self.pesq = PerceptualEvaluationSpeechQuality(self.hp.audio.sample_rate, 'wb')

        # pesq does not support gpu
        # self.pesq = self.pesq.to(device)

    def test(self, data):
        clean_wav, noise_wav = data

        clean_wav = clean_wav.cpu().detach().numpy()
        noise_wav = noise_wav.cpu().detach().numpy()

        # clean_wav = clean_wav.to(device)
        # noise_wav = noise_wav.to(device)

        # self.pesq.update(clean_wav, noise_wav)
        value = pesq(self.hp.audio.sample_rate, clean_wav, noise_wav, 'wb')
        # self.pesq.reset()

        return value

    def get_test_name(self):
        return "pesq"