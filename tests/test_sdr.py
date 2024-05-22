import os

import librosa
import torch
from torchmetrics.audio import SignalDistortionRatio

from tests.abstract_test import AbstractTest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelTestSDR(AbstractTest):
    def __init__(self, hp):
        self.hp = hp
        self.sdr = SignalDistortionRatio().to(device)

    def test(self, data):
        clean_wav, noise_wav = data

        clean_wav = clean_wav.to(device)
        noise_wav = noise_wav.to(device)

        self.sdr.update(clean_wav, noise_wav)
        sdr = self.sdr.compute().item()
        self.sdr.reset()

        return sdr

    def get_test_name(self):
        return "sdr"

