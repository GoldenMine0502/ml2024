import os

import librosa
import torch
# from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from pystoi import stoi
from tests.abstract_test import AbstractTest

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelTestSTOI(AbstractTest):
    def __init__(self, hp):
        self.hp = hp
        # self.stoi = ShortTimeObjectiveIntelligibility(self.hp.audio.sample_rate, False)
        # stoi does not support gpu
        # self.stoi = self.stoi.to(device)

    def test(self, data):
        clean_wav, noise_wav = data


        clean_wav = clean_wav.cpu().detach().numpy()
        noise_wav = noise_wav.cpu().detach().numpy()

        # clean_wav = clean_wav.to(device)
        # noise_wav = noise_wav.to(device)

        if len(clean_wav) >= len(noise_wav):
            clean_wav = clean_wav[0:len(noise_wav)]
        else:
            noise_wav = noise_wav[0:len(clean_wav)]
        value = stoi(clean_wav, noise_wav, self.hp.audio.sample_rate, extended=False)

        # self.stoi.update(clean_wav, noise_wav)
        # stoi = self.stoi.compute().item()
        # self.stoi.reset()

        return value

    def get_test_name(self):
        return "stoi"
