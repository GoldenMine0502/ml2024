import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SpeechDataset, SpeechInferenceDataset


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     device_name = "MPS"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    return device, device_name


DEVICE, _ = get_device()
SAMPLE_RATE = 16000
N_FFT = SAMPLE_RATE * 64 // 1000 + 4
HOP_LENGTH = SAMPLE_RATE * 16 // 1000 + 4

TRAIN_NOISY_DIR = Path('dataset/noisy_trainset_56spk_wav')
TRAIN_CLEAN_DIR = Path('dataset/clean_trainset_56spk_wav')

TEST_NOISY_DIR = Path('dataset/noisy_testset_wav')
TEST_CLEAN_DIR = Path('dataset/clean_testset_wav')

INFERENCE_NOISY_DIR = Path('dataset/wav16k')
INFERENCE_CLEAN_DIR = Path('dataset/wav16k_clean')

train_noisy_files = sorted(list(TRAIN_NOISY_DIR.rglob('*.wav')))
train_clean_files = sorted(list(TRAIN_CLEAN_DIR.rglob('*.wav')))

test_noisy_files = sorted(list(TEST_NOISY_DIR.rglob('*.wav')))
test_clean_files = sorted(list(TEST_CLEAN_DIR.rglob('*.wav')))

inference_noisy_files = sorted(list(INFERENCE_NOISY_DIR.rglob('*.wav')))

test_dataset = SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)
train_dataset = SpeechDataset(train_noisy_files, train_clean_files, N_FFT, HOP_LENGTH)
inference_dataset = SpeechInferenceDataset(inference_noisy_files, N_FFT, HOP_LENGTH)

inference_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)



EPOCH = 2


if not os.path.exists(INFERENCE_CLEAN_DIR):
    os.mkdir(INFERENCE_CLEAN_DIR)
