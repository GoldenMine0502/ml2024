import os
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.audio import Audio


def read_ctrl(file_path):
    li = []
    with open(file_path, 'rt') as f:
        for file in f:
            li.append(file.strip())

    return li


class DataLoaderVCTK:
    def __init__(self, hp, model_name, batch_size=None, shuffle=None, num_workers=None):
        self.audio = Audio(hp)
        self.trainloader = DataLoader(dataset=DatasetVCTK(hp, True),
                                      batch_size=hp.train.batch_size if batch_size is None else batch_size,
                                      shuffle=True if shuffle is None else shuffle,
                                      num_workers=hp.train.num_workers if num_workers is None else num_workers,
                                      collate_fn=lambda x: self.collate_train(x),
                                      pin_memory=True,
                                      drop_last=True,
                                      sampler=None)
        self.validationloader = DataLoader(dataset=DatasetVCTK(hp, False),
                                           collate_fn=lambda x: self.collate_validate(x),
                                           batch_size=hp.train.batch_size if batch_size is None else batch_size,
                                           shuffle=False,
                                           num_workers=hp.train.num_workers if num_workers is None else num_workers)

        self.inferenceloader = DataLoader(dataset=DatasetVCTKInference(hp),
                                          collate_fn=lambda x: self.collate_inference(x),
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=hp.train.num_workers if num_workers is None else num_workers)

        self.testloader = DataLoader(dataset=DatasetVCTKTest(hp, model_name),
                                     collate_fn=lambda x: self.collate_test(x),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=hp.train.num_workers if num_workers is None else num_workers)

    def collate_train(self, batch):
        target_wav_list = list()
        mixed_wav_list = list()
        mixed_spec_list = list()
        target_spec_list = list()

        for target_wav, mixed_wav in batch:
            target_wav_list.append(target_wav)
            mixed_wav_list.append(mixed_wav)

        target_wav_list = torch.nn.utils.rnn.pad_sequence(target_wav_list, batch_first=True)
        mixed_wav_list = torch.nn.utils.rnn.pad_sequence(mixed_wav_list, batch_first=True)

        for mixed_wav, target_wav in zip(mixed_wav_list, target_wav_list):
            mixed_spec = self.audio.wav2spec(np.array(mixed_wav))
            mixed_spec = torch.from_numpy(mixed_spec).float()
            mixed_spec_list.append(mixed_spec)

            target_spec = self.audio.wav2spec(np.array(target_wav))
            target_spec = torch.from_numpy(target_spec).float()
            target_spec_list.append(target_spec)

        mixed_spec_list = torch.stack(mixed_spec_list, dim=0)
        target_spec_list = torch.stack(target_spec_list, dim=0)

        return target_wav_list, mixed_wav_list, target_spec_list, mixed_spec_list

    def collate_validate(self, batch):
        return batch

    def collate_inference(self, batch):
        return batch

    def collate_test(self, batch):
        return batch


class DatasetVCTKTest(Dataset):
    def __init__(self, hp, model_name):
        self.hp = hp
        self.output_dir = os.path.join(self.hp.data.output_dir, model_name)
        self.test_target_dir = hp.data.test_target_dir
        self.enhanced_wav_list = None
        self.mixed_wav_list = None
        pass

    def _file_list(self):
        INFERENCE_NOISY_DIR = Path('../dataset/wav16k')
        mixed = sorted(list(INFERENCE_NOISY_DIR.rglob('*.wav')))

        INFERENCE_CLEAN_DIR = Path('../../../dataset/wav16k_clean')
        enhanced = sorted(list(INFERENCE_CLEAN_DIR.rglob('*.wav')))

        return mixed, enhanced

    def __len__(self):
        return len(self.mixed_wav_list)

    def __getitem__(self, index):
        if self.enhanced_wav_list is None:
            self.mixed_wav_list = self._file_list()[0]
            self.enhanced_wav_list = self._file_list()[1]

        mixed, _ = librosa.load(self.mixed_wav_list[index], sr=self.hp.audio.sample_rate)
        enhanced, _ = librosa.load(self.enhanced_wav_list[index], sr=self.hp.audio.sample_rate)

        if len(mixed) >= len(enhanced):
            mixed = mixed[0:len(enhanced)]
        else:
            enhanced = enhanced[0:len(mixed)]

        return mixed, enhanced


class DatasetVCTKInference(Dataset):
    def __init__(self, hp):
        self.hp = hp
        self.audio = Audio(hp)
        self.test_mixed_dir = hp.data.test_mixed_dir
        self.test_target_dir = hp.data.test_target_dir
        self.test_mixed_wav_list = self._file_list()

    def _file_list(self):
        INFERENCE_NOISY_DIR = Path('../dataset/wav16k')
        inference_noisy_files = sorted(list(INFERENCE_NOISY_DIR.rglob('*.wav')))

        return inference_noisy_files

    def __len__(self):
        return len(self.test_mixed_wav_list)

    def __getitem__(self, index):
        test_wav = librosa.load(self.test_mixed_wav_list[index], sr=self.hp.audio.sample_rate)
        test_wav = torch.from_numpy(test_wav).float()

        length = test_wav.shape[0]

        mixed_wav_padding = np.concatenate([test_wav, np.zeros(self.hp.train.max_audio_len - test_wav.shape[0])],
                                           axis=0)
        mixed_spec_padding = self.audio.wav2spec(mixed_wav_padding)
        mixed_spec_padding = torch.from_numpy(mixed_spec_padding).float()

        return self.test_mixed_wav_list[index], length, test_wav, mixed_wav_padding, mixed_spec_padding


class DatasetVCTK(Dataset):
    def __init__(self, hp, train):
        self.hp = hp
        self.train = train
        self.data_dir = hp.data.mixed_dir
        self.target_dir = hp.data.target_dir
        self.ctrl = hp.data.train_ctrl if train else hp.data.validate_ctrl
        # self.ctrl_dir = hp.data.train_ctrl_dir if train else hp.data.test_ctrl_dir

        new_lists = self._filelist()
        self.target_wav_list = new_lists[0]
        self.mixed_wav_list = new_lists[1]
        self.ctrl_list = new_lists[2]

        self.audio = Audio(hp)

        assert len(self.target_wav_list) == len(self.mixed_wav_list), "number of training files must match"

    def _filelist(self):
        lines = read_ctrl(self.ctrl)

        new_target_wav = [(self.target_dir + x + self.hp.form.target.wav) for x in lines]
        new_mixed_wav = [(self.data_dir + x + self.hp.form.mixed.wav) for x in lines]

        return [new_target_wav, new_mixed_wav, lines]

    def __len__(self):
        return len(self.target_wav_list)

    def __getitem__(self, idx):
        target_wav, _ = librosa.load(self.target_wav_list[idx], sr=self.hp.audio.sample_rate)
        mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr=self.hp.audio.sample_rate)

        if self.train:  # Train set
            target_wav = torch.from_numpy(target_wav).float()
            mixed_wav = torch.from_numpy(mixed_wav).float()

            return target_wav, mixed_wav
        else:  # Valid set
            mixed_wav_padding = mixed_wav.copy()
            target_spec = self.audio.wav2spec(target_wav)

            # padding mixed wav
            if mixed_wav.shape[0] < self.hp.train.min_audio_len:
                mixed_wav_padding = np.concatenate(
                    [mixed_wav, np.zeros(self.hp.train.min_audio_len - mixed_wav.shape[0])], axis=0)

            # fft
            mixed_spec = self.audio.wav2spec(mixed_wav)
            mixed_spec_padding = self.audio.wav2spec(mixed_wav_padding)

            # send to cuda
            target_wav = torch.from_numpy(target_wav).float()
            target_spec = torch.from_numpy(target_spec).float()
            mixed_spec = torch.from_numpy(mixed_spec).float()

            mixed_wav = torch.from_numpy(mixed_wav).float()
            mixed_spec_padding = torch.from_numpy(mixed_spec_padding).float()

            return target_wav, mixed_wav, target_spec, mixed_spec, mixed_spec_padding
