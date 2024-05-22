import logging
import os
import sys

import numpy as np
import torch
from si_snr import SI_SNR
from tests.test_pesq import ModelTestPESQ
from tests.test_sdr import ModelTestSDR
from tests.test_stoi import ModelTestSTOI
from utils.audio import Audio
from torch import nn
from torchmetrics.audio import SignalDistortionRatio
from scipy.io.wavfile import write

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("DCUNET_HUBERT")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DcunetModel():
    def __init__(self, hp, model, flip=False):
        super().__init__()
        self.hp = hp
        self.audio = Audio(hp)
        self.model = model.to(device)
        self.flip = flip
        self.si_snr = SI_SNR()
        self.mse = nn.MSELoss()
        self.sdr = SignalDistortionRatio().to(device)

        self.tests = [ModelTestSDR(hp), ModelTestPESQ(hp), ModelTestSTOI(hp)]

    def train(self, train_data):
        target_wav, mixed_wav, target_spec, mixed_spec = train_data

        mixed_spec = mixed_spec.to(device)

        # 모델에 넣어가지고 마스크 얼마나 없어졌는지 구하는 작업
        mask = self.model(mixed_spec)
        return mask

    # return list of validation result
    def validate(self, validate_data):
        mixed_wav = None
        target_wav = None
        enhanced_wav = None

        results = []
        for data in validate_data:
            target_wav, mixed_wav, target_spec, mixed_spec, mixed_spec_padding = data

            target_wav = target_wav.to(device).unsqueeze(0)
            mixed_spec_padding = mixed_spec_padding.to(device).unsqueeze(0)
            target_spec = target_spec.to(device).unsqueeze(0)

            est_mask = self.model(mixed_spec_padding)

            mixed_mag, mixed_phase = self.audio.spec2magphase_torch(mixed_spec_padding)
            mask_mag, mask_phase = self.audio.spec2magphase_torch(est_mask)

            output_mag = mixed_mag * mask_mag
            output_phase = mixed_phase + mask_phase

            target_mag, _ = self.audio.spec2magphase_torch(target_spec)

            enhanced_wav = self.audio.spec2wav(output_mag, output_phase)

            if enhanced_wav.shape[1] >= target_wav.shape[1]:
                enhanced_wav = enhanced_wav[:, 0:target_wav.shape[1]]
            else:
                target_wav = target_wav[:, 0:enhanced_wav.shape[1]]

            test_loss = self.si_snr(target_wav, enhanced_wav)

            # 기본적으로 mixed_wav에 padding을 붙여서 MSE는 못씀
            # test_loss = self.si_snr(target_wav_torch, enhanced_wav) + 100 * self.mse(target_mag, output_mag)
            test_loss = test_loss.item()

            self.sdr.update(target_wav, enhanced_wav)
            sdr = self.sdr.compute().item()
            self.sdr.reset()

            results.append((test_loss, sdr))

        return results, mixed_wav, target_wav.squeeze(0), enhanced_wav.squeeze(0)

    def inference(self, test_data, output_dir):
        file_path, length, mixed_wav, mixed_wav_padding, mixed_spec_padding = test_data

        mixed_spec_padding = mixed_spec_padding.to(device).unsqueeze(0)
        # target_wav = target_wav.to(device).unsqueeze(0)
        # test_wav = test_wav.to(device)

        #             est_mask = self.model(mixed_spec_padding, mixed_padding_feature)
        #
        #             mixed_mag, mixed_phase = self.audio.spec2magphase_torch(mixed_spec_padding)
        #             mask_mag, mask_phase = self.audio.spec2magphase_torch(est_mask)
        #
        #             output_mag = mixed_mag * mask_mag
        #             output_phase = mixed_phase + mask_phase
        #
        #             target_mag, _ = self.audio.spec2magphase_torch(target_spec)
        #
        #             enhanced_wav = self.audio.spec2wav(output_mag, output_phase)
        est_mask = self.model(mixed_spec_padding)

        mixed_mag, mixed_phase = self.audio.spec2magphase_torch(mixed_spec_padding)
        mask_mag, mask_phase = self.audio.spec2magphase_torch(est_mask)

        output_mag = mixed_mag * mask_mag
        output_phase = mixed_phase + mask_phase

        est_wav = self.audio.spec2wav(output_mag, output_phase)

        # 패딩 제거
        if est_wav.shape[1] > length:
            est_wav = est_wav[:, 0:length]

        est_wav = est_wav[0].cpu().detach().numpy()

        parent_directory_path = os.path.dirname(file_path)
        parent_directory_name = os.path.basename(parent_directory_path)

        if parent_directory_name != 'test':

            pass

        out_path = os.path.join(output_dir, parent_directory_name, os.path.basename(file_path))
        os.makedirs(os.path.join(output_dir, parent_directory_name), exist_ok=True)
        # out_path = os.path.join(output_dir, os.path.basename(file_name))

        est_wav = np.int16(est_wav / np.max(np.abs(est_wav)) * 32767)
        write(out_path, rate=self.hp.audio.sample_rate, data=est_wav)

    def write_tensorboard(self, writer, test_list, epoch):
        results = test_list[0]
        test_loss = np.mean(list(map(lambda x: x[0], results)))
        sdr = np.mean(list(map(lambda x: x[1], results)))

        mixed_wav = test_list[1]
        target_wav = test_list[2]
        enhanced_wav = test_list[3]

        writer.log_evaluation(test_loss, sdr, mixed_wav, target_wav, enhanced_wav, epoch)

        logger.info(f'sdr: {sdr}')

        return test_loss

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hp.train.adam)
        return optimizer

    def get_loss(self, train_data, output):
        target_wav, mixed_wav, target_spec, mixed_spec = train_data

        target_wav = target_wav.to(device)
        target_spec = target_spec.to(device)
        mixed_spec = mixed_spec.to(device)

        target_mag, target_phase = self.audio.spec2magphase_torch(target_spec)
        mixed_mag, mixed_phase = self.audio.spec2magphase_torch(mixed_spec)
        mask_mag, mask_phase = self.audio.spec2magphase_torch(output)

        # istft
        output_mag = mixed_mag * mask_mag
        output_phase = mixed_phase + mask_phase
        enhanced_wav = self.audio.spec2wav(output_mag, output_phase)

        # 사이즈 맞추기
        if enhanced_wav.shape[1] >= target_wav.shape[1]:
            enhanced_wav = enhanced_wav[:, 0:target_wav.shape[1]]
        else:
            target_wav = target_wav[:, 0:enhanced_wav.shape[1]]

        loss = self.si_snr(target_wav, enhanced_wav) + 100 * self.mse(target_mag, output_mag)
        return loss

    def get_model_name(self):
        return "DCUNET"
