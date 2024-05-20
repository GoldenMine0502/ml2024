import os

from scipy.io.wavfile import write

from constant import *
import torch

from dcunet import DCUnet10

chkpt_path = '__weights.pth'
chkpt_model = torch.load(chkpt_path)

dcunet10 = DCUnet10(N_FFT, HOP_LENGTH).to(DEVICE)
dcunet10.load_state_dict(chkpt_model)

dcunet10.eval()

# 배치사이즈 1임
for noisy_x, noisy_path in inference_loader:
    noisy_x = noisy_x.to(DEVICE)

    with torch.no_grad():
        res = dcunet10(noisy_x)
    out_path = os.path.join(INFERENCE_CLEAN_DIR, os.path.basename(noisy_path))
    est_wav = res[0].cpu().detach().numpy()

    write(out_path, rate=16000, data=est_wav)