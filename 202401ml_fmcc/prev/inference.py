import os

from scipy.io.wavfile import write
from tqdm import tqdm

from constant import *
import torch

from dcunet import DCUnet10

MY_EPOCH = 25

chkpt_path = 'weights/weights{}.pth'.format(MY_EPOCH)
chkpt_model = torch.load(chkpt_path)

dcunet10 = DCUnet10(N_FFT, HOP_LENGTH).to(DEVICE)
dcunet10.load_state_dict(chkpt_model)

dcunet10.eval()

# 배치사이즈 1임
for noisy_x, noisy_path, length in tqdm(inference_loader, ncols=100):
    # noisy_x = data['x']
    # noisy_path = data['path']

    # print(noisy_x, noisy_path)
    noisy_x = noisy_x.to(DEVICE).unsqueeze(0)  # give batch size = 1

    with torch.no_grad():
        res = dcunet10(noisy_x)

    parent_directory_path = os.path.dirname(noisy_path)
    parent_directory_name = os.path.basename(parent_directory_path)

    out_path = os.path.join(INFERENCE_CLEAN_DIR, parent_directory_name, os.path.basename(noisy_path))
    os.makedirs(os.path.join(INFERENCE_CLEAN_DIR, parent_directory_name), exist_ok=True)

    # squeeze 후 length 까지 자르기 (패딩 안된 원래길이)
    est_wav = res[0][0:length].cpu().detach().numpy()

    write(out_path, rate=16000, data=est_wav)