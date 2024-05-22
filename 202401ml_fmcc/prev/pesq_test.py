import numpy as np
from pesq import pesq
from tqdm import tqdm
from constant import *


def pesq_score(net, test_loader):
    net.eval()
    test_pesq = 0.
    counter = 0.

    for noisy_x, clean_x in tqdm(test_loader, ncols=100):
        # get the output from the model
        noisy_x = noisy_x.to(DEVICE)
        with torch.no_grad():
            pred_x = net(noisy_x)

        clean_x = torch.complex(pred_x[..., 0], pred_x[..., 1])
        clean_x = torch.istft(clean_x, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, return_complex=False)

        psq = []
        for i in range(len(clean_x)):
            # print(clean_x.shape, pred_x.shape)
            clean_x_16 = clean_x[i, :].view(1, -1)
            pred_x_16 = pred_x[i, :].view(1, -1)

            # print(clean_x_16.shape)

            clean_x_16 = clean_x_16.cpu().cpu().numpy()
            pred_x_16 = pred_x_16.detach().cpu().numpy()

            psq.append(pesq(16000, clean_x_16.squeeze(), pred_x_16.squeeze(), 'wb'))

        psq = float(np.mean(psq))
        test_pesq += psq
        counter += 1

    test_pesq /= counter
    return test_pesq