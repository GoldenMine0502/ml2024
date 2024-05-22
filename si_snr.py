import torch


def si_snr(clean, enhanced):
    snr = torch.zeros(enhanced.shape[0]).cuda()
    for i in range(enhanced.shape[0]):
        target = (torch.dot(clean[i], enhanced[i]) * clean[i]) / torch.norm(clean[i]) ** 2
        noise = enhanced[i] - target[i]
        snr[i] += 10 * torch.log10((torch.norm(target) ** 2) / torch.norm(noise) ** 2)
    return -torch.mean(snr)


class SI_SNR:
    def __init__(self):
        self.loss = si_snr

    def __call__(self, clean_wav, enhanced_wav):
        return self.loss(clean_wav, enhanced_wav)