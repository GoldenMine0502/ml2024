import numpy as np
import gc
from tqdm import tqdm
from pesq import pesq

from dcunet import DCUnet10
from constant import *

def wsdr_fn(x_, y_pred_, y_true_, eps=1e-8):
    # to time-domain waveform
    y_true_ = torch.squeeze(y_true_, 1)
    y_true_ = torch.complex(y_true_[..., 0], y_true_[..., 1])

    x_ = torch.squeeze(x_, 1)
    x_ = torch.complex(x_[..., 0], x_[..., 1])

    y_true = torch.istft(y_true_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, return_complex=False)
    x = torch.istft(x_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, return_complex=False)

    y_pred = y_pred_.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)


    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)


def train_epoch(net, train_loader, loss_fn, optimizer):
    net.train()
    train_ep_loss = 0.
    counter = 0
    print('training')
    for noisy_x, clean_x, in tqdm(train_loader, ncols=100):
        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)

        # zero  gradients
        net.zero_grad()

        # get the output from the model
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        loss.backward()
        optimizer.step()

        train_ep_loss += loss.item()
        counter += 1

    train_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss

def train(net, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs):
    train_losses = []
    test_losses = []

    for e in range(epochs):

        # first evaluating for comparison
        if e == 0:
            with torch.no_grad():
                test_loss = test_epoch(net, test_loader, loss_fn)

            test_losses.append(test_loss)
            print("Loss before training:{:.6f}".format(test_loss))

        train_loss = train_epoch(net, train_loader, loss_fn, optimizer)
        scheduler.step()
        with torch.no_grad():
            test_loss = test_epoch(net, test_loader, loss_fn)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()

        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Loss: {:.6f}...".format(train_loss),
              "Test Loss: {:.6f}".format(test_loss))

        if e % 5 == 0:
            torch.save(net.state_dict(), 'weights/weights{}.pth'.format(EPOCH))

    return train_losses, test_losses


def test_epoch(net, test_loader, loss_fn):
    net.eval()
    test_ep_loss = 0.
    counter = 0.
    print('testing')
    for noisy_x, clean_x in tqdm(test_loader, ncols=100):
        # get the output from the model
        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        test_ep_loss += loss.item()

        counter += 1

    test_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()

    return test_ep_loss



def pesq_score(net, test_loader):
    net.eval()
    test_pesq = 0.
    counter = 0.

    for noisy_x, clean_x in tqdm(test_loader, ncols=100):
        # get the output from the model
        noisy_x = noisy_x.to(DEVICE)
        with torch.no_grad():
            pred_x = net(noisy_x)
        clean_x = torch.squeeze(clean_x, 1)
        clean_x = torch.complex(clean_x[..., 0], clean_x[..., 1])
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


if not os.path.exists('weights'):
    os.mkdir('weights')
# clear cache
gc.collect()
torch.cuda.empty_cache()

dcunet10 = DCUnet10(N_FFT, HOP_LENGTH).to(DEVICE)

loss_fn = wsdr_fn
optimizer = torch.optim.Adam(dcunet10.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

train_losses, test_losses = train(dcunet10, train_loader, test_loader, loss_fn, optimizer, scheduler, EPOCH)

pesq_metric = pesq_score(dcunet10, test_loader)
print('pesq:', pesq_metric)

torch.save(dcunet10.state_dict(), 'weights/weights{}.pth'.format(EPOCH))
