import matplotlib
import wave
# matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np


def fig2np(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data


def plot_speech(wav_path, position):
    NSTEP = 100
    spf = wave.open(wav_path, 'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'int16')
    fs = spf.getframerate()
    len_signal = len(signal)
    Time = np.linspace(0, len_signal/fs, num=len_signal)
    spf.close()

    max_signal, min_signal = max(signal), min(signal)
    detected = np.full(len_signal, min_signal)
    for s, f in position:
        detected[s*NSTEP : f*NSTEP+1] = list(np.full((f-s)*NSTEP+1, max_signal))

    plt.plot(Time, signal, '#000000')
    plt.plot(Time, detected, '#747474')
    plt.show()
