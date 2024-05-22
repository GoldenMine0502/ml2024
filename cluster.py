import argparse
import itertools
import os
import subprocess

import numpy as np
import librosa
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm


# 16kHz, 16bit linear encoding, mono channel, little endian
def load_audio(file_path):
    data, sr = librosa.load(file_path)

    return data, sr


def get_statistical(value, axis=1, only_mean=False):
    mean_val = np.mean(value, axis=axis)
    std_val = np.std(value, axis=axis)
    median_val = np.median(value, axis=axis)
    max_val = np.max(value, axis=axis)
    min_val = np.min(value, axis=axis)

    feature_list = [mean_val]
    if not only_mean:
        feature_list.extend([std_val, median_val, max_val, min_val])

    concatenated = np.concatenate(feature_list)
    return concatenated

def get_simple_feature(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return get_statistical(mfcc)

def get_all_feature(y, sr):
    n_mfcc = 26
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    # tonnetz = librosa.feature.tonnetz(y=y, sr=sr, hop_length=128)

    codebook, _ = kmeans(mfcc.T, k_or_guess=16)
    vq_features, _ = vq(mfcc.T, codebook)

    pca = PCA(n_components=10)
    mfcc_reduced = pca.fit_transform(mfcc.T)

    combined_features = np.concatenate([
        get_statistical(mfcc),
        get_statistical(mfcc_delta, only_mean=True),
        get_statistical(mfcc_delta2, only_mean=True),
        get_statistical(chroma, only_mean=True),
        get_statistical(contrast, only_mean=True),
        get_statistical(mel, only_mean=True),
        get_statistical(rolloff, only_mean=True),
        get_statistical(zcr, only_mean=True),
        get_statistical(np.expand_dims(flux, axis=0), only_mean=True),
        get_statistical(rms, only_mean=True),
        get_statistical(bandwidth, only_mean=True),
        get_statistical(centroid, only_mean=True),
        get_statistical(flatness, only_mean=True),
        # get_statistical(tonnetz, only_mean=True),
        get_statistical(np.expand_dims(vq_features, axis=0), only_mean=True),
        get_statistical(mfcc_reduced.T, only_mean=True),
    ])

    return combined_features

def load_audio_files(root_folder, ctrl):
    audio_data = []
    labels = []
    files = []

    with open(ctrl, 'rt') as file:
        for line in tqdm(list(file), ncols=50):
            line = line.rstrip()

            y, sr = load_audio(os.path.join(root_folder, '{}.wav'.format(line)))
            label = 0 if line[0] == 'F' else 1
            combined_feature = get_all_feature(y, sr)

            audio_data.append(combined_feature)
            labels.append(label)
            files.append(os.path.basename(line))

    return audio_data, labels, files


def perform_kmeans_clustering(data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    return kmeans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wav16k')
    parser.add_argument('--train_folder', type=str, required=True)
    parser.add_argument('--train_ctrl', type=str, default=None)
    parser.add_argument('--test_folder', type=str, default=None)
    parser.add_argument('--test_ctrl', type=str, default=None)
    parser.add_argument('--new_load', type=bool, default=True)
    args = parser.parse_args()

    dataset_name = args.dataset
    train_folder = args.train_folder
    train_ctrl = args.train_ctrl
    test_folder = args.test_folder
    test_ctrl = args.test_ctrl
    new_load = args.new_load

    if new_load:
        print('extracting...')
        # audio_data, labels, _ = load_audio_files(train_folder)
        audio_data, labels, _ = load_audio_files(train_folder, train_ctrl)
        test_data, _, test_files = load_audio_files(test_folder, test_ctrl)

        if not os.path.exists('saves'):
            os.mkdir('saves')

        np.save(f'saves/audio_{dataset_name}', audio_data)
        np.save(f'saves/label_{dataset_name}', labels)
        np.save(f'saves/audio_t_{dataset_name}', test_data)
        np.save(f'saves/file_t_{dataset_name}', test_files)
    else:
        print('loading...')
        audio_data = np.load(f'saves/audio_{dataset_name}.npy')
        labels = np.load(f'saves/label_{dataset_name}.npy')
        test_data = np.load(f'saves/audio_t_{dataset_name}.npy')
        test_files = np.load(f'saves/file_t_{dataset_name}.npy')

    print(f'feature size: {audio_data[0].shape}')

    # label은 잘 나온다
    # for label, file in zip(labels, files):
    #     print(label, file)
    # kmeans = perform_kmeans_clustering(audio_data)

    print('training...')

    degrees = [2, 3, 5]
    gammas = ['scale', 'auto', 0.001, 0.01, 0.1]
    cs = [1, 10, 100, 250, 500, 1000]
    coef0 = [0.0, 0.5, 1.0]
    decision_function_shape = ['ovo', 'ovr']

    acc_best = 0
    svm_best = None

    for degree, gamma, c, coef0, decision_function_shape in tqdm(list(itertools.product(degrees, gammas, cs, coef0, decision_function_shape)), ncols=50):
        # 객체 생성
        svm = SVC(kernel='rbf', probability=False, degree=degree, gamma=gamma, C=c, coef0=coef0, decision_function_shape=decision_function_shape)

        # 학습
        svm.fit(audio_data, labels)

        # 예측
        predictions = svm.predict(test_data)

        # 결과 파일 제작
        result_file = 'classification_results.txt'
        with open(result_file, 'w') as f:
            for file, prediction in zip(test_files, predictions):
                result_label = 'feml' if prediction == 0 else 'male'
                # output_filename = f"fmcc_test_{i:04d}"
                f.write(f"{file} {result_label}\n")

        # eval.pl 실행해서 acc 구하기
        res = subprocess.run(f"./202401ml_fmcc/eval.pl {result_file} ./202401ml_fmcc/fmcc_test_ref.txt",
                             shell=True,
                             capture_output=True,
                             text=True,
                             ).stdout
        acc = float(res.split('\n')[3].split(': ')[1].replace('%', ''))

        # acc가 가장 높은 모델 선택
        if acc_best < acc:
            acc_best = acc
            svm_best = svm

        print('\n[predicted c={}, gamma={}, degree={}, acc={}, coef={}, dec={}] best {} {} {} {} {} {}'.format(
            c, gamma, degree, acc, coef0, decision_function_shape,
            svm_best.C, svm_best.gamma, svm_best.degree, acc_best, svm_best.coef0, svm_best.decision_function_shape
        ))


if __name__ == "__main__":
    main()
