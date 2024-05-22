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
    mean_mfcc = np.mean(value, axis=axis)

    if only_mean:
        return np.concatenate([mean_mfcc])

    std_mfcc = np.std(value, axis=axis)
    # median_mfcc = np.median(value, axis=axis)
    max_mfcc = np.array([np.max(value)])
    min_mfcc = np.array([np.min(value)])
    # print(max_mfcc.shape)
    # return np.concatenate([mean_mfcc, std_mfcc, median_mfcc, max_mfcc, min_mfcc])
    return np.concatenate([mean_mfcc, std_mfcc, max_mfcc, min_mfcc])


def get_simple_feature(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return get_statistical(mfcc)


def get_all_feature(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # MFCC의 1차 미분(속도) 계산
    mfcc_delta = librosa.feature.delta(mfcc)
    # MFCC의 2차 미분(가속도) 계산
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # 몰?루 ㅋㅋㅋㅋㅋㅋ
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    # 4. 벡터 양자화 (Vector Quantization)
    codebook, _ = kmeans(mfcc.T, k_or_guess=16)  # 코드북 생성
    vq_features, _ = vq(mfcc.T, codebook)

    # 5. 차원 축소 기법 (PCA)
    pca = PCA(n_components=10)
    mfcc_reduced = pca.fit_transform(mfcc.T)

    # 6. Contextual Features
    # context_window_size = 5
    # contextual_features = []
    # for i in range(context_window_size, mfcc.shape[1] - context_window_size):
    #     context = mfcc[:, i - context_window_size:i + context_window_size + 1].flatten()
    #     contextual_features.append(context)
    # contextual_features = np.array(contextual_features)

    # print(mfcc.shape, mfcc_delta.shape, mfcc_delta2.shape)
    # print(chroma.shape, contrast.shape, zcr.shape)
    # print(vq_features.shape, mfcc_reduced.shape, contextual_features.shape)

    # 전체 특징 결합
    combined_features = np.concatenate([
        get_statistical(mfcc),
        get_statistical(mfcc_delta, only_mean=True),
        get_statistical(mfcc_delta2, only_mean=True),
        get_statistical(chroma),
        get_statistical(contrast),
        get_statistical(zcr),
        get_statistical(np.expand_dims(vq_features, axis=0), only_mean=True),
        get_statistical(mfcc_reduced.T, only_mean=True),
        # get_statistical(contextual_features.T, only_mean=True),
    ])

    # print(combined_features.shape)

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


NEW_LOAD = True

def main():
    dataset_name = 'wav16k'
    train_folder = f'/mnt/nvme/dataset/{dataset_name}/train'
    train_ctrl = '202401ml_fmcc/fmcc_train.ctl'
    test_folder = f'/mnt/nvme/dataset/{dataset_name}/test/'
    test_ctrl = '202401ml_fmcc/fmcc_test.ctl'

    if NEW_LOAD:
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

    degrees = [2, 3, 4, 5]
    gammas = ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]
    cs = [0.1, 1, 3, 5, 10, 100, 250, 500, 1000]

    acc_best = 0
    svm_best = None

    for degree, gamma, c in tqdm(list(itertools.product(degrees, gammas, cs)), ncols=50):
        # 객체 생성
        svm = SVC(kernel='rbf', probability=False, degree=degree, gamma=gamma, C=c)

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

        print('\npredicted c={}, gamma={}, degree={}, acc={} / best c={}, gamma={}, degree={}, acc={}'.format(
            c, gamma, degree, acc, svm_best.C, svm_best.gamma, svm_best.degree, acc_best)
        )


if __name__ == "__main__":
    main()
