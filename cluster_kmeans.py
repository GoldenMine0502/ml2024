import os
from pathlib import Path

import librosa
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
from constant import *


audio_files = inference_noisy_files


# 각 오디오 파일에 대해 MFCC 추출
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # [n_mfcc, 시간]
    mfccs_mean = np.mean(mfccs.T, axis=0)  # 시간 축을 따라 평균값 계산 [n_mfcc]
    return mfccs_mean


# 모든 오디오 파일에 대해 MFCC 특징 추출
print('extracting features...')
mfcc_features = [extract_mfcc(file) for file in tqdm(audio_files, ncols=100)]
mfcc_features = np.array(mfcc_features)

print('fitting...')
# K-means 클러스터링 수행
n_clusters = 10  # 원하는 클러스터 수
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mfcc_features)
labels = kmeans.labels_

# 클러스터링 결과 출력
for i, label in enumerate(labels):
    print(f"File: {audio_files[i]} - Cluster: {label}")

# 클러스터링 결과 시각화
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    cluster_points = mfcc_features[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
plt.xlabel('MFCC 1')
plt.ylabel('MFCC 2')
plt.legend()
plt.title('K-means Clustering of Audio Files using MFCC')
plt.show()