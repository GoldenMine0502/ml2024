import os
from pathlib import Path, PurePath

import librosa
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoProcessor, HubertForCTC, HubertModel
import numpy as np
from scipy.cluster.vq import kmeans, vq

torch.backends.cudnn.enabled=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = None
model = None

# model_name = 'facebook/wav2vec2-large-960h-lv60-self'
#
# processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
# model = Wav2Vec2Model.from_pretrained(model_name)
# model.to(device)
#

# pretrained_model_name = 'facebook/hubert-xlarge-ls960-ft'
# pretrained_model_name = 'facebook/hubert-base-ls960'
#
# try:
#     processor = AutoProcessor.from_pretrained(pretrained_model_name)
# except:
#     processor = None
#
# model = HubertModel.from_pretrained(pretrained_model_name)
# # model = HubertForCTC.from_pretrained(pretrained_model_name)
# model.to(device)


def extract_features(src_path, dst_path):
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


    def extract_hubert(input_audio):
        if processor is not None:
            inputs = processor(input_audio, sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(device).squeeze(1)
        else:
            input_values = input_audio.unsqueeze(0)

        with torch.no_grad():
            hidden_states = model(input_values)

        result_feature = hidden_states.logits

        return result_feature

    last_shape = None

    for filename in tqdm(list(src_path.rglob('*.wav')), ncols=50):
        file_path = src_path / filename


        y, sr = librosa.load(file_path)


        n_mfcc = 26
        n_mels = 20
        fmax = 400
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mels=n_mels, n_mfcc=n_mfcc, fmax=fmax)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        codebook, _ = kmeans(mfcc.T, k_or_guess=16)
        vq_features, _ = vq(mfcc.T, codebook)

        pca = PCA(n_components=10)
        mfcc_reduced = pca.fit_transform(mfcc.T)

        # tempogram = librosa.feature.tempogram(y=y, sr=sr)

        # wav2vec model result = c
        # feature = extract_wav2vec2(input_audio)
        # inputs = processor(y, return_tensors="pt", sampling_rate=sr)
        # with torch.no_grad():
        #     # z = model(**inputs.to(device)).extract_features
        #     feature = model(**inputs.to(device)).last_hidden_state

        # HuBERT feature
        # if processor is not None:
        #     inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        #     input_values = inputs.input_values.to(device).squeeze(1)
        # else:
        #     input_values = torch.from_numpy(y[np.newaxis, ...]).to(device)
        #
        # with torch.no_grad():
        #     hidden_states = model(input_values)

        # print(hidden_states)
        # feature = hidden_states.logits
        # feature = hidden_states.last_hidden_state

        parents = filename.parent.name if filename.parent.name == 'test' else f'{filename.parent.parent.name}/{filename.parent.name}'

        output_file_path = dst_path / parents / filename.name
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # print('\npath:', file_path, dst_path, output_file_path, filename.name)

        # feature = feature.squeeze(0).transpose(0, 1)
        # feature = feature.cpu().detach().numpy()

        def make_context_feature(feature, context_window_size=15):
            contextual_features = []
            for i in range(context_window_size, feature.shape[1] - context_window_size):
                context = feature[:, i - context_window_size:i + context_window_size + 1].flatten()
                contextual_features.append(context)
            return np.array(contextual_features)

        context_mfcc_feature = make_context_feature(mfcc)
        # context_hubert_feature = make_context_feature(feature, context_window_size=3)
        # context_wav2vec_feature = make_context_feature(feature, context_window_size=5)

        # print(y.shape, mfcc.shape, context_mfcc_feature.shape)
        # print('\n', mfcc.shape, feature.shape, context_mfcc_feature.shape, context_wav2vec_feature.shape)

        res = np.concatenate([
            get_statistical(mfcc),
            get_statistical(mfcc_delta, only_mean=True),  # 평균이 좋음
            get_statistical(mfcc_delta2, only_mean=True),  # 평균이 좋음
            get_statistical(chroma, only_mean=True),  # 평균 전체 성능 똑같음
            get_statistical(contrast, only_mean=True),  # 평균이 도움됨. 전체도 도움됨.
            get_statistical(zcr),  # 전체가 도움됨. 평균은 성능 크게 떨어뜨림
            get_statistical(mel, only_mean=True),  # 평균이 도움됨. 전체는 성능 떨어뜨림
            get_statistical(centroid, only_mean=True),  # 평균 약간 성능 올림. 전체 약간 성능 올림
            # get_statistical(tempogram, only_mean=True),  # 평균 약간 성능 올림. 전체 약간 성능 올림
            get_statistical(mfcc_reduced.T),  # 평균 약간 성능 올림. 전체 약간 성능 올림
            get_statistical(context_mfcc_feature.T, only_mean=True),  # 평균 존나올림 ㅋ 전체는 별로

            # 추가
            # get_statistical(feature, only_mean=True),  # base 크기 32
            # get_statistical(context_hubert_feature.T, only_mean=True),  # xlarge 크기 32
            # get_statistical(context_wav2vec_feature.T, only_mean=True),
        ])

        # print(feature.shape, get_statistical(feature, only_mean=True).shape)
        # print(feature.shape, res.shape)

        if last_shape is None:
            last_shape = res.shape[0]
        else:
            assert last_shape == res.shape[0]

        np.save(output_file_path, res)

    # torch.Size([1, 56, 768])
    # [Batch, T, 768]
    # print(feature.shape, res.shape)
    print(res.shape)

def main():
    # 오디오 경로
    wav_path = Path('/mnt/nvme/dataset/wav16k')

    # 특징 벡터 저장 경로
    output_path = Path('/mnt/nvme/dataset/wav16k_feature')

    extract_features(wav_path, output_path)


if __name__ == '__main__':
    main()
