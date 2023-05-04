# -*-coding:utf-8-*- 
import os
import numpy as np
import librosa
from train_valid_test_fold import train_valid_test_fold as tvt


def get_one_audio_embedding(audio_path, T=400):
    # get features
    y, sr = librosa.load(audio_path) # wave and sampling rate
    # using librosa to get audio features (f0, mfcc, cqt)
    hop_length = 512 # hop_length smaller, seq_len larger
    f0 = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).T # (seq_len, 1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, htk=True).T # (seq_len, 20)
    cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T # (seq_len, 12)

    feature = np.concatenate([f0, mfcc, cqt], axis=-1)
    if feature.shape[0] < T:
        new_feature = np.concatenate((feature, np.zeros((T-feature.shape[0],feature.shape[1]))),axis=0)
    elif feature.shape[0] > T:
        new_feature = feature[:T]
    else:
        new_feature = feature

    return new_feature


def get_fold_audio_embedding(path, fold="train"):
    features = []
    if fold == "train":
        for audio_name in sorted(os.listdir(path)):
            if audio_name[:11] in tvt.train_fold and not audio_name.startswith('.'):
                feature = get_one_audio_embedding(os.path.join(path, audio_name))
                features.append(feature)
    elif fold == "valid":
        for audio_name in sorted(os.listdir(path)):
            if audio_name[:11] in tvt.valid_fold and not audio_name.startswith('.'):
                feature = get_one_audio_embedding(os.path.join(path, audio_name))
                features.append(feature)
    else:
        for audio_name in sorted(os.listdir(path)):
            if audio_name[:11] in tvt.test_fold and not audio_name.startswith('.'):
                feature = get_one_audio_embedding(os.path.join(path, audio_name))
                features.append(feature)
    return np.array(features)


def check_audio_name(path, fold="train"):
    names = []
    if fold == "train":
        for audio_name in sorted(os.listdir(path)):
            if audio_name[:11] in tvt.train_fold and not audio_name.startswith('.'):
                names.append(audio_name[:-4])
    elif fold == "valid":
        for audio_name in sorted(os.listdir(path)):
            if audio_name[:11] in tvt.valid_fold and not audio_name.startswith('.'):
                names.append(audio_name[:-4])
    else:
        for audio_name in sorted(os.listdir(path)):
            if audio_name[:11] in tvt.test_fold and not audio_name.startswith('.'):
                names.append(audio_name[:-4])
    return names


if __name__ == "__main__":
    # train_audio_embeddings = get_fold_audio_embedding("Audio/WAV_16000/Segmented", fold="train")
    # np.save("train_audio_embeddings.npy", train_audio_embeddings)
    # print(train_audio_embeddings.shape)
    # =============================================================================================
    # valid_audio_embeddings = get_fold_audio_embedding("Audio/WAV_16000/Segmented", fold="valid")
    # np.save("valid_audio_embeddings.npy", valid_audio_embeddings)
    # print(valid_audio_embeddings.shape)
    # =============================================================================================
    # test_audio_embeddings = get_fold_audio_embedding("Audio/WAV_16000/Segmented", fold="test")
    # np.save("test_audio_embeddings.npy", test_audio_embeddings)
    # print(test_audio_embeddings.shape)
    # =============================================================================================
    train_names = check_audio_name("Audio/WAV_16000/Segmented", "train")
    valid_names = check_audio_name("Audio/WAV_16000/Segmented", "valid")
    test_names = check_audio_name("Audio/WAV_16000/Segmented", "test")