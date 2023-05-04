# -*-coding:utf-8-*- 
import os
import numpy as np
import librosa
from get_emovoxceleb_speaker import get_chosen_speaker
from ignored_av_segment import ignored_av_segment as ig

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


def get_all_audio_embedding(path):
    features = []
    for dir in sorted(os.listdir(path)):
        if dir in get_chosen_speaker("EmoVoxCeleb_meta.csv"):
            for audio_name in sorted(os.listdir(os.path.join(path, dir))):
                if not dir + "_" + audio_name[:-12] in ig.ignored_audio_segment:
                    if not audio_name.startswith('.'):
                        feature = get_one_audio_embedding(os.path.join(path, dir, audio_name))
                        features.append(feature)
    return np.array(features)


if __name__ == "__main__":
    # audio_embedding = get_one_audio_embedding("audio/Aaron_Tveit/6WxS8rpNjmk_0000001.wav")
    # print(audio_embedding.shape)
    # ========================================================================================
    audio_embeddings = get_all_audio_embedding("audio")
    np.save("audio_embeddings.npy", audio_embeddings)
    print(audio_embeddings.shape)