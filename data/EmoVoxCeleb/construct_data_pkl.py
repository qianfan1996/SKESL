# -*-coding:utf-8-*- 
import os
import numpy as np
import pickle
from get_emovoxceleb_speaker import get_chosen_speaker
from ignored_av_segment import ignored_av_segment as ig


def read_all_txt(path):
    texts = []
    for dir in sorted(os.listdir(path)):
        if dir in get_chosen_speaker("EmoVoxCeleb_meta.csv"):
            for text_name in sorted(os.listdir(os.path.join(path, dir))):
                if not dir + "_" + text_name[:-12] in ig.ignored_audio_segment:
                    if not text_name.startswith('.'):
                        with open(os.path.join(path, dir, text_name),"r") as file:
                            for line in file.readlines():
                                texts.append(line.strip())
    return np.array(texts)


def construct_data_pkl():
    data = {}
    # data["raw_text"] = read_all_txt("text")
    data["raw_text"] = pickle.load(open("texts.pkl", "rb"))
    data["audio"] = np.load("audio_embeddings.npy")
    data["video"] = np.load("face_embeddings.npy")

    with open("emovoxceleb.pkl", "wb") as file:
        pickle.dump(data, file)


if __name__ == "__main__":
    # texts = read_all_txt("text")
    # print(type(texts))
    # print(texts.shape)
    # print(texts[:10])
    # text = pickle.load(open("texts.pkl", "rb"))
    # print(type(text))
    # print(len(text))
    # print(text[:3])
    construct_data_pkl()