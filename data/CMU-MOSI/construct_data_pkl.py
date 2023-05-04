# -*-coding:utf-8-*-
import os
import numpy as np
import pandas as pd
import pickle
from train_valid_test_fold import train_valid_test_fold as tvt
from get_audio_embedding import check_audio_name
from get_video_embedding import check_video_name


def read_txt(path):
    texts = []
    file = open(path, "r")
    text_lines = list(map(lambda x:x.replace("_", ".", 1), file.readlines()))
    for line in sorted(text_lines):
        index = line.index("DELIM")
        texts.append(line[index+6:].strip())
    file.close()
    return texts


def get_fold_text(path, fold="train"):
    texts = []
    if fold == "train":
        for text_name in sorted(os.listdir(path)):
            if text_name[:11] in tvt.train_fold and not text_name.startswith('.'):
                texts += read_txt(os.path.join(path, text_name))
    elif fold == "valid":
        for text_name in sorted(os.listdir(path)):
            if text_name[:11] in tvt.valid_fold and not text_name.startswith('.'):
                texts += read_txt(os.path.join(path, text_name))
    else:
        for text_name in sorted(os.listdir(path)):
            if text_name[:11] in tvt.test_fold and not text_name.startswith('.'):
                texts += read_txt(os.path.join(path, text_name))
    # return np.array(texts)
    return texts

def get_fold_text2(path, fold="train"):
    if fold == "train":
        fold_labels = []
        df = pd.read_csv(path)
        for i in range(len(df)):
            if df.loc[i][df.columns[0]] in tvt.train_fold:
                fold_labels.append((df.loc[i][df.columns[0]]+"_"+str(df.loc[i][df.columns[1]]), df.loc[i][df.columns[3]], df.loc[i][df.columns[2]]))
        fold_labels = sorted(fold_labels, key=lambda x:x[0])
        texts = list(map(lambda x:x[2], fold_labels))
    elif fold == "valid":
        fold_labels = []
        df = pd.read_csv(path)
        for i in range(len(df)):
            if df.loc[i][df.columns[0]] in tvt.valid_fold:
                fold_labels.append((df.loc[i][df.columns[0]]+"_"+str(df.loc[i][df.columns[1]]), df.loc[i][df.columns[3]], df.loc[i][df.columns[2]]))
        fold_labels = sorted(fold_labels, key=lambda x:x[0])
        texts = list(map(lambda x:x[2], fold_labels))
    else:
        fold_labels = []
        df = pd.read_csv(path)
        for i in range(len(df)):
            if df.loc[i][df.columns[0]] in tvt.test_fold:
                fold_labels.append((df.loc[i][df.columns[0]]+"_"+str(df.loc[i][df.columns[1]]), df.loc[i][df.columns[3]], df.loc[i][df.columns[2]]))
        fold_labels = sorted(fold_labels, key=lambda x:x[0])
        texts = list(map(lambda x:x[2], fold_labels))
    return texts


def get_fold_labels(path, fold="train"):
    if fold == "train":
        fold_labels = []
        df = pd.read_csv(path)
        for i in range(len(df)):
            if df.loc[i][df.columns[0]] in tvt.train_fold:
                fold_labels.append((df.loc[i][df.columns[0]]+"_"+str(df.loc[i][df.columns[1]]), df.loc[i][df.columns[3]]))
        fold_labels = sorted(fold_labels, key=lambda x:x[0])
        fold_labels = list(map(lambda x:x[1], fold_labels))
    elif fold == "valid":
        fold_labels = []
        df = pd.read_csv(path)
        for i in range(len(df)):
            if df.loc[i][df.columns[0]] in tvt.valid_fold:
                fold_labels.append((df.loc[i][df.columns[0]]+"_"+str(df.loc[i][df.columns[1]]), df.loc[i][df.columns[3]]))
        fold_labels = sorted(fold_labels, key=lambda x:x[0])
        fold_labels = list(map(lambda x:x[1], fold_labels))
    else:
        fold_labels = []
        df = pd.read_csv(path)
        for i in range(len(df)):
            if df.loc[i][df.columns[0]] in tvt.test_fold:
                fold_labels.append((df.loc[i][df.columns[0]]+"_"+str(df.loc[i][df.columns[1]]), df.loc[i][df.columns[3]]))
        fold_labels = sorted(fold_labels, key=lambda x:x[0])
        fold_labels = list(map(lambda x:x[1], fold_labels))
    return fold_labels


def check_text_name(path, fold="train"):
    names = []
    if fold == "train":
        for text_name in sorted(os.listdir(path)):
            if text_name[:11] in tvt.train_fold and not text_name.startswith('.'):
                file = open(os.path.join(path, text_name), "r")
                text_lines = list(map(lambda x: x.replace("_", ".", 1), file.readlines()))
                name = [text_name[:11] + "_" + text[:text.index('.')] for text in sorted(text_lines)]
                names += name
    elif fold == "valid":
        for text_name in sorted(os.listdir(path)):
            if text_name[:11] in tvt.valid_fold and not text_name.startswith('.'):
                file = open(os.path.join(path, text_name), "r")
                text_lines = list(map(lambda x: x.replace("_", ".", 1), file.readlines()))
                name = [text_name[:11] + "_" + text[:text.index('.')] for text in sorted(text_lines)]
                names += name
    else:
        for text_name in sorted(os.listdir(path)):
            if text_name[:11] in tvt.test_fold and not text_name.startswith('.'):
                file = open(os.path.join(path, text_name), "r")
                text_lines = list(map(lambda x: x.replace("_", ".", 1), file.readlines()))
                name = [text_name[:11] + "_" + text[:text.index('.')] for text in sorted(text_lines)]
                names += name
    return names


def check_label_name(path, fold="train"):
    if fold == "train":
        fold_labels = []
        df = pd.read_csv(path)
        for i in range(len(df)):
            if df.loc[i][df.columns[0]] in tvt.train_fold:
                fold_labels.append((df.loc[i][df.columns[0]]+"_"+str(df.loc[i][df.columns[1]]), df.loc[i][df.columns[3]]))
        fold_labels = sorted(fold_labels, key=lambda x:x[0])
        names = list(map(lambda x:x[0], fold_labels))
    elif fold == "valid":
        fold_labels = []
        df = pd.read_csv(path)
        for i in range(len(df)):
            if df.loc[i][df.columns[0]] in tvt.valid_fold:
                fold_labels.append((df.loc[i][df.columns[0]]+"_"+str(df.loc[i][df.columns[1]]), df.loc[i][df.columns[3]]))
        fold_labels = sorted(fold_labels, key=lambda x:x[0])
        names = list(map(lambda x:x[0], fold_labels))
    else:
        fold_labels = []
        df = pd.read_csv(path)
        for i in range(len(df)):
            if df.loc[i][df.columns[0]] in tvt.test_fold:
                fold_labels.append((df.loc[i][df.columns[0]]+"_"+str(df.loc[i][df.columns[1]]), df.loc[i][df.columns[3]]))
        fold_labels = sorted(fold_labels, key=lambda x:x[0])
        names = list(map(lambda x:x[0], fold_labels))
    return names


def construct_data_pkl():
    data = {}
    data["train"] = {}
    data["train"]["raw_text"] = get_fold_text("Transcript/Segmented", "train")
    data["train"]["audio"] = np.load("train_audio_embeddings.npy")
    data["train"]["video"] = np.load("train_video_embeddings.npy")
    data["train"]["labels"] = get_fold_labels("MOSI-label.csv", "train")
    data["valid"] = {}
    data["valid"]["raw_text"] = get_fold_text("Transcript/Segmented", "valid")
    data["valid"]["audio"] = np.load("valid_audio_embeddings.npy")
    data["valid"]["video"] = np.load("valid_video_embeddings.npy")
    data["valid"]["labels"] = get_fold_labels("MOSI-label.csv", "valid")
    data["test"] = {}
    data["test"]["raw_text"] = get_fold_text("Transcript/Segmented", "test")
    data["test"]["audio"] = np.load("test_audio_embeddings.npy")
    data["test"]["video"] = np.load("test_video_embeddings.npy")
    data["test"]["labels"] = get_fold_labels("MOSI-label.csv", "test")
    with open("mosi.pkl", "wb") as file:
        pickle.dump(data, file)


if __name__ == "__main__":
    # train_text = get_fold_text("Transcript/Segmented", "train")
    # print(type(train_text))
    # print(train_text.shape)
    # print(train_text[:13])
    # =============================================================
    train_text1 = get_fold_text("/home/qianfan/Data/CMU-MOSI/Raw/Transcript/Segmented", "train")
    train_text2 = get_fold_text2("MOSI-label.csv", "train")
    test_text1 = get_fold_text("/home/qianfan/Data/CMU-MOSI/Raw/Transcript/Segmented", "test")
    # print(train_text1 == train_text2)
    # print(train_text1[:10])
    # print(train_text2[:10])
    print(test_text1[:5])
    print(test_text1[33:38])
    print(test_text1[49:54])
    print(test_text1[69])

    # =============================================================
    fold_labels = get_fold_labels("MOSI-label.csv", "test")
    print(type(fold_labels), len(fold_labels))
    print(fold_labels[:5])
    print(fold_labels[33:38])
    print(fold_labels[49:54])
    print(fold_labels[69])
    # =============================================================
    # construct_data_pkl()
    # =============================================================
    # text_train_names = check_text_name("Transcript/Segmented", "train")
    # text_valid_names = check_text_name("Transcript/Segmented", "valid")
    # text_test_names = check_text_name("Transcript/Segmented", "test")
    # audio_train_names = check_audio_name("Audio/WAV_16000/Segmented", "train")
    # audio_valid_names = check_audio_name("Audio/WAV_16000/Segmented", "valid")
    # audio_test_names = check_audio_name("Audio/WAV_16000/Segmented", "test")
    # video_train_names = check_video_name("Video/Segmented", "train")
    # video_valid_names = check_video_name("Video/Segmented", "valid")
    # video_test_names = check_video_name("Video/Segmented", "test")
    # label_train_names = check_label_name("MOSI-label.csv", "train")
    # label_valid_names = check_label_name("MOSI-label.csv", "valid")
    # label_test_names = check_label_name("MOSI-label.csv", "test")
    # print(text_train_names == audio_train_names == video_train_names == label_train_names)
    # print(text_valid_names == audio_valid_names == video_valid_names == label_valid_names)
    # print(text_test_names == audio_test_names == video_test_names == label_test_names)
    # print(text_train_names[-10:])
    # print(audio_train_names[-10:])
    # print(video_train_names[-10:])
    # print(label_train_names[-10:])