# -*-coding:utf-8-*- 
import os
import numpy as np
import pandas as pd
from train_valid_test_fold import train_valid_test_fold as tvt


def get_one_video_embedding(video_path, faces_feature_dir, name, openface_path="/home/qianfan/Openface/build/bin/FeatureExtraction", T=55):
    cmd = openface_path + ' -f ' + video_path + ' -out_dir ' + faces_feature_dir + " -of " + name + " -2Dfp -3Dfp -pdmparams -pose -aus -gaze"
    os.system(cmd)  # output a csv file containing facial landmark, head pose, eye gaze and facial action unit features
    # read features
    features = []
    df_path = os.path.join(faces_feature_dir, name + '.csv')
    df = pd.read_csv(df_path)  # read csv file
    for i in range(len(df)):
        features.append(np.array(df.loc[i][df.columns[5:]]))
    features = np.array(features)
    if features.shape[0] < T:
        new_features = np.concatenate((features, np.zeros((T-features.shape[0],features.shape[1]))),axis=0)
    elif features.shape[0] > T:
        new_features = features[:T]
    else:
        new_features = features

    return new_features  # array shape: (T, 709)


def get_fold_video_embedding(path, faces_feature_dir, fold="train"):
    features = []
    if fold == "train":
        for video_name in sorted(os.listdir(path)):
            if video_name[:11] in tvt.train_fold and not video_name.startswith('.'):
                feature = get_one_video_embedding(os.path.join(path, video_name), faces_feature_dir, video_name[:-4])
                features.append(feature)
    elif fold == "valid":
        for video_name in sorted(os.listdir(path)):
            if video_name[:11] in tvt.valid_fold and not video_name.startswith('.'):
                feature = get_one_video_embedding(os.path.join(path, video_name), faces_feature_dir, video_name[:-4])
                features.append(feature)
    else:
        for video_name in sorted(os.listdir(path)):
            if video_name[:11] in tvt.test_fold and not video_name.startswith('.'):
                feature = get_one_video_embedding(os.path.join(path, video_name), faces_feature_dir, video_name[:-4])
                features.append(feature)
    return np.array(features)


def check_video_name(path, fold="train"):
    names = []
    if fold == "train":
        for video_name in sorted(os.listdir(path)):
            if video_name[:11] in tvt.train_fold and not video_name.startswith('.'):
                names.append(video_name[:-4])
    elif fold == "valid":
        for video_name in sorted(os.listdir(path)):
            if video_name[:11] in tvt.valid_fold and not video_name.startswith('.'):
                names.append(video_name[:-4])
    else:
        for video_name in sorted(os.listdir(path)):
            if video_name[:11] in tvt.test_fold and not video_name.startswith('.'):
                names.append(video_name[:-4])
    return names


if __name__ == "__main__":
    # train_video_embeddings = get_fold_video_embedding("Video/Segmented", "tmp_csv", "train")
    # np.save("train_video_embeddings.npy", train_video_embeddings)
    # print(train_video_embeddings.shape)
    # =========================================================================================
    # valid_video_embeddings = get_fold_video_embedding("Video/Segmented", "tmp_csv", "valid")
    # np.save("valid_video_embeddings.npy", valid_video_embeddings)
    # print(valid_video_embeddings.shape)
    # =========================================================================================
    # test_video_embeddings = get_fold_video_embedding("Video/Segmented", "tmp_csv", "test")
    # np.save("test_video_embeddings.npy", test_video_embeddings)
    # print(test_video_embeddings.shape)
    # =========================================================================================
    train_names = check_video_name("Video/Segmented", "train")
    valid_names = check_video_name("Video/Segmented", "valid")
    test_names = check_video_name("Video/Segmented", "test")
