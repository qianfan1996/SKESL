# -*-coding:utf-8-*- 
import os
import numpy as np
import pandas as pd
from get_emovoxceleb_speaker import get_chosen_speaker
from ignored_av_segment import ignored_av_segment as ig
from glob import glob

def get_one_video_embedding(video_path, faces_feature_dir, name, openface_path="/home/qianfan/Openface/build/bin/FeatureExtraction", T=55):
    if "." in name:
        name = name.replace(".", "_")
    cmd = openface_path + ' -fdir ' + video_path + ' -out_dir ' + faces_feature_dir + " -of " + name + " -2Dfp -3Dfp -pdmparams -pose -aus -gaze"
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


def get_all_videos_embedding(path, faces_feature_dir):
    features = []
    for dir in sorted(os.listdir(path)):
        if dir in get_chosen_speaker("EmoVoxCeleb_meta.csv"):
            for video_name in sorted(os.listdir(os.path.join(path, dir, "1.6"))):
                for clip_name in sorted(os.listdir(os.path.join(path, dir, "1.6", video_name))):
                    if not dir+"_"+video_name+"_"+"000000"+clip_name in ig.ignored_video_segment:
                        feature = get_one_video_embedding(os.path.join(path, dir, "1.6", video_name, clip_name), faces_feature_dir, dir+"_"+video_name+"_"+"0"*(7-len(clip_name))+clip_name)
                        features.append(feature)
    return np.array(features)


def get_face_embeddings(path, T=55):
    csv_files = glob(os.path.join(path, "*.csv"))
    features = []
    for file in sorted(csv_files):
        feature = []
        df = pd.read_csv(file)
        for i in range(len(df)):
            feature.append(np.array(df.loc[i][df.columns[5:]]))
        feature = np.array(feature)
        if feature.shape[0] < T:
            new_feature = np.concatenate((feature, np.zeros((T - feature.shape[0], feature.shape[1]))), axis=0)
        elif feature.shape[0] > T:
            new_feature = feature[:T]
        else:
            new_feature = feature
        features.append(new_feature)
    return np.array(features)



if __name__ == "__main__":
    # face_embedding = get_one_video_embedding("video/Aaron_Tveit/1.6/6WxS8rpNjmk/1", "tmp_faces", "ok")
    # print(face_embedding.shape)
    # face_embeddings = get_all_videos_embedding("video", "tmp_csv")
    # np.save("face_embeddings.npy", face_embeddings)
    # print(face_embeddings.shape)
    # face_embeddings = get_face_embeddings("tmp_csv")
    # np.save("face_embeddings.npy", face_embeddings)
    # print(face_embeddings.shape)
    pass