# -*-coding:utf-8-*- 
import os
from get_emovoxceleb_speaker import get_chosen_speaker
from ignored_av_segment import ignored_av_segment as ig


def get_audio_number(path):
    num = 0
    for dir in sorted(os.listdir(path)):
        if dir in get_chosen_speaker("EmoVoxCeleb_meta.csv"):
            for audio_name in sorted(os.listdir(os.path.join(path, dir))):
                if not audio_name.startswith('.'):
                    num += 1
    return num


def get_video_number(path):
    num = 0
    for dir in sorted(os.listdir(path)):
        if dir in get_chosen_speaker("EmoVoxCeleb_meta.csv"):
            for video_name in sorted(os.listdir(os.path.join(path, dir, "1.6"))):
                for clip_name in sorted(os.listdir(os.path.join(path, dir, "1.6", video_name))):
                    num += 1
    return num


def get_audio_name(path):
    names = []
    for dir in sorted(os.listdir(path)):
        if dir in get_chosen_speaker("EmoVoxCeleb_meta.csv"):
            for audio_name in sorted(os.listdir(os.path.join(path, dir))):
                if not dir+"_"+audio_name[:-12] in ig.ignored_audio_segment:
                    if not audio_name.startswith('.'):
                        names.append(dir+"_"+audio_name[:-4])
    return names


def get_video_name(path):
    names = []
    for dir in sorted(os.listdir(path)):
        if dir in get_chosen_speaker("EmoVoxCeleb_meta.csv"):
            for video_name in sorted(os.listdir(os.path.join(path, dir, "1.6"))):
                for clip_name in sorted(os.listdir(os.path.join(path, dir, "1.6", video_name))):
                    if not dir+"_"+video_name+"_"+"000000"+clip_name in ig.ignored_video_segment:
                        names.append(dir+"_"+video_name+"_"+"0"*(7-len(clip_name))+clip_name)
    return names


if __name__ == "__main__":
    # num_audio = get_audio_number("audio")
    # print("The number of all audio clips is ", num_audio)
    # num_video = get_video_number("video")
    # print("The number of all video clips is ", num_video)
    audio_names = get_audio_name("audio")
    video_names = get_video_name("video")
    print(len(audio_names), len(video_names))
    print(audio_names[:10])
    print(video_names[:10])
    audio_names_set = set(audio_names)
    video_names_set = set(video_names)
    print(len(audio_names_set), len(video_names_set))
    print(audio_names_set.intersection(video_names_set)==video_names_set)
    print(audio_names_set.union(video_names_set)==audio_names_set)
    print(video_names_set.difference(audio_names_set))
    print(audio_names_set.difference(video_names_set))
    print(len(audio_names_set.difference(video_names_set)))