# -*-coding:utf-8-*- 
import os
import pickle
from get_emovoxceleb_speaker import get_chosen_speaker
from ignored_av_segment import ignored_av_segment as ig

def get_all_text(path):
    texts = []
    for dir in sorted(os.listdir(path)):
        if dir in get_chosen_speaker("EmoVoxCeleb_meta.csv"):
            for text_name in sorted(os.listdir(os.path.join(path, dir))):
                if not dir + "_" + text_name[:-12] in ig.ignored_audio_segment:
                    if not text_name.startswith('.'):
                        with open(os.path.join(path, dir, text_name),"r") as file:
                            for line in file.readlines():
                                texts.append(line.strip())
    return texts

if __name__ == "__main__":
    texts = get_all_text("text")
    print(len(texts))
    print(texts[:10])