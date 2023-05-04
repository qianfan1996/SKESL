# -*-coding:utf-8-*- 

import numpy as np


def get_glove_embeddings(emb_file):
    res = {}
    with open(emb_file, "r") as f:
        while True:
            data = f.readline()
            if data:
                data = data.split(' ')
                res[data[0]] = np.array([float(i) for i in data[1:]])
            else:
                break
    return res, len(res['the'])


def extract(text, embedding, dim):
    text_list = text.split(' ')
    result = []
    for word in text_list:
        if word.lower() in embedding:
            result.append(embedding[word.lower()])
        else:
            result.append(np.zeros(dim))
    return np.array(result)

if __name__ == "__main__":
    text = "I love you."
    file_path = "/home/qianfan/Data/Glove/glove.840B.300d.txt"
    embedding, dim = get_glove_embeddings(file_path)
    feature = extract(text, embedding, dim)
    print(dim)
    print(type(feature), type(embedding))
    print(feature.shape, len(embedding))
    # print(embedding['the'])
    print(embedding['you.'])
    # print(feature)
    # print()