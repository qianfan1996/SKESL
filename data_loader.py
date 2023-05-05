# -*-coding:utf-8-*-
import pickle
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# load CMU-MOSI dataset
def load_mosi_pkl(file_path, mode='train'):
    with open(file_path, 'rb') as file:
        info = pickle.load(file)
        raw_text = list(info[mode]['raw_text'])
        audio = info[mode]['audio']
        video = info[mode]['video']
        label = info[mode]['labels']
    return raw_text, audio, video, label

# load CMU-MOSEI dataset
def load_mosei_pkl(file_path, mode='train'):
    with open(file_path, 'rb') as file:
        info = pickle.load(file)
        raw_text = list(info[mode]['raw_text'])
        audio = info[mode]['audio']
        video = info[mode]['video']
        label = info[mode]['labels']
    return raw_text, audio, video, label


# load EmoVoxCeleb dataset
def load_emovoxceleb_pkl(file_path):
    with open(file_path, 'rb') as file:
        info = pickle.load(file)
        raw_text = list(info['raw_text'])
        audio = info['audio']
        video = info['video']
    return raw_text, audio, video


# load Vader lexicon, get a dictionary
def load_lexicon(path):
    with open(path, "r") as file:
        lexicon = {line.split()[0]: line.split()[1] for line in file.readlines()}
        # The following 4 words are phrases in the vader lexicon
        lexicon['fed'] = '-1.8'
        lexicon['screwed'] = '-2.2'
        lexicon['('] = '0.0'
        lexicon["can't"] = '-2.0'
    return lexicon


# tokenizer, padding and truncating to unified length
"""
def encode_words(text, tokenizer):
    return tokenizer.encode(text, padding="max_length", truncation=True, max_length=39, return_tensors="pt")
"""
def encode_words_bert(text, tokenizer):
    dic = tokenizer(text, padding="max_length", truncation=True, max_length=39, return_tensors="pt")
    input_ids = dic['input_ids']
    token_type_ids = dic['token_type_ids']
    attention_mask = dic['attention_mask']
    return input_ids, token_type_ids, attention_mask


# mask a word whose sentiment score absolute value is maximum in the sentence
def mask_text_bert(text, lexicon, tokenizer):
    tokens = tokenizer.tokenize(text)
    dic = tokenizer(text, padding="max_length", truncation=True, max_length=39, return_tensors="pt")
    token_type_ids = dic['token_type_ids']
    attention_mask = dic['attention_mask']
    Tomasked_words = [token for token in tokens if token in lexicon.keys()]
    if len(Tomasked_words) == 0:
        label = 0.0 # label is the sentiment score of masked token.
        index = random.randint(0, min(len(tokens)-1, 36)) # index is position of "[MASK]" token
        tokens[index] = "[MASK]"
        masked_text = tokenizer.convert_tokens_to_string(tokens)
        input_ids = tokenizer.encode(masked_text, padding='max_length', truncation=True, max_length=39, return_tensors='pt')
        return input_ids, token_type_ids, attention_mask, label, index+1
    else:
        scores = []
        for word in Tomasked_words:
            scores.append(float(lexicon[word]))
        Tomasked_word = Tomasked_words[scores.index(max(scores,key=abs))]
        label = float(lexicon[Tomasked_word]) # label is the sentiment score of masked token
        masked_text = ["[MASK]" if token == Tomasked_word else token for token in tokens]
        index = masked_text.index("[MASK]") # index is position of "[MASK]" token
        if index > 36:
            masked_text = masked_text[index-20:index+19]
            index = 20
        masked_text = tokenizer.convert_tokens_to_string(masked_text)
        input_ids = tokenizer.encode(masked_text, padding='max_length', truncation=True, max_length=39, return_tensors='pt')
        return input_ids, token_type_ids, attention_mask, round(label*0.75, 3), index+1


# construct a pretrain BERT dataloader
class PretrainDataset(Dataset):
    def __init__(self, raw_text, audio, video, lexicon, tokenizer_name, device):
        self.text = raw_text
        self.audio = audio
        self.video = video
        self.lexicon = lexicon
        self.size = len(self.text)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = device

    def _to_tensor(self, input_ids, token_type_ids, attention_mask, audio, video, label):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        audio = torch.FloatTensor(audio).to(self.device)
        video = torch.FloatTensor(video).to(self.device)
        label = torch.FloatTensor([label]).to(self.device)
        return input_ids, token_type_ids, attention_mask, audio, video, label

    def __getitem__(self, i):
        text, audio, video = self.text[i], self.audio[i], self.video[i]
        input_ids, token_type_ids, attention_mask, label, index = mask_text_bert(text, self.lexicon, self.tokenizer)
        input_ids, token_type_ids, attention_mask, audio, video, label = self._to_tensor(input_ids, token_type_ids, attention_mask, audio, video, label)
        return input_ids, token_type_ids, attention_mask, audio, video, label, index

    def __len__(self):
        return int(self.size)


# construct a BERT sentiment classification dataloader without nonverbal behavior
class SentiDatasetNoAV(Dataset):
    def __init__(self, raw_text, label, tokenizer_name, device):
        self.text = raw_text
        self.label = label
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.size = len(self.text)
        self.device = device

    def _to_tensor(self, input_ids, token_type_ids, attention_mask, label):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        label = torch.FloatTensor([label]).to(self.device)
        return input_ids, token_type_ids, attention_mask, label

    def __getitem__(self, index):
        text, label = self.text[index], self.label[index]
        input_ids, token_type_ids, attention_mask = encode_words_bert(text, self.tokenizer)
        input_ids, token_type_ids, attention_mask, label = self._to_tensor(input_ids, token_type_ids, attention_mask, label)
        return input_ids, token_type_ids, attention_mask, label

    def __len__(self):
        return int(self.size)


# construct a BERT sentiment classification dataloader with nonverbal behavior
class SentiDataset(Dataset):
    def __init__(self, raw_text, audio, video, label, tokenizer_name, device):
        self.text = raw_text
        self.audio = audio
        self.video = video
        self.label = label
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.size = len(self.text)
        self.device = device

    def _to_tensor(self, input_ids, token_type_ids, attention_mask, audio, video, label):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        audio = torch.FloatTensor(audio).to(self.device)
        video = torch.FloatTensor(video).to(self.device)
        label = torch.FloatTensor([label]).to(self.device)
        return input_ids, token_type_ids, attention_mask, audio, video, label

    def __getitem__(self, index):
        text, audio, video, label = self.text[index], self.audio[index], self.video[index], self.label[index]
        input_ids, token_type_ids, attention_mask = encode_words_bert(text, self.tokenizer)
        input_ids, token_type_ids, attention_mask, audio, video, label = self._to_tensor(input_ids, token_type_ids, attention_mask, audio, video, label)
        return input_ids, token_type_ids, attention_mask, audio, video, label

    def __len__(self):
        return int(self.size)


class DatasetWithGlove(Dataset):
    def __init__(self, text, audio, vision, label, device):
        self.text = text
        self.audio = audio
        self.vision = vision
        self.label = label
        self.size = len(self.text)
        self.device = device

    def _to_tensor(self, text, audio, vision, label):
        text = torch.FloatTensor(text).to(self.device)
        audio = torch.FloatTensor(audio).to(self.device)
        vision = torch.FloatTensor(vision).to(self.device)
        label = torch.FloatTensor(label).to(self.device)
        return text, audio, vision, label

    def __getitem__(self, index):
        text, audio, vision, label = self.text[index], self.audio[index], self.vision[index], self.label[index]
        text, audio, vision, label = self._to_tensor(text, audio, vision, label)
        return text, audio, vision, label

    def __len__(self):
        return int(self.size)


if __name__ == "__main__":
    lexicon = load_lexicon("data/vader_lexicon.txt")
    print(type(lexicon))
    for key, value in lexicon.items():
        try:
            value = float(value)
        except:
            print(key, value)
    print(lexicon["good"], lexicon["bad"], lexicon["nice"], lexicon["fool"])
    # ================================================================================
    # raw_text, audio, video, label = load_mosi_pkl("data/CMU-MOSI/mosi.pkl", "train")
    # print(type(raw_text), type(audio), type(video), type(label))
    # print(len(raw_text), audio.shape, video.shape, len(label))
    # print(raw_text[:5])
    # print(label[:5])
    # ================================================================================
    # text = "ANYHOW IT WAS REALLY GOOD"
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # input_ids = encode_words(text, tokenizer)
    # print(type(input_ids))
    # print(input_ids.size())
    # print(input_ids)
    # ================================================================================
    # text = "ANYHOW IT WAS REALLY GOOD"
    # tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    # input_ids = encode_words(text, tokenizer)
    # print(type(input_ids))
    # print(input_ids.size())
    # print(input_ids)
    # ================================================================================