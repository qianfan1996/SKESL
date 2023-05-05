# -*-coding:utf-8-*-
import os
import numpy as np
import pickle
import time
import argparse
from tqdm import tqdm, trange
from visdom import Visdom

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import PretrainModel
from data_loader import load_lexicon, PretrainDataset
from utils import set_random_seed, get_parameter_number, interval_time

start = time.time()

parser = argparse.ArgumentParser(description="some optional arguments")
parser.add_argument("--seed", type=int, default=666, help="random seed")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epoch")
parser.add_argument("--pretrained_language_model_name", type=str, choices=["bert-base-uncased", "bert-large-uncased"],
                    default="bert-base-uncased", help="pretrain language model which is used"
)
args = parser.parse_args()

set_random_seed(args.seed)

bs = 32
if "large" in args.pretrained_language_model_name:
    text_dim = 1024
else:
    text_dim = 768
audio_dim = 33
video_dim = 709
embed_dim = 256
fc_dim = 256
to_save_epoch = [1, 5, 10, 20, 50, 100]

raw_text = pickle.load(open("data/EmoVoxCeleb/texts.pkl", "rb"))
audio_data = np.load("data/EmoVoxCeleb/audio_embeddings.npy")
video_data = np.load("data/EmoVoxCeleb/face_embeddings.npy")
lexicon = load_lexicon("data/vader_lexicon.txt")

tokenizer_name = args.pretrained_language_model_name

print("Using {} as backbone pretrain language model and training {} epochs".format(args.pretrained_language_model_name, args.num_epochs))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = PretrainDataset(raw_text, audio_data, video_data, lexicon, tokenizer_name, device)

dataloader = DataLoader(dataset=data, batch_size=bs, shuffle=True)

model = PretrainModel(args.pretrained_language_model_name, text_dim, audio_dim, video_dim, embed_dim, fc_dim)

for param in model.pretrained_language_model.parameters():
    param.requires_grad = False

print("Total parameters: {}, Trainable parameters: {}".format(*get_parameter_number(model)))

optim = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

criterion = torch.nn.L1Loss()
model = model.to(device)

# viz = Visdom()
# viz.line([[0.]], [0], win='pretrain', opts=dict(title='pretrain loss'))

def train_epoch(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in tqdm(iterator):
        input_ids, token_type_ids, attention_mask, audio, vision, label, index = batch
        input_ids, token_type_ids, attention_mask = input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze()
        bs = input_ids.size()[0]

        optimizer.zero_grad()

        output = model(input_ids, token_type_ids, attention_mask, audio, vision, index, bs)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

if not os.path.exists("saved_models/pretrain/"+args.pretrained_language_model_name):
    os.makedirs("saved_models/pretrain/"+args.pretrained_language_model_name)

for epoch in trange(args.num_epochs):
    start_time = time.time()
    train_loss = train_epoch(model, dataloader, optim, criterion)
    # viz.line([train_loss], [epoch], win="pretrain", update="append")

    if epoch+1 in to_save_epoch:
        print("Saving pretrained model to saved_models/pretrain/{}/model_epoch{}.pth ...".format(
            args.pretrained_language_model_name,
            epoch+1)
        )

        torch.save(model, "saved_models/pretrain/{}/model_epoch{}.pth".format(
            args.pretrained_language_model_name,
            epoch+1)
        )

        print("Saved pretrained model to saved_models/pretrain/{}/model_epoch{}.pth !".format(
            args.pretrained_language_model_name,
            epoch+1)
        )

    end_time = time.time()
    epoch_mins, epoch_secs = interval_time(start_time, end_time)
    print("Epoch: {} | Train Loss: {} | Time: {}m {}s".format(epoch+1, train_loss, epoch_mins, epoch_secs))

print("Time Usage: {} minutes {} seconds".format(*interval_time(start, time.time())))