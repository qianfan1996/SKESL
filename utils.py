# -*-coding:utf-8-*- 
import numpy as np
import os
import random
import torch
from thop.profile import profile
from thop import clever_format


# set random seed
def set_random_seed(seed):
    print("Random Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num


def get_flops(model, *input):
    macs, num_params = profile(model, input, verbose=False)
    macs, num_params = clever_format([macs, num_params], '%.3f')
    return macs, num_params


# Define a timing function
def interval_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs