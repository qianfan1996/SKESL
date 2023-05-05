# SKESL
Sentiment Knowledge Enhanced Self-supervised Learning for Multimodal Sentiment Analysis

![The framework of SKESL](imgs/framework.jpg)

We propose a Sentiment Knowledge Enhanced Self-supervised Learning (SKESL) method,
which uses contextual and non-verbal information to predict the fine-grained sentiment intensity of a word
to learn the common sentimental patterns in opinion videos.

## 1. Cloning this repository
```bash
$ git clone https://github.com/qianfan1996/SKESL.git
```

## 2. Creating a virtual environment, and then installing required packages
```bash
$ conda create -n envir_name python=3.8
$ source activate envir_name
$ pip install -r requirements.txt
```

## 3. Datasets
Downloading the processed datasets from [Google Drive](https://drive.google.com/drive/my-drive), and putting them into data/CMU-MOSI, data/CMU-MOSEI, data/EmoVoxCeleb.
You can also process raw datasets by yourself.

Raw pretraining datasets can be acquired in [this website](https://www.robots.ox.ac.uk/~vgg/data/voxceleb) 
(You may need to apply for an account and password to get permission to download).

About processing raw datasets, see [data/CMU-MOSI](https://github.com/qianfan1996/SKESL/tree/main/data/CMU-MOSI) and [data/EmoVoxCeleb](https://github.com/qianfan1996/SKESL/tree/main/data/EmoVoxCeleb)
for relevant codes.

## 4. Running the codes
### 4.1 baseline
The baseline model is not pretrained with unlabeled video data.
```bash
$ CUDA_VISIBLE_DEVICES=0 python baseline.py
```
You can change command line arguments to train different models on different datasets and backbone language models.

### 4.2 pretraining
```bash
$ CUDA_VISIBLE_DEVICES=0 python pretrain.py
```

### 4.3 infering the sentiment using purely language models
```bash
$ CUDA_VISIBLE_DEVICES=0 python language_model_classifier.py
```
You can change command line arguments to train different models on different datasets and language models.

### 4.4 our models
```bash
$ CUDA_VISIBLE_DEVICES=0 python main.py
```
You can change command line arguments to train different models on different datasets and backbone language models.