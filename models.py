# -*-coding:utf-8-*- 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

# ---------------------------------------------------------------------------------------------------------------------
# construct a pre-trained language model
class PretrainedLanguageModelBertXLNet(nn.Module):
    def __init__(self, pretrained_language_model_name):
        super(PretrainedLanguageModelBertXLNet, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_language_model_name)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        return output


class PretrainedLanguageModel(nn.Module):
    def __init__(self, pretrained_language_model_name):
        super(PretrainedLanguageModel, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_language_model_name)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        return output


class PretrainedLanguageModelRoBerta(nn.Module):
    def __init__(self, pretrained_language_model_name):
        super(PretrainedLanguageModelRoBerta, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_language_model_name)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return output

# ---------------------------------------------------------------------------------------------------------------------
# non-verbal information injection, i.e., multimodal fusion
class CrossModalAttention(nn.Module):
    def __init__(self, modality1_dim, modality2_dim, embed_dim, attn_dropout=0.5):
        super(CrossModalAttention, self).__init__()
        self.modality1_dim = modality1_dim
        self.modality2_dim = modality2_dim
        self.embed_dim = embed_dim
        self.modality1_ln = nn.LayerNorm(self.modality1_dim)
        self.modality2_ln = nn.LayerNorm(self.modality2_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.scaling = self.embed_dim ** -0.5
        self.proj_modality1 = nn.Linear(self.modality1_dim, self.embed_dim)
        self.proj_modality2_k = nn.Linear(self.modality2_dim, self.embed_dim)
        self.proj_modality2_v = nn.Linear(self.modality2_dim, self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, self.modality1_dim)

    def forward(self, modality1, modality2):
        q = self.proj_modality1(self.modality1_ln(modality1))
        k = self.proj_modality2_k(self.modality2_ln(modality2))
        v = self.proj_modality2_v(self.modality2_ln(modality2))
        attention = F.softmax(torch.bmm(q, k.permute(0, 2, 1)) * self.scaling, dim=-1)
        context = torch.bmm(attention, v)
        output = self.proj(context)
        # output = self.attn_dropout(output)
        # output = output + self.modality1_ln(modality1)
        # output = output + modality1
        return output


class CrossmodalEncoderLayer(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, embed_dim, attn_dropout=0.5):
        super(CrossmodalEncoderLayer, self).__init__()
        self.cma_a = CrossModalAttention(text_dim, audio_dim, embed_dim)
        self.cma_v = CrossModalAttention(text_dim, video_dim, embed_dim)
        self.layernorm = nn.LayerNorm(text_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.fc = nn.Linear(text_dim, text_dim)

    def forward(self, text, audio, video):
        # output = self.cma_a(text, audio) + self.cma_v(text, video) + self.layernorm(text)
        output = self.cma_a(text, audio) + self.cma_v(text, video) + text
        residual = output
        output = self.fc(self.layernorm(output))
        output = self.attn_dropout(output)
        output = output + residual
        return output


class CrossmodalEncoder(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, embed_dim, num_layers=4, attn_dropout=0.5):
        super(CrossmodalEncoder, self).__init__()
        self.encoderlayer = CrossmodalEncoderLayer(text_dim, audio_dim, video_dim, embed_dim, attn_dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])
        for layer in range(self.num_layers):
            new_layer = self.encoderlayer
            self.layers.append(new_layer)

    def forward(self, text, audio, video):
        output = text
        for layer in self.layers:
            output = layer(output, audio, video)
        return output

# ---------------------------------------------------------------------------------------------------------------------
# construct a pre-train model which contains a pre-trained language model and a multimodel fusion module
"""
class PretrainModel(nn.Module):
    def __init__(self, pretrained_language_model_name, text_dim, audio_dim, video_dim, embed_dim, fc_dim, num_layers=4, attn_dropout=0.5, fc_dropout=0.5):
        super(PretrainModel, self).__init__()
        self.pretrained_language_model = PretrainedLanguageModel(pretrained_language_model_name)
        self.crossmodal_encoder = CrossmodalEncoder(text_dim, audio_dim, video_dim, embed_dim, num_layers, attn_dropout)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def feature_extractor(self, text, audio, video):
        output = self.pretrained_language_model(text)
        output = self.crossmodal_encoder(output, audio, video)
        output = torch.mean(output, dim=1)
        return output

    def fitter(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, text, audio, video):
        output = self.feature_extractor(text, audio, video)
        output = self.fitter(output)
        return output
"""
"""
class PretrainModel(nn.Module):
    def __init__(self, pretrained_language_model_name, text_dim, audio_dim, video_dim, embed_dim, fc_dim, num_layers=4, attn_dropout=0.5, fc_dropout=0.5):
        super(PretrainModel, self).__init__()
        self.pretrained_language_model = PretrainedLanguageModel(pretrained_language_model_name)
        self.crossmodal_encoder = CrossmodalEncoder(text_dim, audio_dim, video_dim, embed_dim, num_layers, attn_dropout)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def feature_extractor(self, text, audio, video):
        output = self.pretrained_language_model(text)
        output = self.crossmodal_encoder(output, audio, video)
        # output = torch.mean(output, dim=1)
        return output

    def fitter(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, text, audio, video, index, bs):
        output = self.feature_extractor(text, audio, video)
        output = self.fitter(output[torch.tensor(range(bs)),index,:])
        return output
"""
class PretrainModelBertXLNet(nn.Module):
    def __init__(self, pretrained_language_model_name, text_dim, audio_dim, video_dim, embed_dim, fc_dim, num_layers=4, attn_dropout=0.5, fc_dropout=0.5):
        super(PretrainModelBertXLNet, self).__init__()
        self.pretrained_language_model = PretrainedLanguageModelBertXLNet(pretrained_language_model_name)
        self.crossmodal_encoder = CrossmodalEncoder(text_dim, audio_dim, video_dim, embed_dim, num_layers, attn_dropout)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def feature_extractor(self, input_ids, token_type_ids, attention_mask, audio, video):
        output = self.pretrained_language_model(input_ids, token_type_ids, attention_mask)
        output = self.crossmodal_encoder(output, audio, video)
        # output = torch.mean(output, dim=1)
        return output

    def fitter(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, input_ids, token_type_ids, attention_mask, audio, video, index, bs):
        output = self.feature_extractor(input_ids, token_type_ids, attention_mask, audio, video)
        output = self.fitter(output[torch.tensor(range(bs)),index,:])
        return output


class PretrainModel(nn.Module):
    def __init__(self, pretrained_language_model_name, text_dim, audio_dim, video_dim, embed_dim, fc_dim, num_layers=4, attn_dropout=0.5, fc_dropout=0.5):
        super(PretrainModel, self).__init__()
        self.pretrained_language_model = PretrainedLanguageModel(pretrained_language_model_name)
        self.crossmodal_encoder = CrossmodalEncoder(text_dim, audio_dim, video_dim, embed_dim, num_layers, attn_dropout)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def feature_extractor(self, input_ids, token_type_ids, attention_mask, audio, video):
        output = self.pretrained_language_model(input_ids, token_type_ids, attention_mask)
        output = self.crossmodal_encoder(output, audio, video)
        # output = torch.mean(output, dim=1)
        return output

    def fitter(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, input_ids, token_type_ids, attention_mask, audio, video, index, bs):
        output = self.feature_extractor(input_ids, token_type_ids, attention_mask, audio, video)
        output = self.fitter(output[torch.tensor(range(bs)),index,:])
        return output


class PretrainModelRoBerta(nn.Module):
    def __init__(self, pretrained_language_model_name, text_dim, audio_dim, video_dim, embed_dim, fc_dim, num_layers=4, attn_dropout=0.5, fc_dropout=0.5):
        super(PretrainModelRoBerta, self).__init__()
        self.pretrained_language_model = PretrainedLanguageModelRoBerta(pretrained_language_model_name)
        self.crossmodal_encoder = CrossmodalEncoder(text_dim, audio_dim, video_dim, embed_dim, num_layers, attn_dropout)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def feature_extractor(self, input_ids, attention_mask, audio, video):
        output = self.pretrained_language_model(input_ids, attention_mask)
        output = self.crossmodal_encoder(output, audio, video)
        # output = torch.mean(output, dim=1)
        return output

    def fitter(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, input_ids, attention_mask, audio, video, index, bs):
        output = self.feature_extractor(input_ids, attention_mask, audio, video)
        output = self.fitter(output[torch.tensor(range(bs)),index,:])
        return output

# ---------------------------------------------------------------------------------------------------------------------
# just use pre-trained language model without non-verbal information for infering the sentiment
class LanguageModelClassifierBertXLNet(nn.Module):
    def __init__(self, pretrained_language_model, text_dim, fc_dim, fc_dropout=0.5):
        super(LanguageModelClassifierBertXLNet, self).__init__()
        self.pretrained_language_model = pretrained_language_model
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def predict(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.pretrained_language_model(input_ids, token_type_ids, attention_mask)
        output = torch.mean(output, dim=1)
        # output = output[:, 0, :]
        output = self.predict(output)
        return output


class LanguageModelClassifierRoBerta(nn.Module):
    def __init__(self, pretrained_language_model, text_dim, fc_dim, fc_dropout=0.5):
        super(LanguageModelClassifierRoBerta, self).__init__()
        self.pretrained_language_model = pretrained_language_model
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def predict(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, input_ids, attention_mask):
        output = self.pretrained_language_model(input_ids, attention_mask)
        output = torch.mean(output, dim=1)
        # output = output[:, 0, :]
        output = self.predict(output)
        return output

# ---------------------------------------------------------------------------------------------------------------------
# After pre-training, fine-tuning the model with labeled data
class PredictModelBertXLNet(nn.Module):
    def __init__(self, pretrained_language_model, crossmodal_encoder, text_dim, fc_dim, fc_dropout=0.5):
        super(PredictModelBertXLNet, self).__init__()
        self.pretrained_language_model = pretrained_language_model
        self.crossmodal_encoder = crossmodal_encoder
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def predict(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, input_ids, token_type_ids, attention_mask, audio, video):
        output = self.pretrained_language_model(input_ids, token_type_ids, attention_mask)
        output = self.crossmodal_encoder(output, audio, video)
        output = torch.mean(output, dim=1)
        # output = output[:,0,:]
        output = self.predict(output)
        return output


class PredictModelRoBerta(nn.Module):
    def __init__(self, pretrained_language_model, crossmodal_encoder, text_dim, fc_dim, fc_dropout=0.5):
        super(PredictModelRoBerta, self).__init__()
        self.pretrained_language_model = pretrained_language_model
        self.crossmodal_encoder = crossmodal_encoder
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def predict(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, input_ids, attention_mask, audio, video):
        output = self.pretrained_language_model(input_ids, attention_mask)
        output = self.crossmodal_encoder(output, audio, video)
        output = torch.mean(output, dim=1)
        # output = output[:,0,:]
        output = self.predict(output)
        return output


if __name__ == "__main__":
    # attention = CrossModalAttention(512, 300, 256)
    # x = torch.rand(32, 20, 512)
    # y = torch.rand(32, 30, 300)
    # output = attention(x, y)
    # print(output.size())
    # ==========================================================
    # attention = CrossmodalEncoderLayer(768, 33, 709, 256)
    # x = torch.rand(32, 20, 768)
    # y = torch.rand(32, 400, 33)
    # z = torch.rand(32, 55, 709)
    # output = attention(x, y, z)
    # print(output.size())
    # ==========================================================
    attention = CrossmodalEncoder(768, 33, 709, 256)
    x = torch.rand(32, 20, 768)
    y = torch.rand(32, 400, 33)
    z = torch.rand(32, 55, 709)
    output = attention(x, y, z)
    print(output.size())
    # ==========================================================