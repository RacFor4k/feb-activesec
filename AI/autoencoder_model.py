import torch
import torch.nn as nn

"""
Embedding dim
Encoder/Decoder - массив: [фильры, ядро, смещение]
латентное простанство - модуль
"""


class FileAutoEncoder(nn.Module):
    def __init__(self, emb_dim, encoder_layers, latent_layers):
        self.emb_dim = emb_dim
        self.encoder_layers = encoder_layers
        self.latent_layers = latent_layers
        
        