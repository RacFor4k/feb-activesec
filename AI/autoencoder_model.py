import torch
import torch.nn as nn

"""
Embedding dim
Encoder/Decoder - массив: [фильры, ядро, смещение]
латентное простанство - модуль
"""


class FileAutoEncoder(nn.Module):
    def __init__(self, emb_dim, encoder_layers, decoder_chanels, head_module = None, latent_module = None, is_gelu = False):
        self.emb_dim = emb_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = reversed(encoder_layers)
        self.decoder_chanels = decoder_chanels
        self.head_module = head_module
        self.latent_module = latent_module
        self.activation = nn.GELU() if is_gelu else nn.ReLU() 
        self.embedding = nn.Embedding(256, emb_dim)
        self._createEncoder()
        self._createDecoder()
        self.skip_conn = []

    def _createEncoder(self):
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(nn.Conv1d(self.emb_dim, self.encoder_layers[0][0], self.encoder_layers[0][1], self.encoder_layers[0][2]),
                                          nn.BatchNorm1d(self.encoder_layers[0][0])))
        for i, layer_data in enumerate(self.encoder_layers,1):
            self.encoder.append(nn.Sequential(nn.Conv1d(self.encoder_layers[i-1][0],layer_data[0], layer_data[1], layer_data[2]),
                                              nn.BatchNorm1d(layer_data[0])))
    
    def _createDecoder(self):
        self.decoder = nn.ModuleList()
        for i, layer_data in enumerate(self.decoder_layers):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose1d(
                    layer_data[0], 
                    self.decoder_chanels[i], 
                    layer_data[1], 
                    layer_data[2]),
                    nn.BatchNorm1d(self.decoder_chanels[i])))


    def forward(self, x):
        x = self.embedding(x)
        for conv in self.encoder:
            x = conv(x)
            self.skip_conn.append(x)
        if self.latent_module:
            x = self.latent_module(x)
        for conv_transpose in self.decoder:
            x = torch.cat([x, self.skip_conn], dim=1)
            x = conv_transpose(x)
        if self.head_module:
            x = self.head_module(x)
        return x

# class AutoEncoderLoss(nn.Module):
#     def __init__(self, base_criterion = nn.BCEWithLogitsLoss):
