import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Embedding dim
Encoder/Decoder - массив: [фильтры, ядро, смещение]
латентное пространство - модуль
"""


class FileAutoEncoder(nn.Module):
    def __init__(self, emb_dim, encoder_layers, decoder_chanels, head_module=None, latent_module=None, is_gelu=False):
        self.emb_dim = emb_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = list(reversed(encoder_layers))
        self.decoder_chanels = decoder_chanels
        self.head_module = head_module
        self.latent_module = latent_module
        self.activation = nn.GELU() if is_gelu else nn.ReLU()
        # 257 токенов: 0-255 = байты, 256 = [MASK]
        self.embedding = nn.Embedding(257, emb_dim)
        self._createEncoder()
        self._createDecoder()

    def _createEncoder(self):
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(
            nn.Conv1d(self.emb_dim, self.encoder_layers[0][0], self.encoder_layers[0][1], self.encoder_layers[0][2]),
            nn.BatchNorm1d(self.encoder_layers[0][0])))
        for i, layer_data in enumerate(self.encoder_layers[1:], 1):
            self.encoder.append(nn.Sequential(
                nn.Conv1d(self.encoder_layers[i-1][0], layer_data[0], layer_data[1], layer_data[2]),
                nn.BatchNorm1d(layer_data[0])))

    def _createDecoder(self):
        self.decoder = nn.ModuleList()
        for i, layer_data in enumerate(self.decoder_layers):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose1d(layer_data[0], self.decoder_chanels[i], layer_data[1], layer_data[2]),
                nn.BatchNorm1d(self.decoder_chanels[i])))

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, emb_dim)
        x = x.transpose(1, 2)  # (batch, emb_dim, seq_len) for Conv1d
        
        skip_connections = []
        for conv in self.encoder:
            x = conv(x)
            skip_connections.append(x)
        
        if self.latent_module:
            x = self.latent_module(x)
        
        for i, conv_transpose in enumerate(self.decoder):
            if i < len(skip_connections):
                x = torch.cat([x, skip_connections[-(i+1)]], dim=1)
            x = conv_transpose(x)
        
        if self.head_module:
            x = self.head_module(x)
        else:
            x = x.transpose(1, 2)  # (batch, seq_len, emb_dim)

        return x


class ByteLogitsHead(nn.Module):
    """
    Применяет FC слой к каждому вектору эмбеддинга для получения логитов распределения байтов.
    
    Input: (batch, emb_dim, seq_len) или (batch, seq_len, emb_dim)
    Output: (batch, seq_len, 256) - логиты для каждого из 256 значений байта
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 256)
    
    def forward(self, x):
        # x может быть (batch, emb_dim, seq_len) или (batch, seq_len, emb_dim)
        if x.dim() == 3:
            if x.shape[1] == self.fc.in_features:
                # (batch, emb_dim, seq_len) -> (batch, seq_len, emb_dim)
                x = x.transpose(1, 2)
            elif x.shape[2] == self.fc.in_features:
                # Уже (batch, seq_len, emb_dim)
                pass
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}, expected emb_dim={self.fc.in_features}")
        
        # (batch, seq_len, emb_dim) -> (batch, seq_len, 256)
        return self.fc(x)


class AutoEncoderLoss(nn.Module):
    """
    Masked loss function for file recovery autoencoder.
    
    Uses CrossEntropyLoss for byte classification (0-255).
    Prioritizes accuracy on masked (damaged) positions.

    Args:
        base_criterion: Base loss function (default: CrossEntropyLoss)
        alpha: Weight for loss on masked (damaged) positions. Higher = prioritize recovery.
        beta: Weight for loss on unmasked (intact) positions. Lower = less important.

    Example:
        criterion = AutoEncoderLoss(alpha=5.0, beta=1.0)
        loss = criterion(output, target, mask)
    """
    def __init__(self, base_criterion=None, alpha=5.0, beta=1.0):
        super().__init__()
        if base_criterion is None:
            base_criterion = nn.CrossEntropyLoss(reduction='none')
        self.base_criterion = base_criterion
        self.alpha = alpha  # Weight for masked (damaged) areas
        self.beta = beta    # Weight for unmasked (intact) areas

    def forward(self, output, target, mask=None):
        """
        Args:
            output: Model predictions (batch, seq_len, 256) - logits for each byte value
            target: Original byte values (batch, seq_len) with values 0-255
            mask: Binary mask (batch, seq_len) where 1 = damaged position
        
        Returns:
            Weighted loss scalar
        """
        # Ensure correct shapes
        if output.dim() == 3:
            # (batch, seq_len, 256) -> (batch, 256, seq_len) for CrossEntropy
            output = output.transpose(1, 2)
        
        if target.dim() == 2:
            # (batch, seq_len)
            pass
        else:
            target = target.squeeze(-1)
        
        # Create mask if not provided
        if mask is None:
            mask = torch.zeros_like(target, dtype=torch.float32)
        
        # Compute per-position loss (batch, seq_len)
        per_position_loss = self.base_criterion(output, target)
        
        # Split loss into masked and unmasked components
        masked_loss = per_position_loss * mask
        unmasked_loss = per_position_loss * (1 - mask)
        
        # Sum losses with weights
        masked_sum = masked_loss.sum()
        unmasked_sum = unmasked_loss.sum()
        
        # Normalize by number of elements in each region
        num_masked = mask.sum().clamp(min=1)
        num_unmasked = (1 - mask).sum().clamp(min=1)

        # Weighted average loss
        loss = (self.alpha * masked_sum / num_masked) + (self.beta * unmasked_sum / num_unmasked)

        return loss