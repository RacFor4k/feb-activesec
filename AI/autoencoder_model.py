import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Embedding dim
Encoder/Decoder - массив: [фильтры, ядро, смещение]
латентное пространство - модуль
"""

PRINT = False

def p(*args):
    if PRINT:
        print(args)


class FileAutoEncoder(nn.Module):
    def __init__(self, emb_dim, encoder_layers, decoder_layers, head_module=None, latent_module=None, is_gelu=False, dropout = 0.15):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.head_module = head_module
        self.latent_module = latent_module
        self.activation = nn.GELU() if is_gelu else nn.ReLU()
        self.dropout = dropout
        # 257 токенов: 0-255 = байты, 256 = [MASK]
        self.embedding = nn.Embedding(257, emb_dim)
        self._createEncoder()
        self._createDecoder()

    def _createEncoder(self):
        self.encoder = nn.ModuleList()
        for in_ch, out_ch, kernel, stride, pad in self.encoder_layers:
            self.encoder.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, stride, pad),
                nn.BatchNorm1d(out_ch),
                nn.Dropout1d(self.dropout)))

    def _createDecoder(self):
        self.decoder = nn.ModuleList()
        for in_ch, out_ch, kernel, stride, pad, out_pad in self.decoder_layers:
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch, kernel, stride, pad, out_pad),
                nn.BatchNorm1d(out_ch),
                nn.Dropout1d(self.dropout)))

    def forward(self, x):
        p(x.shape)
        x = self.embedding(x)  # (batch, seq_len, emb_dim)
        p(x.shape)
        x = x.transpose(1, 2)  # (batch, emb_dim, seq_len) for Conv1d
        p(x.shape)
        
        # skip_connections = []
        for conv in self.encoder:
            x = conv(x)
            self.activation(x)
            p(x.shape)
            # skip_connections.append(x)
        
        if self.latent_module:
            x = self.latent_module(x)
        
        # for i, conv_transpose in enumerate(self.decoder):
        #     # if i < len(skip_connections):
        #         # x = torch.cat([x, skip_connections[-(i+1)]], dim=1)
        #     self.activation(x)
        #     x = conv_transpose(x)
        #     p(x.shape)
        
        # if self.head_module:
        #     x = self.head_module(x)
        #     p(x.shape)
        # else:
        #     x = x.transpose(1, 2)  # (batch, seq_len, emb_dim)
            
        return x


class IsCryptH(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, in_size/2)
        self.fc2 = nn.Linear(in_size/2, 2)
        self.activ = nn.GELU()
    def forward(self, x):
        x = x.flatten(start_dim=1) 
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
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

    def forward(self, output, target, mask=None, return_metrics=False):
        """
        Args:
            output: Model predictions (batch, seq_len, 256) - logits for each byte value
            target: Original byte values (batch, seq_len) with values 0-255
            mask: Binary mask (batch, seq_len) where 1 = damaged position
            return_metrics: If True, return dict with loss and metrics

        Returns:
            Weighted loss scalar, or dict with metrics if return_metrics=True
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

        if not return_metrics:
            return loss

        # Compute predictions for metrics
        output_transposed = output.transpose(1, 2)  # (batch, seq_len, 256)
        pred_bytes = output_transposed.argmax(dim=-1)

        # Binary classification: correct prediction (1) vs incorrect (0)
        correct = (pred_bytes == target).float()

        # Masked positions metrics
        masked_correct = correct * mask
        masked_tp = masked_correct.sum()  # True Positives: correctly predicted masked bytes
        masked_fp = (1 - masked_correct) * mask  # False Positives: incorrectly predicted masked bytes
        masked_fp = masked_fp.sum()
        masked_tn = torch.tensor(0.0, device=loss.device)  # No TN for masked (all are positive class)
        masked_fn = torch.tensor(0.0, device=loss.device)  # No FN for masked (all are positive class)

        # Unmasked positions metrics
        unmasked_correct = correct * (1 - mask)
        unmasked_tn = unmasked_correct.sum()  # True Negatives: correctly predicted unmasked bytes
        unmasked_fp = torch.tensor(0.0, device=loss.device)  # No FP for unmasked (all are negative class)
        unmasked_fn = (1 - unmasked_correct) * (1 - mask)  # False Negatives: incorrectly predicted unmasked bytes
        unmasked_fn = unmasked_fn.sum()
        unmasked_tp = torch.tensor(0.0, device=loss.device)  # No TP for unmasked (all are negative class)

        # Total metrics
        total_tp = masked_tp + unmasked_tp
        total_fp = masked_fp + unmasked_fp
        total_tn = masked_tn + unmasked_tn
        total_fn = masked_fn + unmasked_fn

        # Accuracy metrics
        masked_acc = (masked_tp / num_masked).item() if num_masked > 0 else 0.0
        unmasked_acc = (unmasked_tn / num_unmasked).item() if num_unmasked > 0 else 0.0

        # Precision, Recall, F1 for masked positions
        precision = (masked_tp / (masked_tp + masked_fp)).item() if (masked_tp + masked_fp) > 0 else 0.0
        recall = masked_acc  # For masked positions, recall = accuracy
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'loss': loss.item(),
            'masked_loss': (self.alpha * masked_sum / num_masked).item(),
            'unmasked_loss': (self.beta * unmasked_sum / num_unmasked).item(),
            'masked_acc': masked_acc,
            'unmasked_acc': unmasked_acc,
            'masked_tp': masked_tp.item(),
            'masked_fp': masked_fp.item(),
            'masked_tn': masked_tn.item(),
            'masked_fn': masked_fn.item(),
            'unmasked_tp': unmasked_tp.item(),
            'unmasked_fp': unmasked_fp.item(),
            'unmasked_tn': unmasked_tn.item(),
            'unmasked_fn': unmasked_fn.item(),
            'total_tp': total_tp.item(),
            'total_fp': total_fp.item(),
            'total_tn': total_tn.item(),
            'total_fn': total_fn.item(),
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
class FCLatent(nn.Module):
    def __init__(self, channels=256, seq_len=32, latent_dim=1024):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.flat_size = channels * seq_len  # 256 * 64 = 16384

        # Сжимаем в латентное пространство
        self.fc_enc = nn.Linear(self.flat_size, latent_dim)

        # Разжимаем обратно до нужного количества элементов
        self.fc_dec = nn.Linear(latent_dim, self.flat_size)
        self.activation = nn.GELU()
    def forward(self, x):
        B = x.size(0)

        # 1. Сплющиваем: (Batch, 16384)
        x = x.flatten(start_dim=1)

        # 2. Пропускаем через FC (получаем глобальный контекст)
        latent = self.activation(self.fc_enc(x))  # (Batch, 1024)

        # 3. Разжимаем обратно: (Batch, 16384)
        x = self.activation(self.fc_dec(latent))

        # 4. ВОССТАНАВЛИВАЕМ РАЗМЕРНОСТЬ: (Batch, 256, 64)
        x = x.view(B, self.channels, self.seq_len)

        return x


# =============================================================================
# Binary Classification Models (Encoder + Classifier)
# =============================================================================

class FileBinaryClassifierFC(nn.Module):
    """
    Модель для бинарной классификации файлов с полносвязным классификатором.
    Encoder -> Global Pooling -> FC Classifier

    Input: (batch, seq_len) - байты файла (0-255)
    Output: (batch, 2) - логиты [не зашифрован, зашифрован]
    """
    def __init__(self, emb_dim=128, encoder_layers=None, is_gelu=False, dropout=0.15, fc_hidden=256):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoder_layers = encoder_layers or [[64, 3, 1], [128, 3, 2], [256, 3, 2]]
        self.activation = nn.GELU() if is_gelu else nn.ReLU()
        self.dropout = dropout

        # 256 токенов: 0-255 = байты (без маски)
        self.embedding = nn.Embedding(256, emb_dim)
        self._createEncoder()

        # Вычисляем размер после энкодера
        self._calc_encoded_size()
        self.classifier = nn.Sequential(
            nn.Linear(self.encoded_size, fc_hidden),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 2)
        )

    def _createEncoder(self):
        self.encoder = nn.ModuleList()
        for in_ch, out_ch, kernel, stride, pad in self.encoder_layers:
            self.encoder.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, stride, pad),
                nn.BatchNorm1d(out_ch),
                nn.Dropout1d(self.dropout)))

    def _calc_encoded_size(self):
        # Вычисляем размер после всех слоёв энкодера
        with torch.no_grad():
            dummy = torch.zeros(1, 1024, dtype=torch.long)
            x = self.embedding(dummy)
            x = x.transpose(1, 2)
            for conv in self.encoder:
                x = conv(x)
                x = self.activation(x)
            self.encoded_size = x.shape[1]  # channels

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, emb_dim)
        x = x.transpose(1, 2)  # (batch, emb_dim, seq_len)

        for conv in self.encoder:
            x = conv(x)
            x = self.activation(x)

        # Global Average Pooling
        x = x.mean(dim=2)  # (batch, channels)

        # Classifier
        logits = self.classifier(x)  # (batch, 2)
        return logits


class FileBinaryClassifierTransformer(nn.Module):
    """
    Модель для бинарной классификации файлов с трансформер-классификатором.
    Encoder -> Transformer Encoder -> Global Pooling -> FC Classifier

    Input: (batch, seq_len) - байты файла (0-255)
    Output: (batch, 2) - логиты [не зашифрован, зашифрован]
    """
    def __init__(self, emb_dim=128, encoder_layers=None, transformer_layers=2, 
                 transformer_heads=4, transformer_ff_ratio=4, is_gelu=False, dropout=0.15):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoder_layers = encoder_layers or [[64, 3, 1], [128, 3, 2], [256, 3, 2]]
        self.activation = nn.GELU() if is_gelu else nn.ReLU()
        self.dropout = dropout

        # 256 токенов: 0-255 = байты (без маски)
        self.embedding = nn.Embedding(256, emb_dim)
        self._createEncoder()

        # Вычисляем размер после энкодера
        self._calc_encoded_size()

        # Transformer encoder
        transformer_dim = self.encoded_size
        transformer_ff = transformer_dim * transformer_ff_ratio
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff,
            dropout=dropout,
            activation='gelu' if is_gelu else 'relu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, 2)
        )

    def _createEncoder(self):
        self.encoder = nn.ModuleList()
        for in_ch, out_ch, kernel, stride, pad in self.encoder_layers:
            self.encoder.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, stride, pad),
                nn.BatchNorm1d(out_ch),
                nn.Dropout1d(self.dropout)))

    def _calc_encoded_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1024, dtype=torch.long)
            x = self.embedding(dummy)
            x = x.transpose(1, 2)
            for conv in self.encoder:
                x = conv(x)
                x = self.activation(x)
            self.encoded_size = x.shape[1]  # channels

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, emb_dim)
        x = x.transpose(1, 2)  # (batch, emb_dim, seq_len)

        for conv in self.encoder:
            x = conv(x)
            x = self.activation(x)
        # x: (batch, channels, seq_len)

        # Transformer expects (seq_len, batch, channels)
        x = x.permute(2, 0, 1)  # (seq_len, batch, channels)

        x = self.transformer(x)  # (seq_len, batch, channels)

        # Global Average Pooling
        x = x.mean(dim=0)  # (batch, channels)

        # Classifier
        logits = self.classifier(x)  # (batch, 2)
        return logits


class FileBinaryClassifierAttention(nn.Module):
    """
    Модель для бинарной классификации файлов с attention-классификатором.
    Encoder -> Self-Attention -> Global Pooling -> FC Classifier

    Input: (batch, seq_len) - байты файла (0-255)
    Output: (batch, 2) - логиты [не зашифрован, зашифрован]
    """
    def __init__(self, emb_dim=128, encoder_layers=None, attention_heads=4, 
                 attention_dim=256, is_gelu=False, dropout=0.15):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoder_layers = encoder_layers or [[64, 3, 1], [128, 3, 2], [256, 3, 2]]
        self.activation = nn.GELU() if is_gelu else nn.ReLU()
        self.dropout = dropout

        # 256 токенов: 0-255 = байты (без маски)
        self.embedding = nn.Embedding(256, emb_dim)
        self._createEncoder()

        # Вычисляем размер после энкодера
        self._calc_encoded_size()

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.encoded_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=False
        )
        self.attention_norm = nn.LayerNorm(self.encoded_size)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.encoded_size, attention_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 2)
        )

    def _createEncoder(self):
        self.encoder = nn.ModuleList()
        for in_ch, out_ch, kernel, stride, pad in self.encoder_layers:
            self.encoder.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, stride, pad),
                nn.BatchNorm1d(out_ch),
                nn.Dropout1d(self.dropout)))

    def _calc_encoded_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1024, dtype=torch.long)
            x = self.embedding(dummy)
            x = x.transpose(1, 2)
            for conv in self.encoder:
                x = conv(x)
                x = self.activation(x)
            self.encoded_size = x.shape[1]  # channels

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, emb_dim)
        x = x.transpose(1, 2)  # (batch, emb_dim, seq_len)

        for conv in self.encoder:
            x = conv(x)
            x = self.activation(x)
        # x: (batch, channels, seq_len)

        # Attention expects (seq_len, batch, channels)
        x = x.permute(2, 0, 1)  # (seq_len, batch, channels)

        attn_output, _ = self.attention(x, x, x)
        x = self.attention_norm(x + attn_output)  # (seq_len, batch, channels)

        # Global Average Pooling
        x = x.mean(dim=0)  # (batch, channels)

        # Classifier
        logits = self.classifier(x)  # (batch, 2)
        return logits
