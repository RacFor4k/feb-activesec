import torch
from torch.utils.data import dataset

class FileAutoEncoderDataset(dataset.Dataset):
    def __init__(self):
        