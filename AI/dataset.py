import torch
from torch.utils.data import Dataset
import os
import random

MAX_FILE_LEN = 100 * 1024  # 100KB
FILES_COUNT = 5000

class FileAutoEncoderDataset(Dataset):
    """
    Dataset для автоэнкодера файлов.
    
    Возвращает 3 тензора:
    - input: данные с маской (256 на повреждённых позициях), shape: (file_len,)
    - target: оригинальные данные (0-255), shape: (file_len,)
    - mask: бинарная маска (1=повреждено), shape: (file_len,)
    """
    
    def __init__(self, file_len=10240, file_dropout=0.1, data_percent=0.9, is_train=True, multiplier=1):
        if multiplier > MAX_FILE_LEN // file_len:
            raise Exception('Too much multiplier')
        
        self.file_len = file_len
        self.file_dropout = file_dropout
        self.is_train = is_train
        self.dataset_path = os.path.join('AI', 'prepared')
        
        if not os.path.exists(self.dataset_path):
            raise Exception('Dataset not found')
        
        # Загружаем игнорируемые типы
        with open(os.path.join(self.dataset_path, 'train.ignore'), 'r') as ignr:
            self.ignored_types = [line.strip() for line in ignr.readlines() if line.strip() and line[0] != '#']
        
        # Получаем список типов (папок)
        self.types = []
        for type_name in os.listdir(self.dataset_path):
            type_path = os.path.join(self.dataset_path, type_name)
            if os.path.isdir(type_path) and type_name not in self.ignored_types:
                self.types.append(type_name)
        
        if not self.types:
            raise Exception('No valid data types found in dataset')
        
        # Количество файлов на тип
        self.type_len = int(FILES_COUNT * data_percent)
        self.data_len = self.type_len * len(self.types)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # Определяем тип и индекс файла
        type_idx = index // self.type_len
        file_idx = index % self.type_len
        
        type_name = self.types[type_idx]
        type_path = os.path.join(self.dataset_path, type_name)
        
        # Получаем список файлов с данными (префикс '0' = оригинальные данные)
        if self.is_train:
            data_files = [f for f in os.listdir(type_path)[-self.type_len:FILES_COUNT] if f.startswith('0')]
        data_files = [f for f in os.listdir(type_path)[:self.type_len] if f.startswith('0')]
        
        if not data_files:
            raise FileNotFoundError(f'No data files found in {type_path}')
        
        # Выбираем файл (циклически если индекс больше количества файлов)
        file_name = data_files[file_idx]
        file_path = os.path.join(type_path, file_name)
        
        # Читаем данные
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Обрезаем или дополняем до нужной длины
        if len(data) < self.file_len:
            data = data + b'\x00' * (self.file_len - len(data))
        else:
            data = data[:self.file_len]
        
        # Преобразуем в тензор
        target = torch.tensor(list(data), dtype=torch.long)  # (file_len,)
        
        # Создаём маску повреждений
        num_damaged = int(self.file_len * self.file_dropout)
        mask_indices = random.sample(range(self.file_len), num_damaged)
        
        # Бинарная маска
        mask = torch.zeros(self.file_len, dtype=torch.float32)
        mask[mask_indices] = 1.0
        
        # Input с токеном [MASK]=256 на повреждённых позициях
        input_data = target.clone()
        input_data[mask_indices] = 256  # Специальный токен [MASK]
        
        return input_data, target, mask
