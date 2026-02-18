from autoencoder_model import FileAutoEncoder, ByteLogitsHead, AutoEncoderLoss
from dataset import FileAutoEncoderDataset

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


NUM_EPOCHES = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(model, criterion, loader):
    model.eval()
    total_loss = 0
    total_masked_loss = 0
    total_unmasked_loss = 0
    total_masked_acc = 0
    total_unmasked_acc = 0
    total_masked_tp = 0
    total_masked_fp = 0
    total_masked_tn = 0
    total_masked_fn = 0
    total_unmasked_tp = 0
    total_unmasked_fp = 0
    total_unmasked_tn = 0
    total_unmasked_fn = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_samples = 0

    with torch.no_grad():
        progress = tqdm(loader, desc="Testing", leave=False)
        for input, target, mask in progress:
            input = input.to(device)
            target = target.to(device)
            mask = mask.to(device)

            output = model(input)
            metrics = criterion(output, target, mask, return_metrics=True)

            total_loss += metrics['loss']
            total_masked_loss += metrics['masked_loss']
            total_unmasked_loss += metrics['unmasked_loss']
            total_masked_acc += metrics['masked_acc']
            total_unmasked_acc += metrics['unmasked_acc']
            total_masked_tp += metrics['masked_tp']
            total_masked_fp += metrics['masked_fp']
            total_masked_tn += metrics['masked_tn']
            total_masked_fn += metrics['masked_fn']
            total_unmasked_tp += metrics['unmasked_tp']
            total_unmasked_fp += metrics['unmasked_fp']
            total_unmasked_tn += metrics['unmasked_tn']
            total_unmasked_fn += metrics['unmasked_fn']
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']

            total_samples += 1
            progress.set_postfix(loss=metrics['loss'])

    return {
        'loss': total_loss / total_samples,
        'masked_loss': total_masked_loss / total_samples,
        'unmasked_loss': total_unmasked_loss / total_samples,
        'masked_acc': total_masked_acc / total_samples,
        'unmasked_acc': total_unmasked_acc / total_samples,
        'masked_tp': total_masked_tp / total_samples,
        'masked_fp': total_masked_fp / total_samples,
        'masked_tn': total_masked_tn / total_samples,
        'masked_fn': total_masked_fn / total_samples,
        'unmasked_tp': total_unmasked_tp / total_samples,
        'unmasked_fp': total_unmasked_fp / total_samples,
        'unmasked_tn': total_unmasked_tn / total_samples,
        'unmasked_fn': total_unmasked_fn / total_samples,
        'precision': total_precision / total_samples,
        'recall': total_recall / total_samples,
        'f1': total_f1 / total_samples,
    }


def train_epoch(model, optimizer, criterion, loader, epoch):
    model.train()
    total_loss = 0

    progress = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)

    for input, target, mask in progress:
        input = input.to(device)
        target = target.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        output = model(input)
        loss = criterion(output, target, mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def save_checkpoint(model: FileAutoEncoder, optimizer, epoch, metrics, tag='best'):
    checkpoint_path = os.path.join('AI', 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'tag': tag
    }

    filename = f'checkpoint_{tag}_epoch{epoch}_loss{metrics["loss"]:.6f}.pt'
    filepath = os.path.join(checkpoint_path, filename)
    torch.save(checkpoint, filepath)

    return filepath


def load_checkpoint(checkpoint_path=None, tag='best'):
    """
    Загружает чекпоинт модели.
    
    Args:
        checkpoint_path: Полный путь к файлу чекпоинта (приоритет)
        tag: Тег для поиска (например, 'best', 'epoch5'). 
             Если None, загружается последний доступный чекпоинт.
    
    Returns:
        dict с ключами: 'epoch', 'model_state_dict', 'optimizer_state_dict', 'metrics', 'tag'
        или None если чекпоинт не найден
    """
    checkpoint_path_dir = os.path.join('AI', 'checkpoints')
    
    if checkpoint_path is None:
        if not os.path.exists(checkpoint_path_dir):
            print(f"Checkpoint directory not found: {checkpoint_path_dir}")
            return None
        
        # Получаем список всех чекпоинтов
        checkpoints = [f for f in os.listdir(checkpoint_path_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
        
        if not checkpoints:
            print("No checkpoints found")
            return None
        
        # Фильтруем по тегу если указан
        if tag is not None:
            tagged_checkpoints = [f for f in checkpoints if f.startswith(f'checkpoint_{tag}_')]
            if tagged_checkpoints:
                checkpoints = tagged_checkpoints
            else:
                print(f"No checkpoints found with tag: {tag}")
                return None
        
        # Сортируем по имени файла (последний = с наибольшим номером эпохи/loss)
        checkpoints.sort(reverse=True)
        checkpoint_path = os.path.join(checkpoint_path_dir, checkpoints[0])
        print(f"Loading latest checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    print(f"Loaded checkpoint: epoch={checkpoint['epoch']}, loss={checkpoint['metrics']['loss']:.6f}, tag={checkpoint['tag']}")
    
    return checkpoint


def load_model_from_checkpoint(checkpoint, model, optimizer=None, device=None):
    """
    Восстанавливает модель и оптимизатор из чекпоинта.
    
    Args:
        checkpoint: dict от load_checkpoint()
        model: экземпляр модели для загрузки весов
        optimizer: оптимизатор для загрузки состояния (опционально)
        device: устройство для загрузки весов
    
    Returns:
        model, optimizer
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def main():
    train_dataset = FileAutoEncoderDataset(data_percent=0.9)
    test_dataset = FileAutoEncoderDataset(data_percent=0.1,is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    model = FileAutoEncoder(
        16,
        [
            [16, 32, 11, 1, 5],
            [32, 64, 5, 1, 2],
            [64, 128, 3, 1, 1],
            [128, 256, 3, 2, 1],
        ],
        [
            [512, 128, 3, 2, 1, 1],  # kernel=4 для компенсации: (5119-1)*2 + 4 = 10240
            [256, 64, 3, 1, 1, 0],
            [128, 32, 5, 1, 2, 0],
            [64, 16, 11, 1, 5, 0]
        ],
        ByteLogitsHead(16),
        is_gelu=True
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = AutoEncoderLoss(alpha=10,beta=0.5)

    best_test_loss = float('inf')

    for epoch in range(NUM_EPOCHES):
        train_loss = train_epoch(
            model, optimizer, criterion, train_loader, epoch
        )

        test_metrics = test_model(
            model, criterion, test_loader
        )

        print(
            f"Epoch {epoch}: "
            f"Train Loss = {train_loss:.6f}, "
            f"Test Loss = {test_metrics['loss']:.6f}, "
            f"Masked Loss = {test_metrics['masked_loss']:.6f}, "
            f"Unmasked Loss = {test_metrics['unmasked_loss']:.6f}, "
            f"Masked Acc = {test_metrics['masked_acc']:.4f}, "
            f"Unmasked Acc = {test_metrics['unmasked_acc']:.4f}, "
            f"Precision = {test_metrics['precision']:.4f}, "
            f"Recall = {test_metrics['recall']:.4f}, "
            f"F1 = {test_metrics['f1']:.4f}"
        )
        print(
            f"  Masked TP={test_metrics['masked_tp']:.2f}, FP={test_metrics['masked_fp']:.2f}, "
            f"TN={test_metrics['masked_tn']:.2f}, FN={test_metrics['masked_fn']:.2f}"
        )
        print(
            f"  Unmasked TP={test_metrics['unmasked_tp']:.2f}, FP={test_metrics['unmasked_fp']:.2f}, "
            f"TN={test_metrics['unmasked_tn']:.2f}, FN={test_metrics['unmasked_fn']:.2f}"
        )

        # Save best model
        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            save_checkpoint(model, optimizer, epoch, test_metrics, tag='best')
            print(f"  [Saved] New best model with Test Loss = {best_test_loss:.6f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, test_metrics, tag=f'FAE')
            print(f"  [Saved] Checkpoint at epoch {epoch+1}")


if __name__ == "__main__":
    main()
