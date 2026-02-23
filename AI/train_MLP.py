from autoencoder_model import FileBinaryClassifierFC
from dataset import BinaryClassificationDataset

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas
import sys
import msvcrt

NUM_EPOCHES = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_model(model, criterion, loader):
    """
    Тестирование модели для бинарной классификации.
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # Метрики для бинарной классификации
    total_tp = 0  # True Positive: зашифрован, предсказан как зашифрован
    total_fp = 0  # False Positive: не зашифрован, предсказан как зашифрован
    total_tn = 0  # True Negative: не зашифрован, предсказан как не зашифрован
    total_fn = 0  # False Negative: зашифрован, предсказан как не зашифрован

    with torch.no_grad():
        progress = tqdm(loader, desc="Testing", leave=False)
        for input_data, label in progress:
            input_data = input_data.to(device)
            label = label.to(device)

            logits = model(input_data)
            loss = criterion(logits, label)

            # Предсказания
            pred = logits.argmax(dim=1)
            correct = (pred == label).sum().item()

            # Подсчёт TP, FP, TN, FN
            tp = ((pred == 1) & (label == 1)).sum().item()
            fp = ((pred == 1) & (label == 0)).sum().item()
            tn = ((pred == 0) & (label == 0)).sum().item()
            fn = ((pred == 0) & (label == 1)).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            total_samples += 1

            progress.set_postfix(loss=loss.item())

    # Вычисляем метрики
    accuracy = total_correct / total_samples / 64 if total_samples > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    npv = total_tn / (total_tn + total_fn) if (total_tn + total_fn) > 0 else 0.0  # Negative Predictive Value
    fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0  # False Positive Rate
    fnr = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0.0  # False Negative Rate
    mcc = 0.0  # Matthews Correlation Coefficient
    mcc_denom = ((total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn)) ** 0.5
    if mcc_denom > 0:
        mcc = (total_tp * total_tn - total_fp * total_fn) / mcc_denom

    return {
        'loss': total_loss / total_samples if total_samples > 0 else 0.0,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'npv': npv,
        'fpr': fpr,
        'fnr': fnr,
        'mcc': mcc,
        'tp': total_tp,
        'fp': total_fp,
        'tn': total_tn,
        'fn': total_fn,
        'total_samples': total_samples,
    }


def train_epoch(model, optimizer, criterion, loader, epoch):
    """
    Обучение модели для бинарной классификации.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)

    for input_data, label in progress:
        input_data = input_data.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        logits = model(input_data)
        loss = criterion(logits, label)

        loss.backward()
        optimizer.step()

        # Предсказания для метрик
        pred = logits.argmax(dim=1)
        correct = (pred == label).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += 1

        progress.set_postfix(loss=loss.item())

    return {
        'loss': total_loss / total_samples if total_samples > 0 else 0.0,
        'accuracy': total_correct / total_samples / 128 if total_samples > 0 else 0.0,
    }


def save_checkpoint(model, optimizer, epoch, metrics, tag='best'):
    """
    Сохранение чекпоинта модели.
    """
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

    filename = f'checkpoint_binary_{tag}_epoch{epoch}_loss{metrics["loss"]:.6f}_acc{metrics["accuracy"]:.6f}.pt'
    filepath = os.path.join(checkpoint_path, filename)
    torch.save(checkpoint, filepath)

    return filepath


def load_checkpoint(checkpoint_path=None, tag='best'):
    """
    Загрузка чекпоинта модели.
    """
    checkpoint_path_dir = os.path.join('AI', 'checkpoints')

    if checkpoint_path is None:
        if not os.path.exists(checkpoint_path_dir):
            print(f"Checkpoint directory not found: {checkpoint_path_dir}")
            return None

        checkpoints = [f for f in os.listdir(checkpoint_path_dir) if f.startswith('checkpoint_binary_') and f.endswith('.pt')]

        if not checkpoints:
            print("No checkpoints found")
            return None

        if tag is not None:
            tagged_checkpoints = [f for f in checkpoints if f.startswith(f'checkpoint_binary_{tag}_')]
            if tagged_checkpoints:
                checkpoints = tagged_checkpoints
            else:
                print(f"No checkpoints found with tag: {tag}")
                return None

        checkpoints.sort(reverse=True)
        checkpoint_path = os.path.join(checkpoint_path_dir, checkpoints[0])
        print(f"Loading latest checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    print(f"Loaded checkpoint: epoch={checkpoint['epoch']}, loss={checkpoint['metrics']['loss']:.6f}, acc={checkpoint['metrics']['accuracy']:.6f}, tag={checkpoint['tag']}")

    return checkpoint


def load_model_from_checkpoint(checkpoint, model, optimizer=None, device=None):
    """
    Восстановление модели и оптимизатора из чекпоинта.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint['epoch']


def main():
    # Параметры обучения
    FILE_LEN = 4096
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4

    # Датасеты для бинарной классификации
    train_dataset = BinaryClassificationDataset(
        file_len=FILE_LEN,
        data_percent=0.9,
        is_train=True,
        offset=1024,
        rand_offset=1024  # Случайное смещение для аугментации
    )
    test_dataset = BinaryClassificationDataset(
        file_len=FILE_LEN,
        data_percent=0.1,
        is_train=False,
        offset=0,
        rand_offset=0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8
    )

    # Модель: CNN Encoder + MLP Classifier
    model = FileBinaryClassifierFC(
        emb_dim=16,
        encoder_layers=[
            [16, 32, 11, 1, 5],
            [32, 64, 5, 2, 2],
            [64, 128, 3, 2, 1],
            [128, 256, 3, 2, 1],
            [256, 256, 3, 2, 1],
            [256, 256, 3, 2, 1],
            [256, 256, 3, 2, 1],
        ],
        is_gelu=True,
        dropout=0.15,
        fc_hidden=128
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # Загрузка чекпоинта если указан
    if len(sys.argv) > 1:
        ckpt = load_checkpoint(sys.argv[1])
        if ckpt:
            load_model_from_checkpoint(ckpt, model, optimizer, device)
            print(f'Загружена модель: {sys.argv[1]}')

    best_test_loss = float('inf')
    best_test_acc = 0.0

    # Лог для записи метрик
    train_log = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': [],
        'test_specificity': [],
        'test_npv': [],
        'test_fpr': [],
        'test_fnr': [],
        'test_mcc': [],
        'test_tp': [],
        'test_fp': [],
        'test_tn': [],
        'test_fn': [],
    }

    for epoch in range(NUM_EPOCHES):
        # Проверка нажатия клавиши 'q' для выхода
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8').lower()
            if key == 'q':
                print("\nВыход по нажатию 'q'")
                break

        # Обучение
        train_metrics = train_epoch(model, optimizer, criterion, train_loader, epoch)

        # Тестирование
        test_metrics = test_model(model, criterion, test_loader)

        # Запись в лог
        train_log['epoch'].append(epoch)
        train_log['train_loss'].append(train_metrics['loss'])
        train_log['train_accuracy'].append(train_metrics['accuracy'])
        train_log['test_loss'].append(test_metrics['loss'])
        train_log['test_accuracy'].append(test_metrics['accuracy'])
        train_log['test_precision'].append(test_metrics['precision'])
        train_log['test_recall'].append(test_metrics['recall'])
        train_log['test_f1'].append(test_metrics['f1'])
        train_log['test_specificity'].append(test_metrics['specificity'])
        train_log['test_npv'].append(test_metrics['npv'])
        train_log['test_fpr'].append(test_metrics['fpr'])
        train_log['test_fnr'].append(test_metrics['fnr'])
        train_log['test_mcc'].append(test_metrics['mcc'])
        train_log['test_tp'].append(test_metrics['tp'])
        train_log['test_fp'].append(test_metrics['fp'])
        train_log['test_tn'].append(test_metrics['tn'])
        train_log['test_fn'].append(test_metrics['fn'])

        # Вывод метрик
        print(
            f"Epoch {epoch}: "
            f"Train Loss = {train_metrics['loss']:.6f}, "
            f"Train Acc = {train_metrics['accuracy']:.4f}, "
            f"Test Loss = {test_metrics['loss']:.6f}, "
            f"Test Acc = {test_metrics['accuracy']:.4f}"
        )
        print(
            f"  Precision = {test_metrics['precision']:.4f}, "
            f"Recall = {test_metrics['recall']:.4f}, "
            f"F1 = {test_metrics['f1']:.4f}, "
            f"Specificity = {test_metrics['specificity']:.4f}"
        )
        print(
            f"  NPV = {test_metrics['npv']:.4f}, "
            f"FPR = {test_metrics['fpr']:.4f}, "
            f"FNR = {test_metrics['fnr']:.4f}, "
            f"MCC = {test_metrics['mcc']:.4f}"
        )
        print(
            f"  TP={test_metrics['tp']}, FP={test_metrics['fp']}, "
            f"TN={test_metrics['tn']}, FN={test_metrics['fn']}"
        )

        # Сохранение лучшей модели по loss
        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            save_checkpoint(model, optimizer, epoch, test_metrics, tag='best_loss')
            print(f"  [Saved] New best model by loss: {best_test_loss:.6f}")

        # Сохранение лучшей модели по accuracy
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            save_checkpoint(model, optimizer, epoch, test_metrics, tag='best_acc')
            print(f"  [Saved] New best model by accuracy: {best_test_acc:.4f}")

        # Сохранение чекпоинта каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, test_metrics, tag=f'epoch{epoch + 1}')
            print(f"  [Saved] Checkpoint at epoch {epoch + 1}")

    # Сохранение лога в CSV
    df = pandas.DataFrame(train_log)
    log_path = 'binary_classification_log.csv'
    df.to_csv(log_path, sep=';', encoding='utf-8', index=False, na_rep='NaN')
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
