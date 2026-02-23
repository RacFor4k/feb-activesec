import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def parse_training_log(file_path):
    epochs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue # Пропускаем пустые строки
            
            # 1. Проверка на формат CSV (epoch,train,test,acc)
            parts = line.split(',')
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    epochs.append({
                        'epoch': int(parts[0]),
                        'train_loss': float(parts[1]),
                        'test_loss': float(parts[2]),
                        'accuracy': float(parts[3])
                    })
                    continue
                except ValueError:
                    pass

            # 2. Проверка на текстовый формат (Epoch 0: Train Loss = ...)
            if "Epoch" in line and "Train Loss =" in line:
                import re
                pattern = r"Epoch\s+(\d+):.*?Train Loss\s*=\s*([\d.]+).*?Test Loss\s*=\s*([\d.]+).*?Unmasked Acc\s*=\s*([\d.]+)"
                match = re.search(pattern, line)
                if match:
                    epochs.append({
                        'epoch': int(match.group(1)),
                        'train_loss': float(match.group(2)),
                        'test_loss': float(match.group(3)),
                        'accuracy': float(match.group(4))
                    })

    df = pd.DataFrame(epochs)
    if not df.empty:
        df = df.drop_duplicates(subset='epoch', keep='last').sort_values('epoch')
    return df

def plot_metrics(df):
    # Используем стандартный стиль, так как seaborn может быть не установлен
    plt.figure(figsize=(12, 10))
    
    # График Loss
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', alpha=0.7)
    ax1.plot(df['epoch'], df['test_loss'], label='Test Loss', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_title('Model Loss (Log Scale)')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # График Accuracy
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df['epoch'], df['accuracy'], label='Accuracy', color='green', linewidth=2)
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.02)
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig('training_plots.png')
    print("\n[Успех] Графики сохранены в файл training_plots.png")
    plt.show()

if __name__ == "__main__":
    # Если файл не передан аргументом, ищем log.txt
    file_to_open = sys.argv[1] if len(sys.argv) > 1 else 'log.txt'
    
    if not os.path.exists(file_to_open):
        print(f"Ошибка: Файл '{file_to_open}' не найден.")
    else:
        data = parse_training_log(file_to_open)
        if data.empty:
            print("Данные не найдены. Проверьте, что в файле есть строки формата:")
            print("199,0.005449,0.002308,0.9996")
        else:
            print(f"Загружено эпох: {len(data)}")
            plot_metrics(data)