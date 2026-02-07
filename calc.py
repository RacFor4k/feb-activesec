import sys
import math
import os
import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Настройки
MIN_SIZE_BYTES = 512
READ_BUFFER = 1024 * 1024  # Читаем по 1 МБ за раз для скорости

def calculate_metrics_worker(file_info):
    """
    Функция-воркер для отдельного процесса.
    Принимает кортеж (file_path, filename).
    Возвращает словарь с результатами или None.
    """
    file_path, filename = file_info
    
    # Предварительная проверка имени и ID
    name_stem, ext = os.path.splitext(filename)
    file_id_str = name_stem.split("-")[0]
    
    if not file_id_str.isdigit():
        return None

    try:
        # Быстрая проверка размера перед чтением
        stat = os.stat(file_path)
        file_size = stat.st_size
        if file_size < MIN_SIZE_BYTES:
            return None
    except (OSError, IOError):
        return None

    # --- Чтение и подсчет ---
    byte_counts = [0] * 256
    
    try:
        with open(file_path, 'rb') as f:
            while True:
                # Читаем большими блоками
                chunk = f.read(READ_BUFFER)
                if not chunk:
                    break
                for byte in chunk:
                    byte_counts[byte] += 1
    except (OSError, IOError):
        return None

    # --- Вычисления (Math) ---
    # Энтропия
    entropy = 0.0
    # Хи-квадрат
    chi2 = 0.0
    expected_count = file_size / 256.0
    
    # Объединенный цикл для ускорения (немного быстрее, чем два цикла)
    if file_size > 0:
        for count in byte_counts:
            # Энтропия
            if count > 0:
                p = count / file_size
                entropy -= p * math.log2(p)
            
            # Chi2
            diff = count - expected_count
            chi2 += (diff * diff) / expected_count
    else:
        return None

    return {
        'ID': int(file_id_str),
        'SIZE': file_size,
        'ENTROPY': round(entropy, 6),
        'CHI2': round(chi2, 2)
    }

def process_subfolder(subfolder_path, output_csv_path, max_workers):
    """
    Обрабатывает одну подпапку: собирает файлы, запускает процессы, пишет CSV.
    """
    tasks = []
    
    # 1. Сбор всех файлов в подпапке (рекурсивно)
    print(f"Сканирование: {subfolder_path} ...")
    for root, _, files in os.walk(subfolder_path):
        for filename in files:
            full_path = os.path.join(root, filename)
            tasks.append((full_path, filename))

    if not tasks:
        print(f"  -> Файлов не найдено.")
        return

    print(f"  -> Найдено файлов: {len(tasks)}. Обработка в {max_workers} потоках...")
    
    results = []
    
    # 2. Параллельная обработка
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Отправляем задачи
        future_to_file = {executor.submit(calculate_metrics_worker, task): task for task in tasks}
        
        # Получаем результаты по мере готовности
        for future in as_completed(future_to_file):
            res = future.result()
            if res:
                results.append(res)

    # 3. Запись в CSV (если есть результаты)
    if results:
        # Сортируем по ID для красоты
        results.sort(key=lambda x: x['ID'])
        
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['ID', 'SIZE', 'ENTROPY', 'CHI2']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writerows(results)
            print(f"  -> Готово! Сохранено в: {output_csv_path}")
            print(f"  -> Записано строк: {len(results)}")
        except IOError as e:
            print(f"  -> Ошибка записи CSV: {e}")
    else:
        print("  -> Нет подходящих файлов (фильтр по имени или размеру).")

def main():
    root_folder = './datasets'
    start_time = time.time()

    # Получаем список подпапок первого уровня
    # Например: datasets/folder1, datasets/folder2 ...
    subfolders = [
        f.path for f in os.scandir(root_folder) 
        if f.is_dir()
    ]

    print(f"Найдено подпапок для обработки: {len(subfolders)}")
    print(f"Минимальный размер файла: {MIN_SIZE_BYTES} Bytes")
    print("-" * 50)

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        # Имя CSV файла будет соответствовать имени подпапки
        csv_name = f"./statistics/{folder_name}.csv"
        process_subfolder(folder, csv_name, 16)
        print("-" * 50)

    total_time = time.time() - start_time
    print(f"Полное завершение работы за {total_time:.2f} сек.")

if __name__ == "__main__":
    try:
        os.mkdir('statistics')
    finally:
        main()
