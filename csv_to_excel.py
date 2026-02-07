import pandas as pd
import os
import glob

def merge_csv_to_excel(source_folder, output_filename):
    # Получаем список всех csv файлов в указанной папке
    search_path = os.path.join(source_folder, '*.csv')
    csv_files = glob.glob(search_path)

    if not csv_files:
        print(f"В папке '{source_folder}' не найдено CSV файлов.")
        return

    print(f"Найдено файлов: {len(csv_files)}. Начинаю объединение...")

    # Создаем объект для записи в Excel
    # engine='openpyxl' нужен для работы с xlsx
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        for filename in csv_files:
            try:
                # Читаем CSV файл
                # Если у вас разделитель точка с запятой, замените sep=',' на sep=';'
                # Если проблемы с кодировкой (каракули), добавьте encoding='cp1251'
                df = pd.read_csv(filename, sep=',') 
                
                # Получаем имя файла без пути и расширения для названия листа
                base_name = os.path.basename(filename)
                sheet_name = os.path.splitext(base_name)[0]
                
                # Ограничение Excel: имя листа не может быть длиннее 31 символа
                sheet_name = sheet_name[:31]
                
                # Записываем данные на отдельный лист
                # index=False, чтобы не добавлять лишний столбец с нумерацией строк
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"Файл '{base_name}' добавлен как лист '{sheet_name}'")
                
            except Exception as e:
                print(f"Ошибка при обработке файла {filename}: {e}")

    print(f"\nГотово! Результат сохранен в файл: {output_filename}")

# --- НАСТРОЙКИ ---
# Укажите путь к папке с вашими CSV файлами
# '.' означает текущую папку, где лежит скрипт
folder_path = './' 

# Имя итогового файла
result_file = 'result.xlsx'

# Запуск функции
merge_csv_to_excel(folder_path, result_file)
