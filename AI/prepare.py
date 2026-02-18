import os
import sys
import tqdm
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

PACKET_SIZE = 100 * 1024  # 100KB
MAX_WORKERS = 4

KEY = hashlib.sha256(b'activesec').digest()

def create_config(types):
    with open(os.path.join('AI', 'prepared', 'train.ignore'), 'w') as file:
        for type in types:
            file.write(f'#{type[:-6].lower()}\n')


def encrypt(data, path):
    cipher = Cipher(algorithms.AES(KEY), modes.CBC(b'0000000000000000'), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    encrypted = encryptor.update(padder.update(data) + padder.finalize()) + encryptor.finalize()
    with open(path, 'wb') as f:
        f.write(encrypted)


def process(path, type):
    prepared_type_dir = os.path.join('AI', 'prepared', type[:-6].lower())
    if not os.path.exists(prepared_type_dir):
        os.mkdir(prepared_type_dir)

    files = [file for file in os.listdir(path) if file.split('.')[0].lower() != type[:-6].lower()]
    threads = []

    for i, file in enumerate(files):
        src_path = os.path.join(path, file)
        dst_path_0 = os.path.join(prepared_type_dir, f'0{i:04d}')
        dst_path_1 = os.path.join(prepared_type_dir, f'1{i:04d}')
        if os.path.exists(dst_path_0):
            continue
        with open(src_path, 'rb') as reader:
            data = reader.read(PACKET_SIZE)

        # Run encryption asynchronously in separate thread
        t = threading.Thread(target=encrypt, args=(data, dst_path_1), daemon=False)
        t.start()
        threads.append(t)

        # Write unencrypted data
        with open(dst_path_0, 'wb') as writer:
            writer.write(data)



    # Wait for all encryption threads to complete
    for t in threads:
        t.join()


def main():
    prepared_dir = os.path.join('AI', 'prepared')
    if not os.path.exists(prepared_dir):
        os.mkdir(prepared_dir)

    path = sys.argv[1]
    if not os.path.exists(path):
        raise FileNotFoundError("Invalid path")

    types = [type for type in os.listdir(path) if not os.path.isfile(os.path.join(path, type))]

    # Process each type in a separate worker thread
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm.tqdm(executor.map(lambda t: process(os.path.join(path, t), t), types),
                      total=len(types), desc='Processing types'))

    create_config(types)


if __name__ == '__main__':
    main()
