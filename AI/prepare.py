import os
import sys
import tqdm

PACKET_SIZE = 100*1024 #100KB

def create_config(types):
    with open(os.path.join('prepare', 'train.ignore')) as file:
        for type in types:
            file.write(f'#{type}\n')

def proccess(path, type):
    str().lower
    for i, file in enumerate([file for file in os.listdir(path) if file.split('.')[0].lower() != type]):
        with open(os.path.join(path,file), 'rb') as reader:
            with open(os.path.join('prepared', type, f'{i:04d}'), 'wb') as writer:
                writer.write(reader.read(PACKET_SIZE))

def main():
    if not os.path.exists('prepared\\'):
        os.mkdir('prepared\\')
    path = sys.argv[1]
    if not os.path.exists(path):
        raise "Invalid path"
    types = [type[-6].lower() for type in os.listdir(path) if not os.path.isfile(type)]
    for type in tqdm.tqdm(types):
        proccess(os.path.join(path, type), type)
    create_config(types)

if __name__ == '__main__':
    main()