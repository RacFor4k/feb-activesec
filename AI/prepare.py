import os
import sys
import tqdm

PACKET_SIZE = 10*1024 #10KB

def create_config(types):
    with open(os.path.join('prepare', 'train.ignore')) as file:
        for type in types:
            file.write(f'#{type}\n')

def proccess(path, type):
    for file in os.listdir(path):
        with open(os.path.join(path,file), 'rb') as reader:
            with open(os.path.join('prepared', type, file), 'wb') as writer:
                writer.write(reader.read(PACKET_SIZE))

def main():
    if not os.path.exists('prepared\\'):
        os.mkdir('prepared\\')
    path = sys.argv[1]
    if not os.path.exists(path):
        raise "Invalid path"
    types = os.listdir(path)
    for type in tqdm.tqdm(types):
        proccess(os.path.join(path, type), type)
    create_config(types)

if __name__ == '__main__':
    main()