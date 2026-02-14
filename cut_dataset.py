import os
import sys
import tqdm

SIZE_LIMIT = 1024 * 1024 #1MB 

def cut(path):
    for file in os.listdir(path):
        with open(os.path.join(path,file), "r+b") as f:
            f.truncate(SIZE_LIMIT)

def main():
    path = sys.argv[1]
    if not os.path.exists(path):
        raise "Invalid path"
    types = os.listdir(path)
    for type in tqdm.tqdm(types):
        cut(os.path.join(path, type))