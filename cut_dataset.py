import os
import sys
import tqdm

SIZE_LIMIT = 100 * 1024 #512KB 

def cut(path,tq):
    for file in os.listdir(path):
        with open(os.path.join(path,file), "r+b") as f:
            f.truncate(SIZE_LIMIT)
            tq.update(1)

def main():
    path = sys.argv[1]
    if not os.path.exists(path):
        raise "Invalid path"
    types = [i for i in os.listdir(path) if not os.path.path(i)]
    tq = tqdm.tqdm(total=5000*len(types))
    for type in types:
        cut(os.path.join(path, type),tq)

if __name__ == '__main__':
    main()