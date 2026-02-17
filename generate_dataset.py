import os
from pathlib import Path

BASE_DIR = Path(__file__).parent / "gen"
FOLDERS_COUNT = 10
FILES_PER_FOLDER = 100
FILE_SIZE = 10 * 1024  # 10 KB in bytes


def generate_dataset():
    # Create base directory
    BASE_DIR.mkdir(exist_ok=True)
    
    # Generate random data once (10 KB)
    random_data = os.urandom(FILE_SIZE)
    
    for i in range(FOLDERS_COUNT):
        # Create folder [i]-total
        folder_path = BASE_DIR / f"{i}-total"
        folder_path.mkdir(exist_ok=True)
        
        for j in range(FILES_PER_FOLDER):
            # Create file with random data
            file_path = folder_path / f"file_{j}.bin"
            file_path.write_bytes(random_data)
        
        print(f"Created folder {folder_path.name} with {FILES_PER_FOLDER} files")
    
    print(f"\nDataset generated in {BASE_DIR}")
    print(f"Total: {FOLDERS_COUNT} folders × {FILES_PER_FOLDER} files × {FILE_SIZE} bytes")


if __name__ == "__main__":
    generate_dataset()
