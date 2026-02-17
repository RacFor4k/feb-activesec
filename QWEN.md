# Feb-ActiveSec Project

## Project Overview

**Feb-ActiveSec** is a research/educational project focused on **file analysis and recovery using deep learning**. The project builds an autoencoder neural network capable of reconstructing damaged/corrupted file data.

### Main Components

| Component | Description |
|-----------|-------------|
| **AI Model** | Convolutional autoencoder with skip connections for byte-level file reconstruction |
| **Dataset Pipeline** | Tools for generating, processing, and encrypting file datasets |
| **Analysis Tools** | Statistical analysis (entropy, chi-square) and visualization of binary files |
| **Loss Function** | Custom masked loss that prioritizes recovery of damaged bytes |

### Architecture

```
feb-activesec/
├── AI/
│   ├── autoencoder_model.py  # FileAutoEncoder + AutoEncoderLoss (masked loss)
│   ├── dataset.py            # PyTorch Dataset for file data
│   ├── prepare.py            # Dataset preprocessing with AES encryption
│   └── KAN_lib.py            # (Empty - reserved for Kolmogorov-Arnold Networks)
├── generate_dataset.py       # Generate synthetic dataset (10 folders × 100 files × 10KB)
├── calc.py                   # Parallel entropy/chi-square calculation
├── process_data.py           # Aggregate statistics into Excel
├── cut_dataset.py            # Truncate files to fixed size
├── file_visualizer.py        # GUI to visualize binary files as images
└── processed.xlsx            # Output statistics
```

---

## Technologies & Dependencies

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | Main language |
| **PyTorch** | Deep learning framework |
| **cryptography** | AES encryption for dataset preprocessing |
| **openpyxl** | Excel output for statistics |
| **Pillow (PIL)** | File visualization |
| **tqdm** | Progress bars |

### Required Packages (inferred)

```bash
pip install torch cryptography openpyxl pillow tqdm
```

---

## Building and Running

### 1. Generate Synthetic Dataset

```bash
python generate_dataset.py
```
Creates `gen/` folder with 10 subfolders (`0-total` to `9-total`), each containing 100 files of 10KB random data.

### 2. Calculate File Statistics (Entropy, Chi-Square)

```bash
python calc.py
```
- Processes files in `datasets/` folder
- Uses `ProcessPoolExecutor` with 16 workers
- Outputs CSV files to `statistics/` folder
- Metrics: Shannon entropy, normalized chi-square

### 3. Aggregate Statistics

```bash
python process_data.py
```
- Reads CSV files from `statistics/`
- Computes mean, median, standard deviation
- Outputs `processed.xlsx`

### 4. Prepare Dataset for AI Training

```bash
python AI/prepare.py <path_to_dataset>
```
- Encrypts files using AES-256-CBC
- Creates `AI/prepared/` with processed data
- Uses `ThreadPoolExecutor` (4 workers) for parallel processing
- Each file type processed in separate worker thread

### 5. Visualize Binary Files

```bash
python file_visualizer.py
```
- Opens GUI for visual inspection of binary files
- Supports grayscale and RGB modes
- Stretches data to fit window

---

## Key Implementation Details

### AutoEncoder Architecture (`AI/autoencoder_model.py`)

```python
FileAutoEncoder(
    emb_dim=128,           # Embedding dimension for byte tokens
    encoder_layers=[[64, 3, 1], ...],  # [filters, kernel, stride]
    decoder_chanels=[...],
    is_gelu=False          # Use ReLU or GELU activation
)
```

- **Input**: Byte sequences (embedded as 256-dimensional tokens)
- **Encoder**: Conv1d + BatchNorm layers
- **Latent**: Optional custom module
- **Decoder**: ConvTranspose1d with skip connections
- **Output**: Reconstructed byte probabilities

### Masked Loss Function (`AutoEncoderLoss`)

```python
criterion = AutoEncoderLoss(
    base_criterion=nn.MSELoss(),
    alpha=5.0,   # Weight for damaged (masked) positions
    beta=1.0     # Weight for intact positions
)
loss = criterion(output, target, mask=None)
```

**Key insight**: Prioritizes accuracy on damaged bytes vs. intact bytes.

### Dataset (`AI/dataset.py`)

```python
FileAutoEncoderDataset(
    file_len=10240,      # Sequence length
    file_dropout=0.1,    # 10% bytes masked/damaged
    data_percent=0.9,    # Fraction of dataset to use
    is_train=True,
    multiplier=1
)
```

### Parallel Processing Patterns

| File | Pattern | Workers |
|------|---------|---------|
| `calc.py` | `ProcessPoolExecutor` | 16 (CPU-bound) |
| `AI/prepare.py` | `ThreadPoolExecutor` | 4 (I/O-bound) |
| `AI/prepare.py::process()` | `threading.Thread` | Async encryption per file |

---

## Development Conventions

### Code Style
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Comments**: Russian language comments in code
- **Imports**: Standard library first, then third-party

### File Naming
- Dataset folders: `{id}-{type}/` (e.g., `0-total/`)
- Processed files: `{0|1}{4-digit-index}` (0=raw, 1=encrypted)

### Error Handling
- Custom exceptions with descriptive messages
- File existence checks before operations

---

## Usage Notes

### Dataset Structure

```
AI/prepared/
├── train.ignore          # List of ignored file types
└── {type}/               # File type folder
    ├── 00000, 00001...   # Original (truncated) data
    └── 10000, 10001...   # AES-encrypted data
```

### Masking Strategy (for training)

Current implementation uses **zero replacement** for damaged bytes. Consider:

| Strategy | Status |
|----------|--------|
| Zero replacement | ⚠️ Conflicts with real zero bytes |
| Random noise | ✅ Better |
| Mask channel | ✅✅ Recommended (requires arch change) |
| Learnable [MASK] token | ✅ Best (Embedding(257)) |

### AES Encryption Details

- **Algorithm**: AES-256-CBC
- **Key**: SHA256(b'activesec')
- **IV**: Fixed zeros (`b'0000000000000000'`) ⚠️ (should be random per file)
- **Padding**: PKCS7

---

## Known Issues / TODOs

1. **`AI/dataset.py`**: `__getitem__` has incorrect path logic and raises `FileExistsError` inappropriately
2. **Fixed IV**: Using same IV for all encryption reduces security
3. **Masking**: Zero replacement conflicts with real zero bytes in data
4. **`cut_dataset.py`**: Bug in `os.path.path()` (should be `os.path.isfile()`)
5. **No requirements.txt**: Dependencies not explicitly listed

---

## Quick Reference

```bash
# Generate test data
python generate_dataset.py

# Analyze files
python calc.py

# Prepare AI dataset
python AI/prepare.py ./gen

# Visualize binary
python file_visualizer.py
```
