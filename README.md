# entropy-estimation

This project processes data from four sources and integrates C++ code for advanced computation.

## Structure
- `collectors/`: Collectors for each source
- `cpp_integration/`: Python-C++ integration
- `notebooks/`: Jupyter notebooks
- `processed_data/`: Processed data from each source
- `raw_data/`: Raw data from each source
- `scripts/process/`: Data processing scripts
- `scripts/estimate_8bits/`: Standard entropy estimators (use every sample)
- `scripts/estimators_8bits_stride/`: Stride-based entropy estimators (use every n-th sample)
- `SP800-90B_EntropyAssessment/`: C++ entropy assessment code (as a submodule)
- `tests/`: Unit tests

## Setup

### 1. Install Homebrew (brew)
If you don't have Homebrew installed on Linux:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> ~/.profile
source ~/.profile
```

### 2. Install dependencies (Python & uv)
Install Python and uv using Homebrew:
```
brew install python uv
```

Create and activate a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install project dependencies:
```
uv pip sync uv.toml
```

### 3. Add the SP800-90B_EntropyAssessment submodule
If you have not already, add the C++ repo as a submodule:
```
git submodule add https://github.com/usnistgov/SP800-90B_EntropyAssessment.git SP800-90B_EntropyAssessment
```
Then initialize and update submodules (if needed):
```
git submodule update --init --recursive
```

### 4. Install C++ build dependencies
Install the required C++ libraries and tools:
```
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    g++ \
    libbz2-dev \
    libdivsufsort-dev \
    libjsoncpp-dev \
    libssl-dev \
    libmpfr-dev \
    libgmp-dev \
    pkg-config
```

### 5. Build the C++ binaries
Navigate to the submodule directory and run:
```
cd SP800-90B_EntropyAssessment
make
```
This will build all binaries. For specific binaries, you can use:
- `make iid` (for IID tests)
- `make non_iid` (for non-IID tests)
- `make restart` (for restart testing)
- `make conditioning` (for conditioning tests)

### 6. (Optional) Run self-tests
To verify your build, run:
```
cd selftest
./selftest
```
Any observed delta less than 1.0E-6 is considered a pass.

### 7. Configure integration
If your binaries are not in the default location, update `cpp_integration/config.yaml` with the correct path.

### 8. Use from Python
You can now use the Python wrapper to call the C++ binaries from your scripts.

## Entropy Estimator Scripts

### Standard Estimators
Located in `scripts/estimate_8bits/`, these scripts estimate entropy using every sample in the data. Includes:
- `ml_lstm.py`: LSTM-based neural estimator
- `ml_rnn.py`: GRU-based neural estimator
- `lz.py`: LZ78-based estimator
- `nist.py`: NIST min-entropy estimator

### Stride-Based Estimators
Located in `scripts/estimators_8bits_stride/`, these scripts estimate entropy using every n-th sample (stride) from the data. This allows you to analyze how entropy changes with increasing sample interval.

Each script supports a `--stride` argument (default 2). The stride value is included in the output filenames for easy comparison.

#### Example usage:
```bash
uv run -m scripts.estimators_8bits_stride.ml_lstm --stride 5
uv run -m scripts.estimators_8bits_stride.lz --stride 10
```

Output files will be named like `bus_stride5.txt`, `radio_stride10.txt`, etc.

See `scripts/estimators_8bits_stride/README.md` for details.

## Results

- Standard estimator results are saved in `results/8bits/<estimator>/`.
- Stride-based estimator results are saved in `results/8bits_stride/<estimator>/`.
- Output filenames include the data source and stride, e.g., `bus_stride5.txt`.
