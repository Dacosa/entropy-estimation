# entropy-estimation

This project processes data from four sources and integrates C++ code for advanced computation.

## Structure
- `data_sources/`: Data from each source
- `cpp_integration/`: Python-C++ integration
- `scripts/`: Processing scripts
- `notebooks/`: Jupyter notebooks
- `tests/`: Unit tests
- `SP800-90B_EntropyAssessment/`: C++ entropy assessment code (as a submodule)

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

### 4. Install C++ build dependencies (with brew)
Use Homebrew to install the required C++ libraries and tools:
```
brew install gcc mpfr bzip2 jsoncpp openssl libdivsufsort
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
