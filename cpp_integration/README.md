# C++ Integration Setup

This folder contains Python code to interface with the [SP800-90B_EntropyAssessment](https://github.com/usnistgov/SP800-90B_EntropyAssessment) C++ codebase.

## Setup

1. **Clone this repo with submodules:**
   - To automatically clone the C++ repo as a submodule, use:
     ```
     git clone --recursive <this-repo-url>
     ```
   - If you already cloned without `--recursive`, run:
     ```
     git submodule update --init --recursive
     ```

2. **Build the C++ binaries:**
   - Follow the [repo instructions](https://github.com/usnistgov/SP800-90B_EntropyAssessment#building) to build the binaries (e.g., using `make`).
   - The expected path for binaries is `./SP800-90B_EntropyAssessment/bin/`.

3. **Configure the path:**
   - If your binaries are elsewhere, update `cpp_integration/config.yaml` accordingly.

4. **Use from Python:**
   - See `entropy_assessment.py` for an example Python wrapper to run the C++ binaries from your scripts.
