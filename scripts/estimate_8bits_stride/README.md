# Stride-Based Entropy Estimators for 8-bit Data

This folder contains versions of the entropy estimators that use **every n-th sample** (stride) from the input data, allowing you to analyze how entropy estimates change as the time interval between samples increases.

## Available Estimators
- `ml_lstm.py`: LSTM-based neural estimator
- `ml_rnn.py`: GRU-based neural estimator
- `lz.py`: LZ78-based entropy estimator
- `nist.py`: NIST min-entropy estimator

## Usage
Each script supports a `--stride` argument to control the sampling interval, **and a `--bits` argument to control word length (e.g., 8 or 16 bits per symbol)**. The stride and bits values are included in the output filenames.

### Example
```bash
uv run -m scripts.estimate_8bits_stride.ml_lstm --stride 5 --bits 16
uv run -m scripts.estimate_8bits_stride.lz --stride 10 --bits 16
```

- Default stride is 2 (can also be set with the `ESTIMATOR_STRIDE` environment variable).
- Default bits per symbol is 8; set `--bits 16` for 16-bit symbols.
- Output files will be named like `bus_stride5_bits16.txt`, `radio_stride10_bits16.txt`, etc.

## Output
Results are written to the corresponding folder in `results/8bits_stride/<estimator>/`.

---

## Why Use Stride?
Using a stride allows you to investigate how entropy estimates change when you increase the interval between samples (e.g., to check for temporal correlations or memory effects in your data).

---

## See Also
- For standard estimators (using every sample), see `../estimate_8bits/`.
