# Stride-Based Entropy Estimators for 8-bit Data

This folder contains versions of the entropy estimators that use **every n-th sample** (stride) from the input data, allowing you to analyze how entropy estimates change as the time interval between samples increases.

## Available Estimators
- `ml_lstm.py`: LSTM-based neural estimator
- `ml_rnn.py`: GRU-based neural estimator
- `lz.py`: LZ78-based entropy estimator
- `nist.py`: NIST min-entropy estimator

## Usage
Each script supports a `--stride` argument to control the sampling interval. The stride value is also included in the output filenames.

### Example
```bash
uv run -m scripts.estimators_8bits_stride.ml_lstm --stride 5
uv run -m scripts.estimators_8bits_stride.lz --stride 10
```

- Default stride is 2 (can also be set with the `ESTIMATOR_STRIDE` environment variable).
- Output files will be named like `bus_stride5.txt`, `radio_stride10.txt`, etc.

## Output
Results are written to the corresponding folder in `results/8bits_stride/<estimator>/`.

---

## Why Use Stride?
Using a stride allows you to investigate how entropy estimates change when you increase the interval between samples (e.g., to check for temporal correlations or memory effects in your data).

---

## See Also
- For standard estimators (using every sample), see `../estimate_8bits/`.
