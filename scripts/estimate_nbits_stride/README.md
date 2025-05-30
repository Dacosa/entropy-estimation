This folder contains versions of the entropy estimators that use **every n-th sample** (stride) from the input data, and support **arbitrary word length** (e.g. 16 bits per symbol).

## Available Estimators
- `ml_lstm.py`: LSTM-based neural estimator
- `ml_rnn.py`: GRU-based neural estimator
- `lz.py`: LZ78-based entropy estimator
- `ml_transformer.py`: Transformer-based neural estimator

## Usage
Each script supports a `--stride` argument to control the sampling interval, and a `--bits` argument to control word length (e.g., 16 bits per symbol). The stride and bits values are included in the output filenames.

### Example
```bash
uv run -m scripts.estimate_16bits_stride.ml_lstm --stride 5 --bits 16 --input myfile.bin
uv run -m scripts.estimate_16bits_stride.lz --stride 10 --bits 16 --input myfile.bin
```

- Default stride is 2.
- Default bits per symbol is 16.
- Output files will be named like `bus_stride5_bits16.txt`, `radio_stride10_bits16.txt`, etc.

## Output
Results are written to the corresponding folder in `results/16bits_stride/<estimator>/`.

---

## Why Use Stride?
Using a stride allows you to investigate how entropy estimates change when you increase the interval between samples (e.g., to check for temporal correlations or memory effects in your data).

---

## See Also
- For 8-bit estimators, see `../estimate_8bits_stride/`.
