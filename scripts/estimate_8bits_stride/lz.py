import os
import numpy as np

sources = {
    'bus': 'processed_data/bus/8bits.bin',
    'radio': 'processed_data/radio/8bits.bin',
    'sismology': 'processed_data/sismology/8bits.bin',
    'ethereum': 'processed_data/ethereum/8bits.bin',
    'generated': 'processed_data/generated/8bits.bin',
}

RESULTS_DIR = os.path.join('results', '8bits_stride', 'lz')
os.makedirs(RESULTS_DIR, exist_ok=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--stride', type=int, default=int(os.environ.get('ESTIMATOR_STRIDE', 2)), help='Stride for sampling (default: 2)')
args = parser.parse_args()
stride = args.stride

def load_bin_file_as_int_sequence(filepath, stride=1):
    with open(filepath, 'rb') as f:
        byte_arr = np.frombuffer(f.read(), dtype=np.uint8)
    return byte_arr[::stride]

def lz_entropy_estimate(seq):
    # Simple LZ78 entropy estimator
    n = len(seq)
    dictionary = dict()
    w = ()
    phrases = 0
    for symbol in seq:
        wc = w + (symbol,)
        if wc in dictionary:
            w = wc
        else:
            dictionary[wc] = True
            phrases += 1
            w = ()
    if n == 0:
        return float('nan')
    return phrases / n * np.log2(n) if n > 1 else float('nan')

for name, input_file in sources.items():
    print(f"\nProcessing source: {name}")
    try:
        seq = load_bin_file_as_int_sequence(input_file, stride=stride)
        N = len(seq)
        print(f"Loaded sequence of length {N} from {input_file} (stride={stride})")
        if N == 0:
            print("Empty sequence.")
            continue
        lz_complexity = lz_entropy_estimate(seq) * N / np.log2(N) if N > 1 else float('nan')
        entropy_estimate = lz_entropy_estimate(seq) if N > 1 else float('nan')
        result = (
            f"Length: {N}\n"
            f"Alphabet size: 256\n"
            f"LZ76 Complexity Count: {lz_complexity}\n"
            f"Entropy Estimate (manual): {entropy_estimate} bits/symbol\n"
            f"Stride: {stride}\n"
        )
        print(result)
        out_name = f"{name}_stride{stride}.txt"
        with open(os.path.join(RESULTS_DIR, out_name), 'w', encoding='utf-8') as f:
            f.write(result)
    except Exception as e:
        print(f"Failed to process {name}: {e}")
