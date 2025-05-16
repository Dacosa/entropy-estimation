import os
import numpy as np

sources = {
    'bus': 'processed_data/bus/8bits.bin',
    'radio': 'processed_data/radio/8bits.bin',
    'sismology': 'processed_data/sismology/8bits.bin',
    'ethereum': 'processed_data/ethereum/8bits.bin',
    'generated': 'processed_data/generated/8bits.bin',
}

RESULTS_DIR = os.path.join('results', '8bits_stride', 'nist')
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

def nist_min_entropy_estimate(seq):
    # NIST min-entropy estimator (most common value)
    if len(seq) == 0:
        return float('nan')
    counts = np.bincount(seq, minlength=256)
    p_max = np.max(counts) / len(seq)
    h_nist = -np.log2(p_max) if p_max > 0 else float('nan')
    return h_nist

for name, input_file in sources.items():
    print(f"\nProcessing source: {name}")
    try:
        seq = load_bin_file_as_int_sequence(input_file, stride=stride)
        N = len(seq)
        print(f"Loaded sequence of length {N} from {input_file} (stride={stride})")
        if N == 0:
            print("Empty sequence.")
            continue
        h_nist = nist_min_entropy_estimate(seq)
        result = (
            f"Length: {N}\n"
            f"Alphabet size: 256\n"
            f"Min-Entropy (bits/symbol): {h_nist:.6f}\n"
            f"Stride: {stride}\n"
        )
        print(result)
        out_name = f"{name}_stride{stride}.txt"
        with open(os.path.join(RESULTS_DIR, out_name), 'w', encoding='utf-8') as f:
            f.write(result)
    except Exception as e:
        print(f"Failed to process {name}: {e}")
