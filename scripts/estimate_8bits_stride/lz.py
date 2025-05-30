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
parser.add_argument('--bits', type=int, default=8, help='Bits per symbol (default: 8)')
parser.add_argument('--source', type=str, default='all', choices=list(sources.keys()) + ['all'], help='Data source to use (default: all)')
args = parser.parse_args()
stride = args.stride
bits = args.bits
source = args.source

def load_bin_file_as_int_sequence(filepath, stride=1, bits=8):
    with open(filepath, 'rb') as f:
        byte_arr = np.frombuffer(f.read(), dtype=np.uint8)
    if bits == 8:
        return byte_arr[::stride]
    # Pack bits into symbols (for bits > 8)
    total_bits = len(byte_arr) * 8
    n_symbols = total_bits // bits
    if n_symbols == 0:
        return np.array([], dtype=np.uint32)
    all_bits = np.unpackbits(byte_arr)
    all_bits = all_bits[:n_symbols * bits]
    symbols = all_bits.reshape(-1, bits).dot(1 << np.arange(bits)[::-1])
    symbols = symbols[::stride]
    return symbols.astype(np.uint32)

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

if source == 'all':
    selected_sources = sources.items()
else:
    selected_sources = [(source, sources[source])]

for name, input_file in selected_sources:
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
