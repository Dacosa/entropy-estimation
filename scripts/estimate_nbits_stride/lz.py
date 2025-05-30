import os
import numpy as np
import argparse

def load_bin_file_as_int_sequence(filepath, stride=1, bits=8):
    with open(filepath, 'rb') as f:
        byte_arr = np.frombuffer(f.read(), dtype=np.uint8)
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
    dictionary = dict()
    w = []
    c = 1
    for k in seq:
        wk = tuple(w + [k])
        if wk not in dictionary:
            dictionary[wk] = c
            c += 1
            w = []
        else:
            w.append(k)
    if len(seq) == 0:
        return 0
    return len(dictionary) / (len(seq) / np.log2(len(seq)))

sources = {
    'bus': 'processed_data/bus/8bits.bin',
    'radio': 'processed_data/radio/8bits.bin',
    'sismology': 'processed_data/sismology/8bits.bin',
    'ethereum': 'processed_data/ethereum/8bits.bin',
    'generated': 'processed_data/generated/8bits.bin',
}

parser = argparse.ArgumentParser()
parser.add_argument('--stride', type=int, default=2, help='Stride for sampling (default: 2)')
parser.add_argument('--bits', type=int, default=16, help='Bits per symbol (default: 16)')
parser.add_argument('--source', type=str, default='all', help='Source to process (default: all)')
args = parser.parse_args()
stride = args.stride
bits = args.bits
source = args.source

if source == 'all':
    selected_sources = sources.items()
else:
    selected_sources = [(source, sources[source])]

for name, input_file in selected_sources:
    print(f"\nProcessing source: {name}")
    try:
        seq = load_bin_file_as_int_sequence(input_file, stride=stride, bits=bits)
        N = len(seq)
        alphabet_size = 2 ** bits
        lz_estimate = lz_entropy_estimate(seq)

        result = f"Length: {N}\nAlphabet size: {alphabet_size}\nLZ76 Complexity Count: {len(set(seq))}\nEntropy Estimate (manual): {lz_estimate:.6f}\nStride: {stride}\nBits per symbol: {bits}\n"
        print(result)
        out_dir = "results/nbits_stride/lz"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{name}_stride{stride}_bits{bits}.txt"), 'w') as f:
            f.write(result)
    except Exception as e:
        print(f"Failed to process {name}: {e}")
