import os
import antropy as ant
import numpy as np

# Function to load a binary file as a sequence of 8-bit symbols (bytes)
def load_bin_file_as_bytes(filepath):
    with open(filepath, 'rb') as f:
        byte_arr = np.frombuffer(f.read(), dtype=np.uint8)
    # Convert to a string of bytes (each symbol is a value 0-255)
    return ''.join(chr(b) for b in byte_arr), byte_arr

sources = {
    'bus': 'processed_data/bus/8bits.bin',
    'radio': 'processed_data/radio/8bits.bin',
    'sismology': 'processed_data/sismology/8bits.bin',
    'ethereum': 'processed_data/ethereum/8bits.bin',
    'generated': 'processed_data/generated/8bits.bin',
}

RESULTS_DIR = os.path.join('results', '8bits', 'lz')
os.makedirs(RESULTS_DIR, exist_ok=True)

for name, input_file in sources.items():
    print(f'\nEstimating LZ entropy for source: {name}')
    try:
        seq_str, byte_arr = load_bin_file_as_bytes(input_file)
        lz_complexity = ant.lziv_complexity(seq_str, normalize=False)
        N = len(seq_str)
        # For 8-bit symbols, alphabet size is 256
        entropy_estimate = (lz_complexity * np.log2(N)) / N if N > 1 and lz_complexity > 0 else 0
        # Use antropy's normalization (which uses log_b(N), b=alphabet size)
        entropy_estimate_norm = ant.lziv_complexity(seq_str, normalize=True)
        result = (
            f'Length: {N}\n'
            f'Alphabet size: 256\n'
            f'LZ76 Complexity Count: {lz_complexity}\n'
            f'Entropy Estimate (manual): {entropy_estimate} bits/symbol\n'
            f'Entropy Estimate (antropy normalized): {entropy_estimate_norm} bits/symbol\n'
        )
        print(result)
        with open(os.path.join(RESULTS_DIR, f'{name}.txt'), 'w', encoding='utf-8') as f:
            f.write(result)
    except Exception as e:
        print(f'Failed to process {name}: {e}')
